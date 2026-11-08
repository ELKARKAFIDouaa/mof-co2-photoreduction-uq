"""Full pipeline: PDF → text → extraction → merge → JSON."""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    DATA_IN_DIR, DATA_OUT_DIR, LOG_DIR, DEFAULT_CONFIDENCE,
    ExtractorOutput, ensure_dirs, REQUIRED_FIELDS,
)
from .io.pdf_reader import read_pdf
from .io.metadata import fetch_metadata
from .io.writer import write_json, log_extraction
from .preprocessing.cleaning import clean_page_text, strip_repeated_lines
from .preprocessing.sectioning import split_sections
from .extractors.regex_extractor import RegexExtractor
from .extractors.gazetteer_extractor import GazetteerExtractor
from .extractors.linguistic_extractor import LinguisticExtractor
from .extractors.table_extractor import TableExtractor
from .merging.resolve import merge_extractions, nest_record
from .validation.gold_loader import load_gold
from .validation.metrics import evaluate
from .validation.error_report import write_error_report

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Regex constants
# ------------------------------------------------------------------ #
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_MOF_NAME_RE = re.compile(
    r"\b((?:UiO|ZIF|MIL|HKUST|PCN|NU|ACOF|N\d?-COF)[\w\-\.]*|"
    r"[A-Z]{2,6}-\d+[\w\-\.]*)\b"
)


# ------------------------------------------------------------------ #
# Helpers: DOI, refcode, quality
# ------------------------------------------------------------------ #
def _extract_doi(text: str) -> Optional[str]:
    """Fallback DOI detection from raw text if Crossref lookup is unavailable."""
    if not text:
        return None
    m = _DOI_RE.search(text)
    return m.group(0).rstrip(".,;") if m else None


def _extract_mof_name_from_text(text: str) -> Optional[str]:
    """Best-effort MOF/COF name extraction (used when gazetteer missed it)."""
    if not text:
        return None
    m = _MOF_NAME_RE.search(text)
    return m.group(1) if m else None
def _is_clean_note(note: str) -> bool:
    if not note or len(note) > 250:
        return False
    if "(cid:" in note or re.search(r"\[\d+\]", note):
        return False
    if note.count(",") > 3:
        return False
    return True

def build_refcode(author_metadata: Dict[str, Any],
                  mof_name: Optional[str],
                  fallback_author: str = "Unknown",
                  fallback_year: str = "XXXX") -> str:
    """
    Convention: <FirstAuthor><Year>_<MOFName>
    Examples: Fu2018_ACOF-1, Chen2020_1.9Bi2S3-UiO66
    """
    authors = author_metadata.get("authors") or []
    if authors:
        last_token = authors[0].strip().split()[-1] if authors[0].strip() else fallback_author
    else:
        last_token = fallback_author
    last_token = re.sub(r"[^A-Za-z\-]", "", last_token) or fallback_author

    year = author_metadata.get("year") or fallback_year
    year = str(year)

    mof = (mof_name or "NA")
    mof = re.sub(r"\s+", "", mof)
    mof = re.sub(r"[^\w\-\u00b5%\.]", "", mof) or "NA"

    return f"{last_token}{year}_{mof}"


def build_quality_block(record: Dict[str, Any],
                        extractions: List[ExtractorOutput]) -> Dict[str, Any]:
    """Build the `qualite` block exactly as in json3.json."""
    checks = {
        "volume_pores":   "proprietes_structurales.volume_pores_cm3_g",
        "PLD":            "proprietes_structurales.PLD_angstrom",
        "AQE":            "performance.AQE_pct",
        "H2_concurrent":  "performance.H2_concurrent_umol_g_h",
        "band_gap":       "proprietes_electroniques.band_gap_eV",
        "selectivity":    "performance.selectivite_pct",
    }
    missing = [label for label, path in checks.items() if _get(record, path) is None]

    notes_sources = [
    e.source_sentence[:250].strip()
    for e in extractions
    if e.method == "linguistic"
    and e.source_sentence
    and _is_clean_note(e.source_sentence)
    ]
    notes = " ; ".join(dict.fromkeys(notes_sources))  # dedupe, preserve order

    return {"features_manquantes": missing, "notes": notes}


def _get(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


# ------------------------------------------------------------------ #
# Single-PDF processing
# ------------------------------------------------------------------ #
def process_pdf(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """Run the full 6-layer pipeline on one PDF and return a structured record."""
    logger.info("── Processing %s", pdf_path.name)

    # ── Layer 1: ingestion ─────────────────────────────────────────
    doc = read_pdf(pdf_path)
    if not doc.pages:
        logger.warning("No pages extracted from %s", pdf_path.name)
        return None

    # ── Layer 2: preprocessing ─────────────────────────────────────
    raw_pages = [clean_page_text(p.text) for p in doc.pages]
    cleaned_pages = strip_repeated_lines(raw_pages)
    full_text = "\n".join(cleaned_pages)
    sections = split_sections(full_text)
    primary_text = sections.get("_primary", full_text) or full_text
    tables = doc.all_tables

    # ── Layer 3: extraction ────────────────────────────────────────
    extractors = [
        RegexExtractor(),
        GazetteerExtractor(),
        LinguisticExtractor(),
        TableExtractor(),
    ]
    extractions: List[ExtractorOutput] = []
    for ex in extractors:
        try:
            extractions.extend(ex.extract(primary_text, tables))
        except Exception as e:
            logger.exception("Extractor %s failed on %s: %s",
                             ex.method_name, pdf_path.name, e)

    if not extractions:
        logger.warning("No facts extracted from %s", pdf_path.name)
        return None

    # ── Layer 5a: fusion ───────────────────────────────────────────
    flat = merge_extractions(extractions)          # {field: best_value}
    record = nest_record(flat)                     # nested dict

    # ── Identity: DOI + refcode ────────────────────────────────────
    doi = flat.get("doi") or _extract_doi(full_text)
    meta = fetch_metadata(doi) if doi else {}

    mof_nom = _get(record, "mof.nom") or _extract_mof_name_from_text(full_text)
    if mof_nom and not _get(record, "mof.nom"):
        _set(record, "mof.nom", mof_nom)

    refcode = build_refcode(meta, mof_nom)
    record.setdefault("refcode", refcode)
    record.setdefault("doi", doi)

    # ── Ensure required top-level keys exist (schema conformance) ──
    for k in ("mof", "cocatalyseur", "conditions_reaction",
              "proprietes_structurales", "proprietes_electroniques",
              "performance", "stabilite_photocatalytique"):
        record.setdefault(k, {})

    # ── Layer 6a: quality block ────────────────────────────────────
    record["qualite"] = build_quality_block(record, extractions)

    # ── Traceability log ───────────────────────────────────────────
    log_extraction(
        pdf_path, LOG_DIR / "extraction.log",
        {
            "refcode": record["refcode"],
            "doi": record["doi"],
            "fields": sorted(flat.keys()),
            "n_extractions": len(extractions),
            "methods": sorted({e.method for e in extractions}),
            "features_manquantes": record["qualite"]["features_manquantes"],
        },
    )
    return record


# ------------------------------------------------------------------ #
# Corpus-level orchestration
# ------------------------------------------------------------------ #
def run(input_dir: Path = DATA_IN_DIR,
        output_file: Path = DATA_OUT_DIR / "dataset.json",
        gold_file: Optional[Path] = DATA_OUT_DIR / "gold.json") -> List[Dict[str, Any]]:
    """Run the pipeline over every PDF in `input_dir`."""
    ensure_dirs()
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found in %s", input_dir)
        return []

    logger.info("Found %d PDF(s) in %s", len(pdf_paths), input_dir)
    records: List[Dict[str, Any]] = []
    for pdf in pdf_paths:
        try:
            rec = process_pdf(pdf)
            if rec is not None:
                records.append(rec)
        except Exception as e:
            logger.exception("Failed processing %s: %s", pdf.name, e)

    # Deduplicate on (refcode, doi): keep first complete record
    records = _dedupe_records(records)

    # ── Layer 6b: write output ─────────────────────────────────────
    write_json(records, output_file)
    logger.info("✔ Wrote %d records → %s", len(records), output_file)

    # ── Layer 6c: evaluation (optional) ────────────────────────────
    if gold_file and gold_file.exists():
        gold = load_gold(gold_file)
        report = evaluate(records, gold)
        write_json(report, output_file.with_name("eval_report.json"))
        write_error_report(records, gold, output_file.with_name("error_report.txt"))
        logger.info("✔ Evaluation written next to %s", output_file.name)

    return records


def _dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one record per (refcode, doi); prefer the one with fewer missing features."""
    best: Dict[tuple, Dict[str, Any]] = {}
    for r in records:
        key = (r.get("refcode"), r.get("doi"))
        n_missing = len(r.get("qualite", {}).get("features_manquantes", []))
        if key not in best or n_missing < len(best[key].get("qualite", {})
                                              .get("features_manquantes", [])):
            best[key] = r
    return list(best.values())


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="MOF/CO2 NLP pipeline (PDF → JSON)")
    parser.add_argument("--input",  type=Path, default=DATA_IN_DIR,
                        help=f"Directory containing PDFs (default: {DATA_IN_DIR})")
    parser.add_argument("--output", type=Path, default=DATA_OUT_DIR / "dataset.json",
                        help="Output JSON file")
    parser.add_argument("--gold",   type=Path, default=DATA_OUT_DIR / "gold.json",
                        help="Optional gold-standard JSON for evaluation")
    args = parser.parse_args()

    run(input_dir=args.input, output_file=args.output, gold_file=args.gold)