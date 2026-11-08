"""Dictionary-based extraction of categorical features."""
from __future__ import annotations
import re
from typing import Dict, List

from ..config import ExtractorOutput, RESOURCES, DEFAULT_CONFIDENCE
from ..normalization.terms import normalize_term
from .base import BaseExtractor

_GAZ_MAP: Dict[str, str] = {
    "metals.txt": "mof.metal",
    "ligands.txt": "mof.ligand",
    "light_sources.txt": "conditions_reaction.source_lumiere",
    "cocatalysts.txt": "cocatalyseur.type",
    "gases.txt": "conditions_reaction.gaz",
    "products.txt": "performance.produit_majoritaire",
    "structure_types.txt": "mof.type_structure",
    "families.txt": "mof.famille",
}

# Chemistry-consistent metal per MOF family (used to override false positives)
_FAMILY_TO_METAL = {
    "uio": "zr",
    "zif": "zn",
    "mil-100": "cr",
    "mil-101": "cr",
    "mil-53": "al",
    "mil-88": "fe",
    "hkust": "cu",
    "pcn": "zr",
    "nu": "zr",
}

# Cocatalyst must appear with a contextual cue (avoids false positives from "Ru(bpy)3" etc.)
_COCAT_CONTEXT = re.compile(
    r"\b(?:loaded|doped|deposited|modified|with|containing|anchored|"
    r"cocatalyst|co-catalyst|nanoparticles?|NPs?)\b",
    re.IGNORECASE,
)


class GazetteerExtractor(BaseExtractor):
    method_name = "gazetteer"

    def __init__(self):
        self._entries: Dict[str, list[tuple[str, re.Pattern]]] = {}
        for fname, field in _GAZ_MAP.items():
            path = RESOURCES / "gazetteers" / fname
            if not path.exists():
                continue
            terms = [t.strip() for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
            self._entries[field] = [
                (term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
                for term in terms
            ]

    def extract(self, text: str, tables=None) -> List[ExtractorOutput]:
        results: List[ExtractorOutput] = []
        if not text:
            return results

        # ── Step 1: extract all fields with the gazetteers ───────────
        for field, entries in self._entries.items():
            for term, pat in entries:
                m = pat.search(text)
                if not m:
                    continue

                # Context filter for cocatalysts
                if field == "cocatalyseur.type" and term.lower() != "none":
                    window = text[max(0, m.start() - 100): min(len(text), m.end() + 100)]
                    if not _COCAT_CONTEXT.search(window):
                        continue

                s, e = _sentence_bounds(text, m.start())
                results.append(ExtractorOutput(
                    field=field,
                    value=normalize_term(term),
                    unit=None,
                    source_sentence=text[s:e].strip(),
                    method=self.method_name,
                    confidence=DEFAULT_CONFIDENCE["gazetteer"],
                ))

        # ── Step 2: post-process — enforce metal/family consistency ──
        family = next((r.value for r in results if r.field == "mof.famille"), None)
        if family and str(family).lower() in _FAMILY_TO_METAL:
            expected = _FAMILY_TO_METAL[str(family).lower()]
            results = [r for r in results if r.field != "mof.metal"]
            results.append(ExtractorOutput(
                field="mof.metal",
                value=expected,
                unit=None,
                source_sentence=f"(inferred from family {family})",
                method=self.method_name,
                confidence=DEFAULT_CONFIDENCE["gazetteer"] - 0.1,
            ))

        return results


def _sentence_bounds(text: str, idx: int, window: int = 200) -> tuple[int, int]:
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    s = text.rfind(".", start, idx)
    e = text.find(".", idx, end)
    return (s + 1 if s != -1 else start, e if e != -1 else end)