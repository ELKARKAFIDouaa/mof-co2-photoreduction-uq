"""
Evaluation focused on 11 critical fields.

Matching strategy (per your spec):
  1. Group by DOI (string-normalized).
  2. Within a DOI, pair predicted ↔ gold by:
       a) (metal, ligand, cocatalyst) exact match (all normalized), OR
       b) fuzzy similarity of `mof.nom` >= FUZZY_THRESHOLD.
  3. Each gold row is consumed at most once (one-to-one pairing).

Reporting:
  - Per field: n_ref (number of matched gold rows with a non-null value),
    precision, recall, F1.
  - Coverage: |matched gold rows| / |gold rows| (overall and per-field).
  - 0/0 → None ("n/a") instead of 0.0.
  - Numeric values: |a-b| <= max(0.01, 0.05*|b|).
  - String values: normalized (case, hyphens, underscores, prefix order).
"""
from __future__ import annotations
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
# 11 critical fields: name shown in report -> key path in gold JSON
# ------------------------------------------------------------------ #
TARGET_FIELDS: Dict[str, str] = {
    "Cocatalyst type":          "cocatalyseur.type",
    "Ligand":                   "mof.ligand",
    "Metal":                    "mof.metal",
    "Band gap (eV)":            "proprietes_electroniques.band_gap_eV",
    "ΔLUMO CH₄ (eV)":           "proprietes_electroniques.LUMO_eV_vs_NHE",
    "Cocatalyst charge (%)":    "cocatalyseur.charge_pct",
    "Catalyst conc. (g/L)":     "conditions_reaction.concentration_catalyseur_g_L",
    "BET × irradiation time":   "_derived.BET_times_time",
    "Solvent":                  "conditions_reaction.solvant",
    "BET surface (m²/g)":       "proprietes_structurales.BET_m2_g",
    "Light source":             "conditions_reaction.source_lumiere",
    "Irradiation duration (h)": "conditions_reaction.duree_irradiation_h",
}

_NUM_ABS_TOL = 0.01
_NUM_REL_TOL = 0.05
FUZZY_THRESHOLD = 0.75      # for mof.nom similarity


# ------------------------------------------------------------------ #
# Normalization
# ------------------------------------------------------------------ #
_WS = re.compile(r"\s+")
_SEP = re.compile(r"[\s_\-·]+")
_PREFIX_RE = re.compile(r"^(nh2|oh|ch3|no2|so3h|cooh|f|cl|br)[\-\s]?", re.I)
_SUFFIX_RE = re.compile(r"[\-\s]?(nh2|oh|ch3|no2|so3h|cooh|f|cl|br)$", re.I)


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    s = _WS.sub(" ", str(v).strip().lower())
    s = _SEP.sub(" ", s)
    return s.strip()


def _norm_name(name: Any) -> str:
    """Normalize MOF names: `NH2-UiO-66` == `UiO-66-NH2`."""
    s = _norm_str(name)
    if not s:
        return ""
    # If prefix present, move it to suffix
    m = _PREFIX_RE.match(s)
    if m:
        token = m.group(1)
        rest = s[m.end():].strip()
        s = f"{rest} {token}".strip()
    # Remove separators to unify "uio 66" vs "uio66"
    return s.replace(" ", "")


def _norm_value(v: Any) -> Any:
    if isinstance(v, str):
        return _norm_str(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _values_equal(pv: Any, gv: Any) -> bool:
    if pv is None and gv is None:
        return True
    if pv is None or gv is None:
        return False
    if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
        gvf, pvf = float(gv), float(pv)
        if math.isnan(gvf) or math.isnan(pvf):
            return False
        diff = abs(pvf - gvf)
        return diff <= max(_NUM_ABS_TOL, _NUM_REL_TOL * abs(gvf))
    return pv == gv


def _fuzzy(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ------------------------------------------------------------------ #
# Access nested keys (dict) + derived fields
# ------------------------------------------------------------------ #
def _get_path(rec: Dict, path: str) -> Any:
    if path.startswith("_derived."):
        return _derive(rec, path.split(".", 1)[1])
    cur: Any = rec
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _derive(rec: Dict, name: str) -> Any:
    if name == "BET_times_time":
        bet = _get_path(rec, "proprietes_structurales.BET_m2_g")
        dur = _get_path(rec, "conditions_reaction.duree_irradiation_h")
        if isinstance(bet, (int, float)) and isinstance(dur, (int, float)):
            return float(bet) * float(dur)
        return None
    return None


# ------------------------------------------------------------------ #
# Matching: DOI first, then within-DOI key or fuzzy name
# ------------------------------------------------------------------ #
def _match_tuple(rec: Dict) -> Tuple[str, str, str]:
    metal = _norm_str(_get_path(rec, "mof.metal"))
    ligand = _norm_str(_get_path(rec, "mof.ligand"))
    cocat = _norm_str(_get_path(rec, "cocatalyseur.type"))
    return metal, ligand, cocat


def _pair_within_doi(preds: List[Dict], golds: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    Within one DOI group, pair by:
      Pass 1: exact (metal, ligand, cocatalyst) triple
      Pass 2: fuzzy mof.nom >= FUZZY_THRESHOLD
    Each gold consumed once.
    """
    pairs: List[Tuple[Dict, Dict]] = []
    used_gold = set()
    remaining_preds: List[Dict] = []

    # ---- Pass 1: exact triple
    for p in preds:
        t = _match_tuple(p)
        if not any(t):
            remaining_preds.append(p)
            continue
        found = False
        for i, g in enumerate(golds):
            if i in used_gold:
                continue
            if _match_tuple(g) == t:
                pairs.append((p, g))
                used_gold.add(i)
                found = True
                break
        if not found:
            remaining_preds.append(p)

    # ---- Pass 2: fuzzy name
    still_remaining: List[Dict] = []
    for p in remaining_preds:
        p_name = _norm_name(_get_path(p, "mof.nom"))
        best_i, best_score = -1, 0.0
        for i, g in enumerate(golds):
            if i in used_gold:
                continue
            g_name = _norm_name(_get_path(g, "mof.nom"))
            score = _fuzzy(p_name, g_name)
            if score > best_score:
                best_score, best_i = score, i
        if best_i >= 0 and best_score >= FUZZY_THRESHOLD:
            pairs.append((p, golds[best_i]))
            used_gold.add(best_i)
        else:
            still_remaining.append(p)

    return pairs


def _match_all(predicted: List[Dict], gold: List[Dict]) -> List[Tuple[Dict, Dict]]:
    # Group by DOI
    pred_by_doi: Dict[str, List[Dict]] = defaultdict(list)
    gold_by_doi: Dict[str, List[Dict]] = defaultdict(list)
    for p in predicted:
        d = _norm_str(p.get("doi") or "")
        if d:
            pred_by_doi[d].append(p)
    for g in gold:
        d = _norm_str(g.get("doi") or "")
        if d:
            gold_by_doi[d].append(g)

    pairs: List[Tuple[Dict, Dict]] = []
    for doi in set(pred_by_doi) & set(gold_by_doi):
        pairs.extend(_pair_within_doi(pred_by_doi[doi], gold_by_doi[doi]))
    return pairs


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def evaluate(predicted: List[Dict], gold: List[Dict]) -> Dict[str, Any]:
    pairs = _match_all(predicted, gold)
    matched_gold_ids = {id(g) for _, g in pairs}

    # Per-field counters
    stats: Dict[str, Dict[str, int]] = {k: {"tp": 0, "fp": 0, "fn": 0, "n_ref": 0}
                                        for k in TARGET_FIELDS}

    for pred, g in pairs:
        for label, path in TARGET_FIELDS.items():
            gv = _norm_value(_get_path(g, path))
            pv = _norm_value(_get_path(pred, path))

            if gv is not None:
                stats[label]["n_ref"] += 1
            if gv is None:
                continue

            if _values_equal(pv, gv):
                stats[label]["tp"] += 1
            elif pv is None:
                stats[label]["fn"] += 1
            else:
                stats[label]["fp"] += 1

    # Per-field coverage: how many gold rows (with value) are matched
    gold_coverage = _gold_coverage(gold, matched_gold_ids)

    report: Dict[str, Any] = {
        "_gold_records": len(gold),
        "_matched_gold_rows": len(matched_gold_ids),
        "_coverage_gold_rows": len(matched_gold_ids) / len(gold) if gold else 0.0,
    }

    for label, s in stats.items():
        tp, fp, fn, n_ref = s["tp"], s["fp"], s["fn"], s["n_ref"]

        if n_ref == 0:
            report[label] = {
                "n_ref": 0,
                "coverage": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            continue

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        report[label] = {
            "n_ref": n_ref,
            "coverage": gold_coverage.get(label, 0.0),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return report


def _gold_coverage(gold: List[Dict], matched_ids: set) -> Dict[str, float]:
    """Fraction of gold rows (having a value for the field) that were matched."""
    counts_total: Dict[str, int] = defaultdict(int)
    counts_matched: Dict[str, int] = defaultdict(int)
    for g in gold:
        for label, path in TARGET_FIELDS.items():
            gv = _get_path(g, path)
            if gv is not None:
                counts_total[label] += 1
                if id(g) in matched_ids:
                    counts_matched[label] += 1
    return {
        label: (counts_matched[label] / counts_total[label])
        if counts_total[label] else 0.0
        for label in TARGET_FIELDS
    }