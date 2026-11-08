"""Regex-based numeric extraction (temperature, pressure, BET, band gap, ...)."""
from __future__ import annotations
import re
from pathlib import Path
from typing import List

from ..config import ExtractorOutput, RESOURCES, DEFAULT_CONFIDENCE, load_yaml
from ..normalization.units import convert
from .base import BaseExtractor




_FIELD_TO_FAMILY = {
    "conditions_reaction.temperature_C": "temperature",
    "conditions_reaction.pression_CO2_bar": "pressure",
    "conditions_reaction.duree_irradiation_h": "time",
    "proprietes_structurales.BET_m2_g": "surface_area",
    "proprietes_electroniques.band_gap_eV": "band_gap",
    "proprietes_electroniques.HOMO_eV_vs_NHE": "band_gap",
    "proprietes_electroniques.LUMO_eV_vs_NHE": "band_gap",
    "performance.taux_production_umol_g_h": "production_rate",
}
class RegexExtractor(BaseExtractor):
    method_name = "regex"

    def __init__(self):
        self._patterns = self._load_patterns()

    def _load_patterns(self) -> list[dict]:
        pats = []
        for yml in sorted((RESOURCES / "patterns").glob("*.yaml")):
            for entry in load_yaml(yml) or []:
                pats.append({
                    "field": entry["name"],
                    "regex": re.compile(entry["regex"], re.IGNORECASE),
                    "unit": entry.get("unit"),
                })
        return pats

    def extract(self, text: str, tables=None) -> List[ExtractorOutput]:
        results: List[ExtractorOutput] = []
        if not text:
            return results
        for pat in self._patterns:
            for m in pat["regex"].finditer(text):
                try:
                    raw_val = float(m.group("value").replace(",", "."))
                except (ValueError, IndexError):
                    continue
                # choose normalization family based on field prefix
                family = _FIELD_TO_FAMILY.get(pat["field"], None)
                normalized = convert(raw_val, pat["unit"], family) if (pat["unit"] and family) else raw_val
                # find enclosing sentence for provenance
                s, e = _sentence_bounds(text, m.start())
                results.append(ExtractorOutput(
                    field=pat["field"],
                    value=normalized,
                    unit=pat["unit"],
                    source_sentence=text[s:e].strip(),
                    method=self.method_name,
                    confidence=DEFAULT_CONFIDENCE["regex"],
                ))
        return results


def _sentence_bounds(text: str, idx: int, window: int = 200) -> tuple[int, int]:
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    s = text.rfind(".", start, idx)
    e = text.find(".", idx, end)
    return (s + 1 if s != -1 else start, e if e != -1 else end)