"""Extract numeric values from pdfplumber tables (BET, band gap, rates)."""
from __future__ import annotations
import re
from typing import List

from ..config import ExtractorOutput, DEFAULT_CONFIDENCE
from .base import BaseExtractor

_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


class TableExtractor(BaseExtractor):
    method_name = "table"

    def extract(self, text: str, tables=None) -> List[ExtractorOutput]:
        results: List[ExtractorOutput] = []
        if not tables:
            return results
        for table in tables:
            if not table or len(table) < 2:
                continue
            header = [str(c or "").strip().lower() for c in table[0]]
            for row in table[1:]:
                if not row:
                    continue
                for i, cell in enumerate(row):
                    if cell is None or i >= len(header):
                        continue
                    field = _map_header(header[i])
                    if not field:
                        continue
                    m = _NUM.search(str(cell))
                    if not m:
                        continue
                    try:
                        val = float(m.group().replace(",", "."))
                    except ValueError:
                        continue
                    results.append(ExtractorOutput(
                        field=field,
                        value=val,
                        unit=None,
                        source_sentence=f"{header[i]}: {cell}",
                        method=self.method_name,
                        confidence=DEFAULT_CONFIDENCE["table"],
                    ))
        return results


_HEADER_MAP = {
    "bet": "proprietes_structurales.BET_m2_g",
    "surface area": "proprietes_structurales.BET_m2_g",
    "band gap": "proprietes_electroniques.band_gap_eV",
    "homo": "proprietes_electroniques.HOMO_eV_vs_NHE",
    "lumo": "proprietes_electroniques.LUMO_eV_vs_NHE",
    "selectivity": "performance.selectivite_pct",
    "rate": "performance.taux_production_umol_g_h",
    "temperature": "conditions_reaction.temperature_C",
    "pressure": "conditions_reaction.pression_CO2_bar",
    "time": "conditions_reaction.duree_irradiation_h",
}


def _map_header(h: str) -> str | None:
    for key, field in _HEADER_MAP.items():
        if key in h:
            return field
    return None