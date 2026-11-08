"""Human-readable error report, aligned with metrics v3."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

from .metrics import (
    TARGET_FIELDS, _match_all, _get_path, _norm_value, _values_equal,
)


def write_error_report(predicted: List[Dict], gold: List[Dict], out_path: Path) -> None:
    pairs = _match_all(predicted, gold)
    matched_gold_ids = {id(g) for _, g in pairs}

    lines: List[str] = []
    # Unmatched gold rows
    for g in gold:
        if id(g) not in matched_gold_ids:
            lines.append(
                f"[UNMATCHED GOLD] ref={g.get('refcode')!r} "
                f"doi={g.get('doi')!r} mof={_get_path(g, 'mof.nom')!r}"
            )

    # Field-level discrepancies on matched pairs
    for pred, g in pairs:
        for label, path in TARGET_FIELDS.items():
            gv = _norm_value(_get_path(g, path))
            if gv is None:
                continue
            pv = _norm_value(_get_path(pred, path))
            if not _values_equal(pv, gv):
                lines.append(
                    f"[{g.get('refcode')}] {label}: gold={gv!r} pred={pv!r}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")