"""Merge multiple extraction outputs into one record per (refcode, doi)."""
from __future__ import annotations
from typing import Any, Dict, List
from ..config import ExtractorOutput, METHOD_PRIORITY


def merge_extractions(extractions: List[ExtractorOutput]) -> Dict[str, Any]:
    """
    For each field, choose the value from the highest-priority method.
    Ties broken by confidence.
    """
    by_field: Dict[str, List[ExtractorOutput]] = {}
    for ex in extractions:
        by_field.setdefault(ex.field, []).append(ex)

    merged: Dict[str, Any] = {}
    for field, exs in by_field.items():
        exs.sort(key=lambda e: (METHOD_PRIORITY.get(e.method, 0), e.confidence), reverse=True)
        merged[field] = exs[0].value
    return merged


def nest_record(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dotted keys ('mof.metal') into nested dicts."""
    out: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        cursor = out
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = value
    return out