"""Unit conversion to standard units."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from ..config import RESOURCES, load_yaml

_UNITS = load_yaml(RESOURCES / "units.yaml")


def convert(value: float, from_unit: str, to_family: str) -> Optional[float]:
    """Convert `value` from `from_unit` into the standard unit of `to_family`."""
    try:
        spec = _UNITS[to_family][from_unit]
    except (KeyError, TypeError):
        return value
    return (value + spec.get("offset", 0)) * spec.get("factor", 1)