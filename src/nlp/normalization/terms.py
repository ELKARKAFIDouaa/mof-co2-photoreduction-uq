"""Unify synonyms for categorical terms."""
from __future__ import annotations
import re

_SYNONYMS = {
    "visible light": "visible_light",
    "vis": "visible_light",
    "uv": "UV_light",
    "uv-vis": "UV_light",
    "xe lamp": "xenon_lamp",
    "xenon lamp": "xenon_lamp",
    "300 w xe lamp": "xenon_300W",
    "am 1.5g": "AM1.5G",
    "solar simulator": "solar_simulator",
    "pristine": "pristine",
    "composite": "composite",
    "composites": "composite",
}


def normalize_term(term: str) -> str:
    if term is None:
        return ""
    key = re.sub(r"\s+", " ", term.strip().lower())
    return _SYNONYMS.get(key, key)