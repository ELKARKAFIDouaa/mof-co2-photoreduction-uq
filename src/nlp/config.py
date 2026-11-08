"""
Central configuration: paths, seeds, pipeline version.
All other modules import from here to avoid magic strings.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import yaml

# ------------------------------------------------------------------ #
# Project paths
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(r"D:\Mes projets\mof-co2-ml")
SRC_DIR      = PROJECT_ROOT / "src" / "nlp"
RESOURCES    = SRC_DIR / "resources"

DATA_IN_DIR  = PROJECT_ROOT / "data" / "Eau g"           # PDFs to process
DATA_OUT_DIR = PROJECT_ROOT / "data" / "output"          # final JSON
LOG_DIR      = PROJECT_ROOT / "data" / "logs"

# ------------------------------------------------------------------ #
# Runtime
# ------------------------------------------------------------------ #
SEED = 42
PIPELINE_VERSION = "1.0.0"

# ------------------------------------------------------------------ #
# Extraction defaults
# ------------------------------------------------------------------ #
DEFAULT_CONFIDENCE = {
    "regex": 0.85,
    "table": 0.90,
    "gazetteer": 0.80,
    "linguistic": 0.65,
}

# Cross-source priority: when a field is extracted by multiple methods,
# the one with the highest priority wins (higher number = higher priority).
METHOD_PRIORITY = {
    "table": 4,
    "regex": 3,
    "gazetteer": 2,
    "linguistic": 1,
    "none": 0,
}

# Fields that are required for a "valid" record (missing fields are set to null).
REQUIRED_FIELDS = {"refcode", "doi"}

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def ensure_dirs() -> None:
    for d in (DATA_OUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ExtractorOutput:
    """
    Uniform output for every extractor.
    One instance == one extracted fact from one source.
    """
    field: str                    # e.g. "temperature_C", "mof.nom"
    value: Any                    # scalar or list
    unit: str | None = None       # raw unit before normalization
    source_sentence: str = ""     # sentence/context used to extract
    method: str = "none"          # regex | table | gazetteer | linguistic
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)