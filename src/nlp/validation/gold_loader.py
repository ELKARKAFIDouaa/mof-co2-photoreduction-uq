"""Load manually annotated gold JSON for evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List


def load_gold(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)