"""Abstract base for all extractors."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from ..config import ExtractorOutput


class BaseExtractor(ABC):
    """Every extractor returns a list of ExtractorOutput."""

    method_name: str = "none"

    @abstractmethod
    def extract(self, text: str, tables: List[list] | None = None) -> List[ExtractorOutput]:
        """Run extraction over a text (and optionally tables)."""
        ...