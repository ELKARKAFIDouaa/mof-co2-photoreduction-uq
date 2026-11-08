"""Rule-based linguistic extraction (Material + Property, performance class)."""
from __future__ import annotations
import re
from typing import List

from ..config import ExtractorOutput, DEFAULT_CONFIDENCE
from .base import BaseExtractor

_PROPERTY_NOUNS = re.compile(
    r"\b(stability|activity|selectivity|performance|efficiency|yield|crystallinity)\b",
    re.IGNORECASE,
)
_HIGH_WORDS = re.compile(
    r"\b(high(ly)?|enhanced|excellent|superior|outstanding)\b", re.IGNORECASE
)
_LOW_WORDS = re.compile(
    r"\b(low|poor|weak|reduced|inferior)\b", re.IGNORECASE
)

# Indices qu'une phrase appartient à une zone bibliographique ou est du bruit d'extraction PDF
_REF_MARKER = re.compile(
    r"\[\d+\]"                          # [11]
    r"|\(\d{4}\)"                       # (2017)
    r"|\b(?:Appl|J|Chem|Phys|Mater|Catal|Energy|Nanoscale|ACS|RSC|Angew|Nature|Science)\b\s*[\.\,]"
    r"|(?:\d+\s*\(\d{4}\)\s*\d+)",      # volume (année) page
    re.IGNORECASE,
)
_MAX_COMMAS = 3
_MAX_CHARS = 280


def _looks_like_reference(sent: str) -> bool:
    """Return True if `sent` looks like a bibliographic entry or extraction noise."""
    if not sent:
        return True
    if _REF_MARKER.search(sent):
        return True
    if sent.count(",") > _MAX_COMMAS:
        return True
    if len(sent) > _MAX_CHARS:
        return True
    # PDF extraction noise: broken chemical formulas, (cid:0) artefacts
    if "(cid:" in sent or re.search(r"\bCH\s+\d\b", sent):
        return True
    return False


class LinguisticExtractor(BaseExtractor):
    method_name = "linguistic"

    def extract(self, text: str, tables=None) -> List[ExtractorOutput]:
        results: List[ExtractorOutput] = []
        if not text:
            return results

        for m in _PROPERTY_NOUNS.finditer(text):
            s, e = _sentence_bounds(text, m.start())
            sent = text[s:e].strip()
            if _looks_like_reference(sent):
                continue

            if _HIGH_WORDS.search(sent):
                perf = "high"
            elif _LOW_WORDS.search(sent):
                perf = "low"
            else:
                continue

            results.append(ExtractorOutput(
                field="performance.classe_performance",
                value=perf,
                unit=None,
                source_sentence=sent,
                method=self.method_name,
                confidence=DEFAULT_CONFIDENCE["linguistic"],
            ))
        return results


def _sentence_bounds(text: str, idx: int, window: int = 200) -> tuple[int, int]:
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    s = text.rfind(".", start, idx)
    e = text.find(".", idx, end)
    return (s + 1 if s != -1 else start, e if e != -1 else end)