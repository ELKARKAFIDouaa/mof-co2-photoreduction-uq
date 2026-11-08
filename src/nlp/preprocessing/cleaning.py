"""Text cleaning: headers, footers, hyphenation, control chars."""
from __future__ import annotations
import re

_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_WS   = re.compile(r"[ \t]+")
_HYPHEN_BRK = re.compile(r"(\w)-\n(\w)")


def clean_page_text(text: str) -> str:
    """Normalize text extracted from a single page."""
    if not text:
        return ""
    text = _CTRL_CHARS.sub(" ", text)
    text = _HYPHEN_BRK.sub(r"\1\2", text)     # join broken words
    text = _MULTI_WS.sub(" ", text)
    text = text.replace("\u00a0", " ")        # NBSP
    return text.strip()


def strip_repeated_lines(pages: list[str], min_repeat_ratio: float = 0.5) -> list[str]:
    """Remove headers/footers appearing on more than `min_repeat_ratio` of pages."""
    if len(pages) < 3:
        return pages
    from collections import Counter
    line_counts = Counter()
    for p in pages:
        for line in {l.strip() for l in p.splitlines() if l.strip()}:
            line_counts[line] += 1
    threshold = max(2, int(len(pages) * min_repeat_ratio))
    to_drop = {line for line, n in line_counts.items() if n >= threshold and len(line) < 120}
    return ["\n".join(l for l in p.splitlines() if l.strip() not in to_drop) for p in pages]