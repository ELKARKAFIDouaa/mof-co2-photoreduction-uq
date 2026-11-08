"""Detect Experimental / Results sections and isolate table-like blocks."""
from __future__ import annotations
import re
from typing import Dict

_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(Experimental|Methods?|Results?(?:\s+and\s+Discussion)?|"
    r"Discussion|Conclusion|Introduction|Supplementary)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Markers that begin the reference list — everything after is noise
_REFERENCES_MARKER = re.compile(
    r"(?:^|\n)\s*(?:\d+\.?\s*)?"
    r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY|"
    r"Références|RÉFÉRENCES|Reference list)\b",
    re.MULTILINE,
)

def _strip_references(text: str) -> str:
    """Remove everything after the reference section header."""
    m = _REFERENCES_MARKER.search(text)
    if m:
        return text[:m.start()]
    return text


def split_sections(text: str) -> Dict[str, str]:
    """Split raw text into named sections. Returns dict with lowercased section keys."""
    if not text:
        return {}

    # Cut references first — before any other processing
    text = _strip_references(text)

    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {"_full": text, "_primary": text}

    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = m.group(1).strip().lower().split()[0]
        sections.setdefault(name, "")
        sections[name] += "\n" + text[start:end].strip()

    if "experimental" in sections:
        sections["_primary"] = sections["experimental"]
    elif "results" in sections:
        sections["_primary"] = sections["results"]
    else:
        sections["_primary"] = text

    return sections