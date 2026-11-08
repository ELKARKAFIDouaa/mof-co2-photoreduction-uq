"""Crossref lookup for DOI → title, authors, year, journal."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger(__name__)
CROSSREF_URL = "https://api.crossref.org/works/{doi}"


def fetch_metadata(doi: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Fetch bibliographic metadata from Crossref. Returns empty dict on failure."""
    if not doi:
        return {}
    url = CROSSREF_URL.format(doi=doi)
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "mof-nlp-pipeline/1.0 (mailto:research@example.org)"})
        if r.status_code != 200:
            logger.warning("Crossref %s → HTTP %s", doi, r.status_code)
            return {}
        msg = r.json().get("message", {})
        return {
            "title": (msg.get("title") or [None])[0],
            "authors": [
                f"{a.get('given','')} {a.get('family','')}".strip()
                for a in msg.get("author", [])
            ],
            "year": (
                msg.get("published-print", {}).get("date-parts", [[None]])[0][0]
                or msg.get("published-online", {}).get("date-parts", [[None]])[0][0]
            ),
            "journal": (msg.get("container-title") or [None])[0],
        }
    except Exception as e:
        logger.warning("Crossref lookup failed for %s: %s", doi, e)
        return {}