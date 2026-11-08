"""pdfplumber-based PDF reader. Produces per-page text and tables."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: List[List[List[str]]]   # list of tables, each is rows of cells


@dataclass
class DocumentContent:
    path: Path
    pages: List[PageContent]

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def all_tables(self) -> List[List[List[str]]]:
        return [t for p in self.pages for t in p.tables]


def read_pdf(path: Path) -> DocumentContent:
    """Read a PDF file and return per-page text and tables."""
    logger.info("Reading PDF: %s", path.name)
    pages: List[PageContent] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            pages.append(PageContent(page_number=i, text=text, tables=tables))
    return DocumentContent(path=path, pages=pages)