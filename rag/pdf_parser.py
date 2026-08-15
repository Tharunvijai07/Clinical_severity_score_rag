"""
rag/pdf_parser.py
─────────────────
Extracts text from PDF files using PyMuPDF (fitz).

Supports both:
  • Uploaded patient lab reports (file-like objects from Streamlit)
  • Knowledge-base guideline PDFs (file paths on disk)
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF

from utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(source: Union[str, Path, bytes, io.BytesIO]) -> str:
    """
    Extract and return all text from a PDF.

    Parameters
    ----------
    source : str | Path | bytes | BytesIO
        • str / Path  → file path on disk
        • bytes       → raw PDF bytes (e.g. from Streamlit uploader.read())
        • BytesIO     → in-memory PDF stream

    Returns
    -------
    str
        Full extracted text, pages separated by form-feed characters.
    """
    try:
        if isinstance(source, (str, Path)):
            doc = fitz.open(str(source))
            logger.info(f"Opened PDF from path: {source}  ({doc.page_count} pages)")
        elif isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
            logger.info(f"Opened PDF from bytes ({doc.page_count} pages)")
        elif isinstance(source, io.BytesIO):
            doc = fitz.open(stream=source.read(), filetype="pdf")
            logger.info(f"Opened PDF from BytesIO ({doc.page_count} pages)")
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        pages: list[str] = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {page_num}]\n{text.strip()}")

        doc.close()

        full_text = "\n\n".join(pages)
        logger.info(f"Extracted {len(full_text):,} characters across {len(pages)} non-empty pages.")
        return full_text

    except Exception as exc:
        logger.error(f"PDF extraction failed: {exc}")
        raise


def extract_text_from_txt(source: Union[str, Path]) -> str:
    """Fallback: read plain-text knowledge-base files (.txt)."""
    path = Path(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    logger.info(f"Read text file: {path.name}  ({len(text):,} characters)")
    return text
