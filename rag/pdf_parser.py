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
import re
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF

from utils.logger import get_logger

logger = get_logger(__name__)

_PERSONAL_INFO_RE = re.compile(
    r"(?i)\b(patient|name|age|sex|gender|dob|date of birth|birth date|mrn|medical record|record number|address|phone|mobile|email|doctor|physician|attending|room|bed|unit|ward|admission|discharge|encounter|visit)\b"
)

_MEDICAL_LABEL_RE = re.compile(
    r"(?i)\b(creatinine|bun|blood urea nitrogen|potassium|sodium|chloride|bicarbonate|hco3|hemoglobin|hgb|wbc|platelets|alt|ast|bilirubin|inr|lactate|ph|oxygen saturation|spo2|bp|blood pressure|systolic|diastolic|pulse|temperature|resp|respiratory|glucose|albumin|prothrombin|troponin|ammonia|calcium|magnesium|urinalysis|protein|ketones|anion gap)\b"
)


def sanitize_clinical_report_text(text: str) -> str:
    """Keep only clinical lab values and discard personal/administrative metadata."""
    if not text or not text.strip():
        return ""

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        # Drop known patient-identifying metadata.
        if _PERSONAL_INFO_RE.search(line):
            line = re.sub(
                r"(?i)\b(?:patient|name|age|sex|gender|dob|date of birth|birth date|mrn|medical record|record number|address|phone|mobile|email|doctor|physician|attending|room|bed|unit|ward|admission|discharge|encounter|visit)\b\s*[:=-]?\s*.*",
                "",
                line,
            ).strip()
            if not line:
                continue

        # Keep only lines that look like lab/value data, not free-form notes.
        if not re.search(r"\d", line):
            continue

        if _MEDICAL_LABEL_RE.search(line):
            cleaned_lines.append(line)
            continue

        if re.search(r"(?i)\b[A-Za-z][A-Za-z\-/() .]{0,35}\s*[:=]\s*<?\d+(?:\.\d+)?\b", line):
            cleaned_lines.append(line)
            continue

        if re.search(r"(?i)\b\d+(?:\.\d+)?\s*(?:mg/dl|mg/dl|mmol/l|g/dl|mmhg|%|u/l|iu/l|meq/l|ng/ml|pg/ml|k|fl|bpm|°c|c|mmol|g/l)\b", line):
            cleaned_lines.append(line)
            continue

    # Remove repeated or noisy labels and normalize spacing.
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


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
        sanitized = sanitize_clinical_report_text(full_text)
        logger.info(
            f"Extracted {len(full_text):,} chars across {len(pages)} pages; "
            f"kept {len(sanitized):,} chars after filtering personal metadata."
        )
        return sanitized

    except Exception as exc:
        logger.error(f"PDF extraction failed: {exc}")
        raise


def extract_text_from_txt(source: Union[str, Path]) -> str:
    """Fallback: read plain-text knowledge-base files (.txt)."""
    path = Path(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    logger.info(f"Read text file: {path.name}  ({len(text):,} characters)")
    return text
