"""
rag/text_chunker.py
────────────────────
Splits extracted text into overlapping chunks suitable for embedding.

Two-pass strategy:
  1. Section-aware pre-split on this project's guideline format
     ("═══ / SECTION N: TITLE / ═══" dividers + ALL-CAPS subheadings),
     so each piece stays topic-pure and keeps a "[SECTION — subheading]"
     label prefix (prevents cross-framework threshold collisions, e.g.
     MAP <65 general shock vs MAP <70 SOFA cardiovascular score).
  2. RecursiveCharacterTextSplitter as a fallback ONLY for any section
     that's still longer than chunk_size, to preserve the existing
     overlap behavior for oversized blocks.

Public signature is unchanged — chunk_text(text, chunk_size, chunk_overlap)
-> list[str] — so nothing else in the app needs to change.
"""
from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

logger = get_logger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]

_SECTION_DIVIDER = re.compile(
    r'═{10,}\s*\nSECTION\s+\d+:\s*(.+?)\s*\n═{10,}\s*\n', re.MULTILINE
)
_SUBHEADING = re.compile(
    r'^[ \t]{0,6}([A-Z][A-Za-z0-9 /×%\-\(\)\+²₂₃\',\.]{1,80}):[ \t]*$', re.MULTILINE
)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split on SECTION divider blocks. Falls back to one section if none found."""
    matches = list(_SECTION_DIVIDER.finditer(text))
    if not matches:
        return [("", text)]
    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, text[start:end].strip()))
    return sections


def _split_section_by_subheading(title: str, body: str, min_chars: int = 150) -> list[str]:
    """Split a section body at ALL-CAPS subheadings, prefixing each piece
    with '[SECTION — subheading]' so the label survives into the chunk."""
    subheads = list(_SUBHEADING.finditer(body))
    if not subheads:
        label = f"[{title}]\n" if title else ""
        return [f"{label}{body.strip()}"] if body.strip() else []

    pieces = []
    if subheads[0].start() > 0:
        preamble = body[: subheads[0].start()].strip()
        if preamble:
            label = f"[{title}]\n" if title else ""
            pieces.append(f"{label}{preamble}")

    for i, m in enumerate(subheads):
        subhead = m.group(1).strip()
        start = m.start()
        end = subheads[i + 1].start() if i + 1 < len(subheads) else len(body)
        content = body[start:end].strip()
        label = f"[{title} — {subhead}]" if title else f"[{subhead}]"
        pieces.append(f"{label}\n{content}")

    merged = []
    for p in pieces:
        if merged and len(p) < min_chars:
            merged[-1] += "\n\n" + p
        else:
            merged.append(p)
    return merged


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping chunks.

    Section/subheading-aware first pass keeps each guideline rule with its
    framework label intact; RecursiveCharacterTextSplitter only runs as a
    fallback on pieces that are still oversized after that pass.

    Parameters
    ----------
    text : str
        Full document text to split.
    chunk_size : int
        Target character length per chunk (also the oversize threshold
        for the section-aware pass).
    chunk_overlap : int
        Number of characters shared between consecutive chunks, used only
        for the fallback splitter on oversized sections.

    Returns
    -------
    list[str]
        Non-empty text chunks ready for embedding.
    """
    if not text or not text.strip():
        logger.warning("chunk_text received empty text — returning empty list.")
        return []

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[str] = []
    sections = _split_into_sections(text)

    for title, body in sections:
        for piece in _split_section_by_subheading(title, body):
            if len(piece) <= chunk_size:
                chunks.append(piece)
            else:
                # Oversized section/subheading — fall back to recursive
                # character splitting, but keep the label on each sub-piece.
                label_match = re.match(r'^(\[[^\]]+\]\n)', piece)
                label = label_match.group(1) if label_match else ""
                body_only = piece[len(label):] if label else piece
                for sub in fallback_splitter.split_text(body_only):
                    if sub.strip():
                        chunks.append(f"{label}{sub.strip()}" if label else sub.strip())

    chunks = [c.strip() for c in chunks if c.strip()]

    logger.info(
        f"Chunked text into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap}, section-aware pre-split)."
    )
    return chunks