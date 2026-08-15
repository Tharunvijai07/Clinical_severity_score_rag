"""
rag/text_chunker.py
────────────────────
Splits extracted text into overlapping chunks suitable for embedding.
Uses LangChain's RecursiveCharacterTextSplitter to respect natural
paragraph and sentence boundaries.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

logger = get_logger(__name__)

# Separators tried in order; falls back to character split if none found
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping chunks.

    Parameters
    ----------
    text : str
        Full document text to split.
    chunk_size : int
        Target character length per chunk.
    chunk_overlap : int
        Number of characters shared between consecutive chunks.

    Returns
    -------
    list[str]
        Non-empty text chunks ready for embedding.
    """
    if not text or not text.strip():
        logger.warning("chunk_text received empty text — returning empty list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    raw_chunks = splitter.split_text(text)
    # Filter out chunks that are just whitespace
    chunks = [c.strip() for c in raw_chunks if c.strip()]

    logger.info(
        f"Chunked text into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})."
    )
    return chunks
