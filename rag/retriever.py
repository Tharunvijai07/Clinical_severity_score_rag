"""
rag/retriever.py
─────────────────
High-level retriever that converts a natural-language query into
a ranked list of relevant medical guideline chunks.

Deduplicates results by content hash to avoid repeating the same passage
when multiple top-K queries hit the same chunk.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from rag.vector_store import VectorStore
from utils.config import TOP_K_RESULTS
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved knowledge-base passage."""

    text: str
    source: str
    distance: float  # cosine distance (lower = more similar)

    @property
    def similarity(self) -> float:
        """Convert cosine distance → similarity score in [0, 1]."""
        return max(0.0, 1.0 - self.distance)

    def short_repr(self) -> str:
        """One-line summary for logging / display."""
        return f"[{self.source}] ({self.similarity:.2%}) {self.text[:80]}…"


class Retriever:
    """
    Orchestrates VectorStore queries and post-processes results.

    Parameters
    ----------
    vector_store : VectorStore, optional
        Pre-initialised store; a default one is created if not provided.
    top_k : int, optional
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int = TOP_K_RESULTS,
    ) -> None:
        self._store = vector_store or VectorStore()
        self._store.initialize()
        self._top_k = top_k

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant guideline chunks for *query*.

        Parameters
        ----------
        query : str
            Clinical findings text (or any natural-language query).
        top_k : int, optional
            Override the default top-K value.

        Returns
        -------
        list[RetrievedChunk]
            Deduplicated chunks ranked by similarity (best first).
        """
        k = top_k or self._top_k
        logger.info(f"Retrieving top-{k} chunks for query ({len(query)} chars).")

        if self._store.count() == 0:
            logger.warning("Vector store is empty — no results returned.")
            return []

        raw = self._store.query(query_texts=[query], n_results=k)

        chunks = self._parse_results(raw)
        chunks = self._deduplicate(chunks)
        chunks.sort(key=lambda c: c.distance)

        logger.info(
            f"Retrieved {len(chunks)} unique chunks. "
            f"Best match: {chunks[0].short_repr() if chunks else 'N/A'}"
        )
        return chunks

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(raw: dict) -> list[RetrievedChunk]:
        """Convert raw ChromaDB result dict → list of RetrievedChunk."""
        chunks: list[RetrievedChunk] = []
        docs      = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            if not doc or not doc.strip():
                continue
            chunks.append(
                RetrievedChunk(
                    text=doc.strip(),
                    source=meta.get("source", "unknown") if meta else "unknown",
                    distance=float(dist),
                )
            )
        return chunks

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove chunks whose text content is identical (by hash)."""
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for chunk in chunks:
            h = hashlib.sha256(chunk.text.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        return unique
