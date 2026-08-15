"""
rag/vector_store.py
────────────────────
ChromaDB wrapper that handles collection lifecycle, document ingestion,
and similarity queries.

Key design points:
  • Uses a persistent client so the index survives process restarts.
  • Accepts batch upserts with per-chunk source metadata.
  • Returns a clean result dict from queries.
"""
from __future__ import annotations

import hashlib
from typing import Any

import chromadb
from chromadb.config import Settings

from rag.embedder import MedicalEmbeddingFunction
from utils.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
from utils.logger import get_logger

logger = get_logger(__name__)


def _make_id(text: str) -> str:
    """Deterministic SHA-256 ID so re-ingesting the same chunk is idempotent."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class VectorStore:
    """Thin wrapper around a single ChromaDB collection."""

    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._db_path = db_path or str(CHROMA_DB_PATH)
        self._collection_name = collection_name or CHROMA_COLLECTION_NAME
        self._client: chromadb.PersistentClient | None = None
        self._collection = None
        self._embed_fn = MedicalEmbeddingFunction()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create (or load) the persistent ChromaDB client and collection."""
        logger.info(f"Initialising ChromaDB at: {self._db_path}")
        self._client = chromadb.PersistentClient(
            path=self._db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info(
            f"Collection '{self._collection_name}' ready — {count} documents stored."
        )

    def _ensure_ready(self) -> None:
        if self._collection is None:
            self.initialize()

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        chunks: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Upsert chunks into the collection.

        Parameters
        ----------
        chunks : list[str]
            Text chunks to store.
        metadatas : list[dict], optional
            Parallel list of metadata dicts (e.g. {"source": "sepsis_guidelines"}).

        Returns
        -------
        int
            Number of chunks successfully upserted.
        """
        self._ensure_ready()
        if not chunks:
            return 0

        ids = [_make_id(c) for c in chunks]
        metas = metadatas or [{}] * len(chunks)

        # Upsert in batches of 100 to stay within ChromaDB limits
        batch_size = 100
        upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch_ids   = ids[i : i + batch_size]
            batch_docs  = chunks[i : i + batch_size]
            batch_metas = metas[i : i + batch_size]
            self._collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            upserted += len(batch_ids)

        logger.info(f"Upserted {upserted} chunks into '{self._collection_name}'.")
        return upserted

    # ── Read ───────────────────────────────────────────────────────────────────

    def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """
        Semantic search.

        Returns the raw ChromaDB result dict with keys:
            ids, documents, metadatas, distances
        """
        self._ensure_ready()
        n = min(n_results, self.count())
        if n == 0:
            logger.warning("Collection is empty — cannot query.")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = self._collection.query(
            query_texts=query_texts,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def count(self) -> int:
        """Return the number of stored documents."""
        self._ensure_ready()
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (use with caution)."""
        self._ensure_ready()
        logger.warning(f"Resetting collection '{self._collection_name}'.")
        self._client.delete_collection(self._collection_name)
        self._collection = None
        self.initialize()
