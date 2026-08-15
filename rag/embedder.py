"""
rag/embedder.py
───────────────
Wraps sentence-transformers to generate dense vector embeddings.

Design decisions:
  • Singleton pattern — model loaded once per process.
  • Exposes a ChromaDB-compatible EmbeddingFunction subclass so ChromaDB
    can call the same model transparently.
  • Falls back gracefully if the model download fails (raises clear error).
"""
from __future__ import annotations

from typing import List

import chromadb.utils.embedding_functions as ef
from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Singleton embedding engine backed by a sentence-transformers model."""

    _instance: "Embedder | None" = None

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def _init_model(self) -> None:
        if self._initialised:
            return
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} …")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._initialised = True
        logger.info("Embedding model ready.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings.

        Returns
        -------
        list[list[float]]
            One vector per input text.
        """
        self._init_model()
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> List[float]:
        """Convenience wrapper for a single string."""
        return self.embed([text])[0]


class MedicalEmbeddingFunction(ef.EmbeddingFunction):
    """
    ChromaDB-compatible wrapper so the same local model is used for both
    ingestion and query without any API keys.
    """

    def __init__(self) -> None:
        self._embedder = Embedder()

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self._embedder.embed(input)
