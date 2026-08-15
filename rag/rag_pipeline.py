"""
rag/rag_pipeline.py
────────────────────
Top-level orchestrator for the RAG workflow.

Two public responsibilities:
  1. build_knowledge_base() — one-time ingestion of medical guidelines.
  2. run_query()            — per-request retrieval for patient reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from tqdm import tqdm

from rag.pdf_parser import extract_text_from_pdf, extract_text_from_txt
from rag.retriever import RetrievedChunk, Retriever
from rag.text_chunker import chunk_text
from rag.vector_store import VectorStore
from utils.config import KNOWLEDGE_BASE_PATH, TOP_K_RESULTS
from utils.logger import get_logger

logger = get_logger(__name__)

# Supported document extensions
_SUPPORTED_EXTS = {".pdf", ".txt"}


class RAGPipeline:
    """
    Orchestrates the full RAG lifecycle.

    Parameters
    ----------
    knowledge_base_path : Path, optional
        Directory containing medical guideline documents.
    top_k : int, optional
        Default number of chunks to retrieve per query.
    """

    def __init__(
        self,
        knowledge_base_path: Path | None = None,
        top_k: int = TOP_K_RESULTS,
    ) -> None:
        self._kb_path = knowledge_base_path or KNOWLEDGE_BASE_PATH
        self._top_k = top_k
        self._store = VectorStore()
        self._retriever = Retriever(vector_store=self._store, top_k=top_k)

    # ── Knowledge Base ─────────────────────────────────────────────────────────

    def is_knowledge_base_ready(self) -> bool:
        """Return True if the vector store contains at least one document."""
        try:
            return self._store.count() > 0
        except Exception:
            return False

    def get_document_count(self) -> int:
        """Total chunks stored in ChromaDB."""
        try:
            return self._store.count()
        except Exception:
            return 0

    def build_knowledge_base(
        self,
        progress_callback: Callable[[str], None] | None = None,
        force_rebuild: bool = False,
    ) -> int:
        """
        Scan the knowledge-base directory, parse documents, chunk, and
        upsert embeddings into ChromaDB.

        Parameters
        ----------
        progress_callback : callable, optional
            Called with a status string at each step (useful for Streamlit).
        force_rebuild : bool
            If True, reset the collection before re-ingesting.

        Returns
        -------
        int
            Total chunks stored after ingestion.
        """
        def _cb(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        if force_rebuild:
            _cb("🔄 Resetting vector store…")
            self._store.reset()

        _cb(f"📂 Scanning knowledge base: {self._kb_path}")
        files = [
            f for f in self._kb_path.iterdir()
            if f.suffix.lower() in _SUPPORTED_EXTS and f.is_file()
        ]

        if not files:
            _cb("⚠️  No documents found in knowledge_base/. Add .pdf or .txt files.")
            return 0

        _cb(f"📄 Found {len(files)} documents: {[f.name for f in files]}")

        total_chunks = 0
        for file in tqdm(files, desc="Ingesting guidelines"):
            try:
                _cb(f"  → Parsing: {file.name}")
                if file.suffix.lower() == ".pdf":
                    text = extract_text_from_pdf(file)
                else:
                    text = extract_text_from_txt(file)

                if not text.strip():
                    _cb(f"  ⚠️  Empty content in {file.name}, skipping.")
                    continue

                _cb(f"  → Chunking: {file.name}")
                chunks = chunk_text(text)

                source_name = file.stem  # e.g. "sepsis_guidelines"
                metadatas = [{"source": source_name, "file": file.name}
                             for _ in chunks]

                _cb(f"  → Embedding & storing {len(chunks)} chunks from {file.name}…")
                self._store.add_documents(chunks, metadatas)
                total_chunks += len(chunks)
                _cb(f"  ✅ {file.name}: {len(chunks)} chunks ingested.")

            except Exception as exc:
                _cb(f"  ❌ Error processing {file.name}: {exc}")
                logger.exception(exc)

        _cb(f"🎉 Knowledge base ready — {total_chunks} total chunks across {len(files)} documents.")
        return total_chunks

    # ── Inference ──────────────────────────────────────────────────────────────

    def run_query(
        self,
        patient_text: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant guideline chunks for a patient report.

        Parameters
        ----------
        patient_text : str
            Extracted text from the patient's lab report.
        top_k : int, optional
            Override the default top-K.

        Returns
        -------
        list[RetrievedChunk]
            Ranked, deduplicated guideline passages.
        """
        k = top_k or self._top_k
        logger.info(f"Running RAG query (top_k={k})…")
        return self._retriever.retrieve(patient_text, top_k=k)
