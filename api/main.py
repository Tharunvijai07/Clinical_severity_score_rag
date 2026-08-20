from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm.severity_analyzer import SeverityAnalyzer
from rag.pdf_parser import extract_text_from_pdf
from rag.rag_pipeline import RAGPipeline
from rag.retriever import RetrievedChunk
from utils.config import CHROMA_COLLECTION_NAME
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="MedSeverity AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = RAGPipeline()
_analyzer = SeverityAnalyzer()


class RetrievedChunkResponse(BaseModel):
    text: str
    source: str
    distance: float
    similarity: float


class SeverityResultResponse(BaseModel):
    severity_score: float = Field(ge=0, le=10)
    severity_level: Literal["Low", "Moderate", "High", "Critical"]
    confidence: float
    key_findings: list[str]
    evidence: list[str]
    summary: str


class AnalyzeResponse(BaseModel):
    extracted_text: str
    retrieved_chunks: list[RetrievedChunkResponse]
    severity_result: SeverityResultResponse
    raw_model_response: str


class KnowledgeBaseStatusResponse(BaseModel):
    ready: bool
    chunk_count: int
    collection_name: str


class BuildStatusResponse(BaseModel):
    running: bool
    ready: bool
    chunk_count: int
    collection_name: str
    progress: int
    message: str
    error: str | None = None


@dataclass
class BuildState:
    running: bool = False
    progress: int = 0
    message: str = "Idle"
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    messages: list[str] = field(default_factory=list)


_build_state = BuildState()
_build_lock = threading.Lock()


def _kb_status() -> KnowledgeBaseStatusResponse:
    chunk_count = _pipeline.get_document_count()
    return KnowledgeBaseStatusResponse(
        ready=chunk_count > 0,
        chunk_count=chunk_count,
        collection_name=CHROMA_COLLECTION_NAME,
    )


def _chunk_to_response(chunk: RetrievedChunk) -> RetrievedChunkResponse:
    return RetrievedChunkResponse(
        text=chunk.text,
        source=chunk.source,
        distance=chunk.distance,
        similarity=chunk.similarity,
    )


def _run_build(force: bool) -> None:
    def update(message: str) -> None:
        with _build_lock:
            _build_state.message = message
            _build_state.messages.append(message)
            _build_state.progress = min(95, _build_state.progress + 8)

    try:
        with _build_lock:
            _build_state.running = True
            _build_state.progress = 5
            _build_state.message = "Starting knowledge base build"
            _build_state.error = None
            _build_state.started_at = time.time()
            _build_state.completed_at = None
            _build_state.messages = [_build_state.message]

        count = _pipeline.build_knowledge_base(
            progress_callback=update,
            force_rebuild=force,
        )

        with _build_lock:
            _build_state.running = False
            _build_state.progress = 100
            _build_state.message = f"Knowledge base ready with {count} chunks"
            _build_state.completed_at = time.time()
            _build_state.messages.append(_build_state.message)
    except Exception as exc:
        logger.exception(exc)
        with _build_lock:
            _build_state.running = False
            _build_state.error = str(exc)
            _build_state.message = "Knowledge base build failed"
            _build_state.completed_at = time.time()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/knowledge-base/status", response_model=KnowledgeBaseStatusResponse)
def knowledge_base_status() -> KnowledgeBaseStatusResponse:
    return _kb_status()


@app.get("/api/knowledge-base/build/status", response_model=BuildStatusResponse)
def knowledge_base_build_status() -> BuildStatusResponse:
    status = _kb_status()
    with _build_lock:
        return BuildStatusResponse(
            running=_build_state.running,
            ready=status.ready,
            chunk_count=status.chunk_count,
            collection_name=status.collection_name,
            progress=_build_state.progress,
            message=_build_state.message,
            error=_build_state.error,
        )


@app.post("/api/knowledge-base/build", response_model=BuildStatusResponse)
def build_knowledge_base(
    force: bool = Query(False, description="Reset and rebuild the vector store first."),
) -> BuildStatusResponse:
    with _build_lock:
        if _build_state.running:
            raise HTTPException(
                status_code=409,
                detail="Knowledge base build is already running.",
            )

    thread = threading.Thread(target=_run_build, args=(force,), daemon=True)
    thread.start()
    return knowledge_base_build_status()


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF report.")

    status = _kb_status()
    if not status.ready:
        raise HTTPException(
            status_code=409,
            detail="Knowledge base is not ready. Build it before analyzing a report.",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

    try:
        extracted_text = extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        logger.exception(exc)
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the PDF. Check that it is a readable lab report PDF.",
        ) from exc

    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No clinical text was found in the PDF after extraction.",
        )

    try:
        retrieved_chunks = _pipeline.run_query(extracted_text)
        severity_result = _analyzer.analyze(extracted_text, retrieved_chunks)
    except RuntimeError as exc:
        logger.exception(exc)
        raise HTTPException(
            status_code=502,
            detail="The model request failed. Check your OpenRouter API key, model, and account credits.",
        ) from exc
    except Exception as exc:
        logger.exception(exc)
        raise HTTPException(
            status_code=500,
            detail="Analysis failed while retrieving guidelines or generating the assessment.",
        ) from exc

    return AnalyzeResponse(
        extracted_text=extracted_text,
        retrieved_chunks=[_chunk_to_response(chunk) for chunk in retrieved_chunks],
        severity_result=SeverityResultResponse(**severity_result.to_dict()),
        raw_model_response=severity_result.raw_response,
    )
