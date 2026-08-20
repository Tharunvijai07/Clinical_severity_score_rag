"""
utils/config.py
───────────────
Centralised configuration loader.
All modules import from here so .env changes propagate automatically.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the project root (two levels up from this file: utils/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=_ENV_FILE, override=False)


def _require(key: str) -> str:
    """Raise a clear error when a required env-var is missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Please check your .env file at {_ENV_FILE}."
        )
    return value


def _path(key: str, default: str) -> Path:
    raw = os.getenv(key, default)
    p = Path(raw)
    # If relative, anchor to project root
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── API Keys ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = _require("OPENROUTER_API_KEY")

# ── Model ─────────────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ── Paths ─────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: Path        = _path("CHROMA_DB_PATH",        "./vectordb")
KNOWLEDGE_BASE_PATH: Path   = _path("KNOWLEDGE_BASE_PATH",   "./knowledge_base")
UPLOADS_PATH: Path          = _path("UPLOADS_PATH",          "./uploads")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int     = int(os.getenv("CHUNK_SIZE",    "600"))
CHUNK_OVERLAP: int  = int(os.getenv("CHUNK_OVERLAP", "80"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int  = int(os.getenv("TOP_K_RESULTS", "5"))

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = os.getenv(
    "CHROMA_COLLECTION_NAME", "medical_guidelines"
)
