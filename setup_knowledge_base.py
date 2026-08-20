"""
setup_knowledge_base.py
────────────────────────
Standalone script to build (or rebuild) the ChromaDB knowledge base
from medical guideline documents in ./knowledge_base/.

Run this once before launching the Streamlit app:
    python setup_knowledge_base.py
    python setup_knowledge_base.py --force   # wipe and rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB medical knowledge base."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Delete existing vector store and rebuild from scratch.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MedSeverity AI - Knowledge Base Setup")
    print("=" * 60 + "\n")

    pipeline = RAGPipeline()

    if pipeline.is_knowledge_base_ready() and not args.force:
        count = pipeline.get_document_count()
        print(f"✅ Knowledge base already contains {count:,} chunks.")
        print("   Use --force to rebuild from scratch.")
        return

    def _log(msg: str) -> None:
        print(f"  {msg}")

    total = pipeline.build_knowledge_base(
        progress_callback=_log,
        force_rebuild=args.force,
    )

    if total > 0:
        print(f"\n✅ Setup complete — {total:,} chunks stored in ChromaDB.")
        print("   You can now launch the app with:\n")
        print("   streamlit run app/streamlit_app.py\n")
    else:
        print("\n⚠️  No chunks were stored.")
        print("   Ensure .txt or .pdf files exist in the knowledge_base/ directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
