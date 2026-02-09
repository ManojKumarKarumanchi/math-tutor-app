"""
Index the math KB (rag/kb/*.txt) into LanceDB: parse → semantic chunk → embed → index.
Run from project root: py rag/index_kb.py [--reindex] [--chunk-size 250]
Uses SemanticChunking + SentenceTransformerEmbedder (all-MiniLM-L6-v2).
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag.vectorstore import get_knowledge, index_kb, get_kb_docs_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index rag/kb docs into LanceDB (SemanticChunking → embed → index) with metadata for filtering."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-index all docs (do not skip_if_exists). Clear collection first if you want a clean rebuild.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        metavar="N",
        help="SemanticChunking chunk size (default: 250).",
    )
    args = parser.parse_args()

    kb_dir = get_kb_docs_dir()
    print(f"KB directory: {kb_dir}")
    print(f"Chunking: SemanticChunking, size={args.chunk_size}")
    knowledge = get_knowledge()
    inserted = index_kb(
        knowledge=knowledge,
        kb_dir=kb_dir,
        chunk_size=args.chunk_size,
        skip_if_exists=not args.reindex,
    )
    print(f"Indexed {len(inserted)} documents: {', '.join(inserted)}")
    print(
        "Vector store ready. Query with Agent(knowledge=..., search_knowledge=True). Metadata: doc_type, source, category."
    )


if __name__ == "__main__":
    main()
