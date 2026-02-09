"""
RAG vector store and Knowledge base for the math KB.
Uses Agno Knowledge + LanceDB with Semantic Chunking and SentenceTransformers.
Index sources: rag/kb/*.txt
Flow: parse (reader) → semantic chunk → embed → index into LanceDB.
Embedder: single instance from config.models.get_embedder() reused for LanceDb and SemanticChunking (model loads once).
Refs: https://docs.agno.com/reference/knowledge/embedder/sentence-transformer
      https://docs.agno.com/knowledge/concepts/chunking/semantic-chunking
"""

import asyncio
from pathlib import Path
from typing import Any, Optional

from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.base import Reader
from agno.knowledge.reader.text_reader import TextReader
from agno.vectordb.lancedb import LanceDb

from config.models import get_embedder

# Default paths (relative to this file)
RAG_DIR = Path(__file__).resolve().parent
KB_DIR = RAG_DIR / "kb"
VECTORSTORE_DIR = RAG_DIR / "tmp" / "lancedb"
TABLE_NAME = "math_kb"

# SemanticChunking defaults
DEFAULT_CHUNK_SIZE = 250
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_SIMILARITY_WINDOW = 3
DEFAULT_MIN_SENTENCES_PER_CHUNK = 1


def get_vectorstore_path() -> Path:
    """Return path for LanceDB persistence."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return VECTORSTORE_DIR


def get_knowledge(
    table_name: str = TABLE_NAME,
    path: str | Path | None = None,
    *,
    embedder: Optional[Any] = None,
) -> Knowledge:
    """
    Create and return an Agno Knowledge instance backed by LanceDB.
    Uses single embedder from config.models.get_embedder() (same instance as chunking).
    """
    base_path = path or get_vectorstore_path()
    effective_embedder = embedder if embedder is not None else get_embedder()
    return Knowledge(
        name="Math KB",
        description="JEE-style math knowledge: algebra, calculus, probability, linear algebra, formulas, templates, pitfalls.",
        vector_db=LanceDb(
            table_name=table_name,
            uri=str(base_path),
            embedder=effective_embedder,
        ),
    )


def get_kb_docs_dir() -> Path:
    """Return the directory containing KB .txt documents."""
    if not KB_DIR.is_dir():
        raise FileNotFoundError(f"KB directory not found: {KB_DIR}")
    return KB_DIR


def make_text_reader(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    embedder: Optional[Any] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    similarity_window: int = DEFAULT_SIMILARITY_WINDOW,
    min_sentences_per_chunk: int = DEFAULT_MIN_SENTENCES_PER_CHUNK,
) -> Reader:
    """
    TextReader for .txt/.md in rag/kb using SemanticChunking (same embedder as LanceDb).
    """
    effective_embedder = embedder if embedder is not None else get_embedder()
    return TextReader(
        chunking_strategy=SemanticChunking(
            embedder=effective_embedder,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            similarity_window=similarity_window,
            min_sentences_per_chunk=min_sentences_per_chunk,
        ),
    )


def _metadata_for_path(path: Path) -> dict:
    """Build metadata for filtering: doc_type, source, category."""
    name = path.stem.lower()
    if name.startswith("template_"):
        category = "template"
    elif name.startswith("pitfalls_"):
        category = "pitfalls"
    elif name.startswith("formulas_") or name in (
        "algebra",
        "calculus",
        "probability",
        "linear_algebra",
        "scope_math",
    ):
        category = "formulas"
    elif name in ("domain_constraints", "units_conventions"):
        category = "constraints"
    else:
        category = "reference"
    return {
        "source": path.name,
        "doc_type": "math_kb",
        "category": category,
    }


async def index_kb_async(
    knowledge: Knowledge | None = None,
    kb_dir: Path | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    reader: Optional[Reader] = None,
    skip_if_exists: bool = True,
    extensions: tuple[str, ...] = (".txt", ".md"),
) -> list[str]:
    """
    Index all documents in rag/kb: parse → SemanticChunking → embed → index into LanceDB.
    """
    kb_dir = kb_dir or get_kb_docs_dir()
    knowledge = knowledge or get_knowledge()
    effective_reader = reader or make_text_reader(chunk_size=chunk_size)

    inserted: list[str] = []
    for path in sorted(kb_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        # Use path= for local files so Agno reads from disk; url= triggers HTTP fetch (fails for file://).
        name = path.stem
        metadata = _metadata_for_path(path)
        kwargs: dict = {
            "name": name,
            "path": str(path.resolve()),
            "metadata": metadata,
            "reader": effective_reader,
        }
        if skip_if_exists:
            kwargs["skip_if_exists"] = True
        await knowledge.ainsert(**kwargs)
        inserted.append(name)
    return inserted


def index_kb(
    knowledge: Knowledge | None = None,
    kb_dir: Path | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    reader: Optional[Reader] = None,
    skip_if_exists: bool = True,
    extensions: tuple[str, ...] = (".txt", ".md"),
) -> list[str]:
    """Synchronous wrapper. SemanticChunking only. Metadata: doc_type, source, category."""
    return asyncio.run(
        index_kb_async(
            knowledge=knowledge,
            kb_dir=kb_dir,
            chunk_size=chunk_size,
            reader=reader,
            skip_if_exists=skip_if_exists,
            extensions=extensions,
        )
    )


def get_indexed_knowledge() -> Knowledge:
    """Return Knowledge instance (run index_kb first if you need a fresh index)."""
    return get_knowledge()
