"""
Knowledge base loader: Indexes KB documents into LanceDB.
Uses config.models.get_embedder() for chunking and vector DB.
"""

from rag.vectorstore import index_kb


def load_kb():
    """Load and index the knowledge base. Returns True on success."""
    try:
        indexed = index_kb(skip_if_exists=True)
        return True
    except Exception as e:
        print(f"Error loading KB: {e}")
        return False
