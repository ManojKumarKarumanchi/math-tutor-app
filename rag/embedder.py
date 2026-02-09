"""
Embedding interface for Memory and RAG.
Uses config.models.get_embedder() so the model loads once and is reused.
"""

from config.models import get_embedder


def embed(texts: list | str) -> list:
    """Generate embeddings. texts: string or list of strings. Returns list of vectors."""
    if isinstance(texts, str):
        texts = [texts]
    emb = get_embedder()
    result = emb.get_embedding(texts)
    if not result:
        return []
    if isinstance(result[0], (int, float)):
        return [result]
    return result
