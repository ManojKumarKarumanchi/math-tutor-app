"""
RAG Retriever: Retrieves relevant context from LanceDB.
Returns chunk texts and citation metadata for UI.
"""

from rag.vectorstore import get_indexed_knowledge


def get_retriever():
    """Return the LanceDB knowledge base."""
    return get_indexed_knowledge()


def retrieve_context(query: str, topic: str = None, top_k: int = 3) -> tuple:
    """Retrieve relevant chunks from LanceDB. Returns (chunk_strings, citation_infos)."""
    try:
        kb = get_retriever()
        search_kwargs = {"max_results": top_k}
        results = kb.search(query, **search_kwargs)

        chunks = []
        citation_infos = []
        for doc in results or []:
            content = getattr(doc, "content", None) or ""
            meta = getattr(doc, "meta_data", None) or {}
            name = getattr(doc, "name", None)
            chunks.append(content)
            citation_infos.append(
                {
                    "content": content,
                    "source": meta.get("source") or name,
                    "name": name,
                    "category": meta.get("category"),
                }
            )
        return chunks, citation_infos

    except Exception as e:
        print(f"Retrieval error: {e}")
        return [], []


def check_context_sufficiency(chunks: list, min_chunks: int = 2) -> bool:
    """Return True if number of chunks >= min_chunks."""
    return len(chunks) >= min_chunks
