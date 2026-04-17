"""Embedding pipeline for the RAG system.

Uses ChromaDB's built-in default embedding model (no external API needed).
"""

import chromadb

from src.retrieval.store import add_documents


def embed_and_store(
    chunks: list[dict], collection: chromadb.Collection
) -> int:
    """Embed document chunks and store them in ChromaDB.

    ChromaDB handles embedding automatically using its default model
    when documents are added without explicit embeddings.

    Args:
        chunks: List of chunk dicts with 'text' and 'metadata'.
        collection: The ChromaDB collection to store in.

    Returns:
        Number of chunks stored.
    """
    if not chunks:
        return 0

    add_documents(collection, chunks)
    return len(chunks)
