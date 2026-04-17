"""Tests for embedding pipeline and ChromaDB store."""


import pytest

from src.retrieval.embeddings import embed_and_store
from src.retrieval.store import (
    add_documents,
    get_client,
    get_or_create_collection,
    query_collection,
)


@pytest.fixture()
def chroma_collection(tmp_path):
    """Create a temporary ChromaDB collection for testing."""
    client = get_client(persist_dir=str(tmp_path / "chroma_test"))
    collection = get_or_create_collection(client, name="test_collection")
    return collection


class TestChromaStore:
    """Tests for ChromaDB store operations."""

    def test_add_and_query_documents(self, chroma_collection):
        """Documents can be added and queried by similarity."""
        chunks = [
            {"text": "Python is a programming language.", "metadata": {"source": "doc1"}},
            {"text": "ChromaDB is a vector database.", "metadata": {"source": "doc2"}},
            {"text": "RAG combines retrieval with generation.", "metadata": {"source": "doc3"}},
        ]

        add_documents(chroma_collection, chunks)
        assert chroma_collection.count() == 3

        results = query_collection(chroma_collection, "What is a vector database?", top_k=2)
        assert len(results) == 2
        assert all("text" in r for r in results)

    def test_add_empty_chunks_is_noop(self, chroma_collection):
        """Adding an empty list does nothing."""
        add_documents(chroma_collection, [])
        assert chroma_collection.count() == 0

    def test_query_returns_metadata(self, chroma_collection):
        """Query results include metadata."""
        chunks = [
            {"text": "Test document.", "metadata": {"filename": "test.md"}},
        ]
        add_documents(chroma_collection, chunks)

        results = query_collection(chroma_collection, "test", top_k=1)
        assert results[0]["metadata"]["filename"] == "test.md"


class TestEmbedAndStore:
    """Tests for embed_and_store()."""

    def test_embed_and_store_returns_count(self, chroma_collection):
        """embed_and_store returns the number of chunks stored."""
        chunks = [
            {"text": "Chunk one.", "metadata": {}},
            {"text": "Chunk two.", "metadata": {}},
        ]

        count = embed_and_store(chunks, chroma_collection)
        assert count == 2
        assert chroma_collection.count() == 2

    def test_embed_empty_list_returns_zero(self, chroma_collection):
        """Embedding an empty list returns 0."""
        count = embed_and_store([], chroma_collection)
        assert count == 0
