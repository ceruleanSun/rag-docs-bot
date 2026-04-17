"""Integration tests for the full RAG pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from src.generation.llm import generate_answer
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_directory
from src.retrieval.embeddings import embed_and_store
from src.retrieval.store import (
    get_client,
    get_or_create_collection,
    query_collection,
)


@pytest.fixture()
def sample_docs_dir(tmp_path):
    """Create a temporary directory with sample documents."""
    (tmp_path / "rag.md").write_text(
        "# RAG\nRAG stands for Retrieval Augmented Generation. "
        "It combines search with LLM generation.",
        encoding="utf-8",
    )
    (tmp_path / "vectors.md").write_text(
        "# Vectors\nVector embeddings represent text as numbers. "
        "Similar texts have similar vectors.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def chroma_collection(tmp_path):
    """Create a temporary ChromaDB collection."""
    client = get_client(persist_dir=str(tmp_path / "chroma_pipeline"))
    return get_or_create_collection(client, name="pipeline_test")


class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_ingest_and_retrieve(self, sample_docs_dir, chroma_collection):
        """Documents can be ingested and retrieved by query."""
        # Load
        documents = load_directory(str(sample_docs_dir))
        assert len(documents) == 2

        # Chunk
        all_chunks = []
        for doc in documents:
            chunks = chunk_text(doc["text"], chunk_size=200, overlap=20, metadata=doc["metadata"])
            all_chunks.extend(chunks)
        assert len(all_chunks) > 0

        # Embed and store
        embed_and_store(all_chunks, chroma_collection)
        assert chroma_collection.count() == len(all_chunks)

        # Query
        results = query_collection(chroma_collection, "What is RAG?", top_k=3)
        assert len(results) > 0
        # At least one result should mention RAG
        texts = " ".join(r["text"] for r in results)
        assert "RAG" in texts

    @patch("src.generation.llm.anthropic")
    def test_generate_answer_with_mock(self, mock_anthropic):
        """generate_answer calls Claude and returns text (mocked)."""
        # Set up mock
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="RAG combines retrieval with generation.")
        ]
        mock_client.messages.create.return_value = mock_response

        # Call
        context = [{"text": "RAG stands for Retrieval Augmented Generation."}]
        answer = generate_answer("What is RAG?", context, api_key="fake-key")

        assert "RAG" in answer
        mock_client.messages.create.assert_called_once()

    @patch("src.generation.llm.anthropic")
    def test_full_pipeline_with_mock_llm(
        self, mock_anthropic, sample_docs_dir, chroma_collection
    ):
        """Full pipeline: ingest, query, generate (with mocked LLM)."""
        # Set up mock
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Vector embeddings represent text as numbers.")]
        mock_client.messages.create.return_value = mock_response

        # Ingest
        documents = load_directory(str(sample_docs_dir))
        all_chunks = []
        for doc in documents:
            chunks = chunk_text(doc["text"], chunk_size=200, overlap=20, metadata=doc["metadata"])
            all_chunks.extend(chunks)
        embed_and_store(all_chunks, chroma_collection)

        # Query + Generate
        results = query_collection(chroma_collection, "What are vector embeddings?", top_k=3)
        answer = generate_answer("What are vector embeddings?", results, api_key="fake-key")

        assert isinstance(answer, str)
        assert len(answer) > 0
