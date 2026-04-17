"""Tests for text chunking."""

from src.ingestion.chunker import chunk_text


class TestChunkText:
    """Tests for chunk_text()."""

    def test_basic_chunking(self):
        """Text is split into chunks of the correct size."""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=0)

        assert len(chunks) == 2
        assert len(chunks[0]["text"]) == 500
        assert len(chunks[1]["text"]) == 500

    def test_chunking_with_overlap(self):
        """Overlapping chunks share characters at boundaries."""
        text = "abcdefghij"  # 10 chars
        chunks = chunk_text(text, chunk_size=6, overlap=2)

        # First chunk: chars 0-5 = "abcdef"
        # Second chunk: starts at 4, chars 4-9 = "efghij"
        assert chunks[0]["text"] == "abcdef"
        assert chunks[1]["text"] == "efghij"
        assert len(chunks) >= 2

    def test_empty_text_returns_empty_list(self):
        """Empty text returns no chunks."""
        assert chunk_text("") == []

    def test_text_shorter_than_chunk_size(self):
        """Text shorter than chunk_size produces a single chunk."""
        chunks = chunk_text("hello", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello"

    def test_metadata_is_propagated(self):
        """Metadata from input is included in each chunk."""
        meta = {"filename": "test.md", "path": "/tmp/test.md"}
        chunks = chunk_text("hello world", chunk_size=500, metadata=meta)

        assert chunks[0]["metadata"]["filename"] == "test.md"
        assert chunks[0]["metadata"]["path"] == "/tmp/test.md"
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_chunk_indices_are_sequential(self):
        """Each chunk has a sequential chunk_index."""
        text = "a" * 200
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
