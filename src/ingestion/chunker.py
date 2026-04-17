"""Text chunking utilities for the RAG pipeline."""


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    metadata: dict | None = None,
) -> list[dict]:
    """Split text into overlapping chunks.

    Args:
        text: The full text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
        metadata: Optional metadata dict to attach to each chunk.

    Returns:
        List of dicts, each with 'text', 'metadata', and 'chunk_index'.
    """
    if not text:
        return []

    if metadata is None:
        metadata = {}

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(
            {
                "text": chunk,
                "metadata": {**metadata, "chunk_index": chunk_index},
                "chunk_index": chunk_index,
            }
        )

        chunk_index += 1
        start += chunk_size - overlap

    return chunks
