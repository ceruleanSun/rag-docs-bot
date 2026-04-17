"""ChromaDB vector store for the RAG pipeline."""

import chromadb


def get_client(persist_dir: str = "./chroma_db") -> chromadb.ClientAPI:
    """Initialize a persistent ChromaDB client.

    Args:
        persist_dir: Directory for ChromaDB persistent storage.

    Returns:
        A ChromaDB PersistentClient instance.
    """
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(
    client: chromadb.ClientAPI, name: str = "documents"
) -> chromadb.Collection:
    """Get an existing collection or create a new one.

    Args:
        client: ChromaDB client instance.
        name: Name of the collection.

    Returns:
        A ChromaDB Collection.
    """
    return client.get_or_create_collection(name=name)


def add_documents(
    collection: chromadb.Collection,
    chunks: list[dict],
) -> None:
    """Add document chunks to a ChromaDB collection.

    Each chunk dict should have 'text' and 'metadata' keys.

    Args:
        collection: The ChromaDB collection to add to.
        chunks: List of chunk dicts with 'text' and 'metadata'.
    """
    if not chunks:
        return

    ids = [f"chunk_{i}" for i in range(collection.count(), collection.count() + len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {k: str(v) for k, v in c.get("metadata", {}).items()}
        if not meta:
            meta = {"source": "unknown"}
        metadatas.append(meta)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """Query the collection for the most similar chunks.

    Args:
        collection: The ChromaDB collection to search.
        query_text: The query string to search for.
        top_k: Number of top results to return.

    Returns:
        List of result dicts with 'text', 'metadata', and 'distance'.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, collection.count()),
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            }
        )

    return output
