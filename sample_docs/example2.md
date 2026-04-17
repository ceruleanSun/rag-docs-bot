# Vector Databases

A vector database is a specialized database designed to store, index, and query
high-dimensional vector embeddings efficiently.

## What Are Vector Embeddings?

Vector embeddings are numerical representations of data (text, images, audio)
in a high-dimensional space. Similar items end up close together in this space,
enabling semantic search rather than keyword matching.

For example, the sentences "The cat sat on the mat" and "A kitten rested on
the rug" would have similar vector embeddings despite sharing few exact words.

## Key Operations

- **Insert**: Store a vector along with its metadata and original content.
- **Search**: Given a query vector, find the K nearest neighbors (top-K
  similar vectors) using distance metrics like cosine similarity or
  Euclidean distance.
- **Filter**: Combine vector similarity search with metadata filtering.
- **Delete/Update**: Remove or modify stored vectors.

## Popular Vector Databases

- **ChromaDB**: Open-source, lightweight, runs locally. Great for
  prototyping and small-to-medium workloads.
- **Pinecone**: Managed cloud service with high scalability.
- **Weaviate**: Open-source with hybrid search (vector + keyword).
- **FAISS**: Facebook's library for efficient similarity search (not a
  full database, but widely used).

## Why Use a Vector Database for RAG?

Vector databases enable fast semantic retrieval of relevant document chunks.
When a user asks a question, the query is embedded and compared against all
stored document chunks. The most similar chunks are returned as context for
the LLM, enabling accurate and grounded answers.
