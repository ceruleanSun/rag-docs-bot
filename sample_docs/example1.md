# Retrieval Augmented Generation (RAG)

RAG is a technique that enhances Large Language Models (LLMs) by providing them
with relevant external knowledge at query time, rather than relying solely on
information learned during training.

## How RAG Works

1. **Ingestion**: Documents are loaded, split into chunks, and converted into
   vector embeddings. These embeddings are stored in a vector database.

2. **Retrieval**: When a user asks a question, the question is also converted
   into an embedding. The vector database finds the most similar document
   chunks using cosine similarity or other distance metrics.

3. **Generation**: The retrieved chunks are passed as context to the LLM along
   with the user's question. The LLM generates an answer grounded in the
   provided context.

## Benefits of RAG

- **Reduced hallucination**: The model answers based on actual documents, not
  just parametric knowledge.
- **Up-to-date information**: New documents can be ingested without retraining
  the model.
- **Source attribution**: Answers can be traced back to specific documents.
- **Cost-effective**: Avoids expensive fine-tuning for domain-specific knowledge.

## Common Components

- Document loaders (PDF, Markdown, code files)
- Text chunking / splitting strategies
- Embedding models (e.g. sentence-transformers, OpenAI embeddings)
- Vector databases (ChromaDB, Pinecone, Weaviate)
- LLMs for generation (Claude, GPT-4, open-source models)
