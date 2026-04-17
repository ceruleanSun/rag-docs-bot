# Architecture Overview

## Pipeline

```
Documents  -->  Loader  -->  Chunker  -->  Embedder  -->  ChromaDB
                                                              |
User Query  ------------------------------------------>  Retrieval
                                                              |
                                                        Top-K chunks
                                                              |
                                                     Claude LLM (Haiku)
                                                              |
                                                          Answer
```

## Components

### Ingestion (`src/ingestion/`)
- **loader.py** -- reads Markdown, Python, and text files from disk.
- **chunker.py** -- splits documents into overlapping chunks (default 500 chars, 50 overlap).

### Retrieval (`src/retrieval/`)
- **store.py** -- manages a persistent ChromaDB collection (local `./chroma_db/`).
- **embeddings.py** -- uses ChromaDB's built-in default embedding model to embed and store chunks.

### Generation (`src/generation/`)
- **llm.py** -- sends retrieved context + user query to Claude (Haiku) and returns the answer.

### API (`src/api/`)
- **routes.py** -- FastAPI stub with `/health` and `/query` endpoints (to be wired up in Phase 2).

### Entry Point
- **main.py** -- CLI interface: `--ingest` to load docs, `--query` to ask questions.

## Data Flow

1. User runs `python src/main.py --ingest ./sample_docs`
2. Loader reads all `.md` / `.py` / `.txt` files from the directory.
3. Chunker splits each document into overlapping chunks.
4. Chunks are embedded and stored in ChromaDB.
5. User runs `python src/main.py --query "What is RAG?"`
6. Query is embedded, top-K similar chunks are retrieved from ChromaDB.
7. Chunks + query are sent to Claude Haiku, which generates a grounded answer.
