"""FastAPI routes (stub for future expansion)."""

from fastapi import FastAPI

app = FastAPI(title="rag-docs-bot", version="0.1.0")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/query")
def query_docs(query: str):
    """Query the document store (stub).

    This will be wired up to the full RAG pipeline in a future phase.
    """
    return {"query": query, "answer": "Not implemented yet."}
