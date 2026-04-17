"""Main entry point for the RAG docs bot.

Usage:
    python src/main.py --ingest ./sample_docs
    python src/main.py --query "What is RAG?"
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from src.generation.llm import generate_answer
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_directory
from src.retrieval.embeddings import embed_and_store
from src.retrieval.store import get_client, get_or_create_collection, query_collection

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"


def ingest(docs_path: str) -> None:
    """Ingest documents from a directory into the vector store.

    Args:
        docs_path: Path to the directory containing documents.
    """
    print(f"Loading documents from {docs_path}...")
    documents = load_directory(docs_path)
    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} document(s).")

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], metadata=doc["metadata"])
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunk(s).")

    client = get_client(CHROMA_DIR)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    count = embed_and_store(all_chunks, collection)
    print(f"Stored {count} chunk(s) in ChromaDB.")


def query(question: str) -> None:
    """Query the vector store and generate an answer.

    Args:
        question: The user's question.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    client = get_client(CHROMA_DIR)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    if collection.count() == 0:
        print("No documents in store. Run --ingest first.")
        sys.exit(1)

    print(f"Searching for: {question}")
    results = query_collection(collection, question, top_k=5)

    print(f"Found {len(results)} relevant chunk(s). Generating answer...\n")
    answer = generate_answer(question, results, api_key=api_key)
    print(answer)


def main():
    """Parse CLI arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(
        description="RAG Docs Bot - ingest documents and ask questions"
    )
    parser.add_argument(
        "--ingest",
        type=str,
        help="Path to directory of documents to ingest",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask about the ingested documents",
    )

    args = parser.parse_args()

    if args.ingest:
        ingest(args.ingest)
    elif args.query:
        query(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
