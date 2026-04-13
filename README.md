# rag-docs-bot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about your codebase and documentation.

## Features

- Document ingestion (Markdown, code files)
- Vector embedding and storage via ChromaDB
- Semantic retrieval + LLM-powered answers
- FastAPI backend with streaming responses
- React/Next.js chat frontend

## Tech Stack

- **Python 3.11+** — core pipeline
- **LangChain** — RAG orchestration
- **ChromaDB** — vector database (local)
- **Claude API** — LLM for answer generation
- **FastAPI** — backend API
- **Next.js + Tailwind** — frontend

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-docs-bot.git
cd rag-docs-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run
python src/main.py
```

## Project Structure

```
rag-docs-bot/
├── src/              # Source code
│   ├── ingestion/    # Document loading + chunking
│   ├── retrieval/    # Vector search + retrieval
│   ├── generation/   # LLM answer generation
│   └── api/          # FastAPI endpoints
├── tests/            # Test suite
├── docs/             # Documentation
├── frontend/         # React/Next.js chat UI
├── requirements.txt
└── README.md
```

## License

MIT
