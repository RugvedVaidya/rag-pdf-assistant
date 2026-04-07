# RAG PDF Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for querying PDF documents using **FastAPI**, **LangChain**, **Pinecone**, and **Ollama** (local LLMs).

---

## Architecture

```
PDFs → Text Extraction → Chunking → Embeddings (Ollama) → Pinecone Index
                                                                   ↓
User Query → Embed Query → Similarity Search → Retrieved Chunks → Ollama LLM → Answer
```

---

## Project Structure

```
rag-pdf-assistant/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   └── routes/
│   │       ├── ingest.py        # PDF upload & indexing endpoints
│   │       └── query.py         # Question answering endpoints
│   ├── core/
│   │   ├── config.py            # Settings & environment variables
│   │   └── logging.py           # Structured logging setup
│   ├── services/
│   │   ├── pdf_processor.py     # PDF text extraction & chunking
│   │   ├── embedder.py          # Ollama embedding service
│   │   ├── vector_store.py      # Pinecone operations
│   │   └── rag_chain.py         # LangChain RAG pipeline
│   └── models/
│       └── schemas.py           # Pydantic request/response models
├── scripts/
│   └── ingest_bulk.py           # CLI script for bulk PDF ingestion
├── tests/
│   ├── test_ingest.py
│   └── test_query.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A [Pinecone](https://pinecone.io) account (free tier works)

### Pull required Ollama models

```bash
# LLM for answering
ollama pull llama3.2

# Embedding model
ollama pull nomic-embed-text
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd rag-pdf-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Pinecone credentials
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## Usage

### Upload & index a PDF

```bash
curl -X POST http://localhost:8000/api/ingest/upload \
  -F "file=@your_document.pdf" \
  -F "namespace=my-docs"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/query/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings?",
    "namespace": "my-docs",
    "top_k": 5
  }'
```

### Bulk ingest from a folder

```bash
python scripts/ingest_bulk.py --folder ./pdfs --namespace my-docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ingest/upload` | Upload and index a PDF |
| `DELETE` | `/api/ingest/delete/{doc_id}` | Remove a document from index |
| `GET` | `/api/ingest/list` | List indexed documents |
| `POST` | `/api/query/ask` | Ask a question (RAG) |
| `POST` | `/api/query/search` | Raw similarity search (no LLM) |
| `GET` | `/health` | Health check |

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key | required |
| `PINECONE_INDEX_NAME` | Pinecone index name | `rag-pdf-index` |
| `PINECONE_ENVIRONMENT` | Pinecone environment | `us-east-1` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_LLM_MODEL` | LLM model name | `llama3.2` |
| `OLLAMA_EMBED_MODEL` | Embedding model name | `nomic-embed-text` |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RESULTS` | Default retrieval count | `5` |

---

## How RAG Works Here

1. **Ingestion**: PDFs are parsed with `PyMuPDF`, split into overlapping chunks via LangChain's `RecursiveCharacterTextSplitter`, then embedded using Ollama's `nomic-embed-text` model and stored in Pinecone with metadata (filename, page number, chunk index).

2. **Querying**: The user's question is embedded with the same model, a cosine similarity search retrieves the top-k most relevant chunks from Pinecone, and these are injected into a prompt sent to the local Ollama LLM, which generates a grounded answer.

3. **Metadata**: Each answer includes source citations (filename + page number) so you know exactly where the answer came from.