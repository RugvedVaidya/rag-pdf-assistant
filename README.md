---
title: Docwise RAG PDF Assistant
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: Chat with your PDF documents using RAG + Groq + Pinecone
---

# Docwise — RAG PDF Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for querying PDF documents using **FastAPI**, **LangChain**, **Groq** (LLM), **FastEmbed** (local embeddings), and **Pinecone** (vector database).

💬 **Streaming answers** · 🧠 **Conversation memory** · 📊 **Table extraction** · ✍️ **Query rewriting**

---

## Architecture

```
PDFs → Text + Table Extraction → Chunking → FastEmbed (local) → Pinecone Index
                                                                        ↓
User Query → Query Rewriter (Groq) → Embed Query → Similarity Search + Dedup
                                                                        ↓
                                          Retrieved Chunks → Groq LLM → Streaming Answer
                                                                        ↓
                                                          SQLite (sessions + registry)
```

---

## Features

- **Streaming answers** — tokens stream word-by-word via Server-Sent Events
- **Conversation memory** — multi-turn chat with full history persisted to SQLite
- **Query rewriting** — vague follow-up questions are rewritten into better search queries before retrieval
- **Table extraction** — pdfplumber extracts tables from PDFs and stores them as structured chunks
- **Chunk deduplication** — overlapping chunks are deduplicated before sending to the LLM
- **Document registry** — SQLite tracks all indexed documents
- **Duplicate detection** — re-uploading a PDF automatically replaces the old version
- **Namespace support** — separate document collections in isolated Pinecone namespaces
- **Dark/light theme** — persistent theme preference
- **Export** — download any conversation as Markdown, JSON, or plain text

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API framework | FastAPI + Uvicorn |
| LLM (inference) | Groq — `llama-3.1-8b-instant` |
| Embeddings | FastEmbed — `BAAI/bge-base-en-v1.5` (local, 768-dim) |
| Vector database | Pinecone (serverless) |
| PDF processing | PyMuPDF + pdfplumber |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Session storage | SQLite |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |
| Deployment | HuggingFace Spaces (Docker) |

---

## Project Structure

```
rag-pdf-assistant/
├── app/
│   ├── main.py                      # FastAPI entry point, serves frontend
│   ├── api/
│   │   └── routes/
│   │       ├── ingest.py            # PDF upload & indexing endpoints
│   │       ├── query.py             # RAG query + SSE streaming endpoints
│   │       └── sessions.py          # Conversation session CRUD
│   ├── core/
│   │   ├── config.py                # Pydantic settings (.env)
│   │   ├── database.py              # SQLite connection + schema init
│   │   └── logging.py               # Structured logging (structlog)
│   ├── services/
│   │   ├── pdf_processor.py         # Text + table extraction, chunking
│   │   ├── embedder.py              # FastEmbed local embeddings
│   │   ├── vector_store.py          # Pinecone operations
│   │   ├── rag_chain.py             # Core RAG pipeline (Groq LLM)
│   │   ├── query_rewriter.py        # Query rewriting via Groq
│   │   ├── memory.py                # SQLite-backed session store
│   │   └── document_registry.py     # SQLite document metadata registry
│   └── models/
│       └── schemas.py               # Pydantic request/response models
├── frontend/
│   └── index.html                   # Full chat UI (served at /)
├── scripts/
│   └── ingest_bulk.py               # CLI for bulk PDF ingestion
├── data/
│   └── .gitkeep                     # SQLite DB created here at runtime
├── Dockerfile                       # Docker image for HF Spaces
├── render.yaml                      # Alternative Render deployment config
├── .env.example
├── requirements.txt
└── README.md
```

---

## Prerequisites

- A [Pinecone](https://pinecone.io) account — free tier works
- A [Groq](https://console.groq.com) API key — free tier works

> No GPU or local model required. Groq runs the LLM in the cloud and FastEmbed runs a small embedding model (~120MB) locally inside the container.

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-pdf-assistant.git
cd rag-pdf-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` — the UI is served directly by FastAPI.

---

## Deploying to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Visibility: Public
2. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/rag-pdf-assistant
   git push space main
   ```
3. Add secrets in Space **Settings → Variables and Secrets**:
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`
4. The Space builds automatically — first build takes ~5 minutes.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the chat frontend |
| `POST` | `/api/ingest/upload` | Upload and index a PDF |
| `DELETE` | `/api/ingest/delete/{doc_id}` | Remove a document |
| `GET` | `/api/ingest/list` | List indexed documents |
| `POST` | `/api/query/ask` | Ask a question (RAG) |
| `POST` | `/api/query/ask/stream` | Streaming SSE response |
| `POST` | `/api/query/search` | Raw similarity search |
| `POST` | `/api/sessions` | Create session |
| `GET` | `/api/sessions` | List sessions |
| `GET` | `/api/sessions/{id}` | Get session with history |
| `DELETE` | `/api/sessions/{id}` | Delete session |
| `POST` | `/api/sessions/{id}/clear` | Clear session history |
| `GET` | `/health` | Health check |

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Pinecone API key | required |
| `PINECONE_INDEX_NAME` | Pinecone index name | `rag-pdf-index` |
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Groq model ID | `llama-3.1-8b-instant` |
| `EMBED_MODEL` | FastEmbed model | `BAAI/bge-base-en-v1.5` |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RESULTS` | Default retrieval count | `5` |
| `QUERY_REWRITE_ENABLED` | Enable query rewriting | `true` |
| `DB_PATH` | SQLite database path | `data/docwise.db` |
| `MAX_FILE_SIZE_MB` | Max PDF upload size | `50` |

---

## SSE Stream Events

| Event | Payload | Description |
|-------|---------|-------------|
| `session` | `{session_id}` | Session ID assigned |
| `rewrite` | `{original, rewritten}` | Query was rewritten |
| `sources` | `[{filename, page, score}]` | Retrieved sources |
| `token` | `{text}` | LLM token |
| `done` | `{model, session_id}` | Stream complete |
| `error` | `{message}` | Error occurred |

---

## License

MIT