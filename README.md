# Docwise — RAG PDF Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for querying PDF documents using **FastAPI**, **LangChain**, **Groq** (LLM), **FastEmbed** (local embeddings), and **Pinecone** (vector database).

🚀 **Deployed on Render** · 💬 **Streaming answers** · 🧠 **Conversation memory** · 📊 **Table extraction** · ✍️ **Query rewriting**

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
- **Document registry** — SQLite tracks all indexed documents (no unreliable Pinecone list queries)
- **Duplicate detection** — re-uploading a PDF automatically replaces the old version
- **Namespace support** — separate document collections in isolated Pinecone namespaces
- **Bulk ingestion CLI** — ingest entire folders of PDFs from the command line
- **Dark/light theme** — persistent theme preference saved to localStorage
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
| Session storage | SQLite (via Python stdlib) |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |
| Deployment | Render (Docker) |

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
├── tests/
│   ├── test_ingest.py
│   └── test_query.py
├── data/
│   └── .gitkeep                     # SQLite DB created here at runtime
├── Dockerfile                       # Docker image for Render
├── render.yaml                      # Render deployment config
├── .env.example
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- A [Pinecone](https://pinecone.io) account — free tier works
- A [Groq](https://console.groq.com) API key — free tier works

> No GPU or local model required. Groq runs the LLM in the cloud and FastEmbed runs a small embedding model (~120MB) locally.

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

Edit `.env` with your credentials:

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

## Deployment on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repository — Render detects `render.yaml` automatically
4. Go to **Environment** tab and add:
   - `PINECONE_API_KEY` — your Pinecone key
   - `GROQ_API_KEY` — your Groq key
5. Click **Deploy**

First deploy takes ~5 minutes (Docker build + FastEmbed model download).
Your app will be live at `https://rag-pdf-assistant.onrender.com`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the chat frontend |
| `POST` | `/api/ingest/upload` | Upload and index a PDF |
| `DELETE` | `/api/ingest/delete/{doc_id}` | Remove a document from Pinecone + registry |
| `GET` | `/api/ingest/list` | List indexed documents (from SQLite) |
| `POST` | `/api/query/ask` | Ask a question (RAG, non-streaming) |
| `POST` | `/api/query/ask/stream` | Ask a question (SSE streaming) |
| `POST` | `/api/query/search` | Raw similarity search (no LLM) |
| `POST` | `/api/sessions` | Create a new conversation session |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/sessions/{id}` | Get session with full turn history |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `POST` | `/api/sessions/{id}/clear` | Clear a session's history |
| `GET` | `/health` | Health check (Groq + Pinecone status) |

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Pinecone API key | required |
| `PINECONE_INDEX_NAME` | Pinecone index name | `rag-pdf-index` |
| `PINECONE_ENVIRONMENT` | Pinecone region | `us-east-1` |
| `GROQ_API_KEY` | Groq API key | required |
| `GROQ_MODEL` | Groq model ID | `llama-3.1-8b-instant` |
| `EMBED_MODEL` | FastEmbed model name | `BAAI/bge-base-en-v1.5` |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RESULTS` | Default retrieval count | `5` |
| `QUERY_REWRITE_ENABLED` | Enable query rewriting | `true` |
| `MAX_FILE_SIZE_MB` | Max PDF upload size | `50` |

---

## How It Works

### Ingestion
1. PDF uploaded to `/api/ingest/upload`
2. PyMuPDF extracts text per page; pdfplumber extracts tables
3. Tables are formatted as markdown and stored as separate chunks with `chunk_type=table`
4. Text chunks are split using `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
5. All chunks are embedded using FastEmbed (`BAAI/bge-base-en-v1.5`, 768-dim, runs locally)
6. Vectors + metadata upserted to Pinecone; document registered in SQLite

### Querying
1. User question + conversation history passed to the **query rewriter** (Groq)
2. Rewritten query is embedded with FastEmbed
3. Top-(k+3) chunks retrieved from Pinecone via cosine similarity
4. Near-duplicate chunks removed (>60% word overlap or same-page adjacent chunks)
5. Top-k deduplicated chunks built into numbered context
6. Prompt assembled with context + history and streamed through Groq LLM
7. Tokens streamed to frontend via SSE; turn saved to SQLite on completion

### SSE Events (streaming)
| Event | Payload | Description |
|-------|---------|-------------|
| `session` | `{session_id}` | Session ID assigned |
| `rewrite` | `{original, rewritten}` | Query was rewritten (when applicable) |
| `sources` | `[{filename, page, score, chunk_type}]` | Retrieved sources |
| `token` | `{text}` | LLM token |
| `done` | `{model, session_id, was_rewritten}` | Stream complete |
| `error` | `{message}` | Error occurred |

---

## Bulk Ingestion CLI

```bash
python scripts/ingest_bulk.py --folder ./pdfs --namespace my-docs
python scripts/ingest_bulk.py --folder ./pdfs --namespace my-docs --recursive
```

---

## License

MIT