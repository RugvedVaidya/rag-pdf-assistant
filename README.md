# RAG PDF Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for querying PDF documents using **FastAPI**, **LangChain**, **Pinecone**, and **Ollama** (local LLMs).

---

## Architecture

```
PDFs → Text Extraction → Chunking → Embeddings (Ollama) → Pinecone Index
                                                                   ↓
User Query → Embed Query → Similarity Search → Retrieved Chunks → Ollama LLM → Answer
```

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
git clone <repo-link>
cd rag-pdf-assistant
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

## How RAG Works Here

1. **Ingestion**: PDFs are parsed with `PyMuPDF`, split into overlapping chunks via LangChain's `RecursiveCharacterTextSplitter`, then embedded using Ollama's `nomic-embed-text` model and stored in Pinecone with metadata (filename, page number, chunk index).

2. **Querying**: The user's question is embedded with the same model, a cosine similarity search retrieves the top-k most relevant chunks from Pinecone, and these are injected into a prompt sent to the local Ollama LLM, which generates a grounded answer.

3. **Metadata**: Each answer includes source citations (filename + page number) so you know exactly where the answer came from.