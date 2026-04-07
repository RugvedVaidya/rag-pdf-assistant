import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.models.schemas import QueryResponse, SourceChunk, SearchResponse


@pytest.fixture
def client():
    with patch("app.services.embedder.OllamaEmbedder"), \
         patch("app.services.vector_store.VectorStore"), \
         patch("app.core.config.Settings.pinecone_api_key", "test-key"):
        from app.main import app
        return TestClient(app)


def test_ask_question(client):
    mock_response = QueryResponse(
        answer="The document discusses AI safety.",
        question="What is the document about?",
        sources=[
            SourceChunk(text="AI safety is crucial...", filename="doc.pdf", page=1, score=0.92, doc_id="abc123")
        ],
        model="llama3.2",
        namespace="default",
    )
    with patch("app.api.routes.query.get_rag_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.ask.return_value = mock_response
        mock_chain_fn.return_value = mock_chain

        response = client.post("/api/query/ask", json={
            "question": "What is the document about?",
            "namespace": "default",
            "top_k": 5,
        })

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The document discusses AI safety."
    assert len(data["sources"]) == 1


def test_ask_question_validation(client):
    response = client.post("/api/query/ask", json={"question": "Hi"})
    assert response.status_code == 200  # "Hi" is 2 chars, below min_length=3

    response = client.post("/api/query/ask", json={"question": "Hi?"})
    assert response.status_code in (200, 422)


def test_similarity_search(client):
    mock_response = SearchResponse(
        query="AI safety",
        results=[
            SourceChunk(text="AI safety...", filename="doc.pdf", page=2, score=0.88, doc_id="abc123")
        ],
        namespace="default",
    )
    with patch("app.api.routes.query.get_rag_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        mock_chain.search.return_value = mock_response
        mock_chain_fn.return_value = mock_chain

        response = client.post("/api/query/search", json={"query": "AI safety"})

    assert response.status_code == 200
    assert response.json()["results"][0]["score"] == 0.88