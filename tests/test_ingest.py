import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.services.embedder.OllamaEmbedder"), \
         patch("app.services.vector_store.VectorStore"), \
         patch("app.core.config.Settings.pinecone_api_key", "test-key"):
        from app.main import app
        return TestClient(app)


def make_minimal_pdf() -> bytes:
    """Return a tiny valid-ish PDF for testing."""
    return b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""


def test_upload_non_pdf(client):
    response = client.post(
        "/api/ingest/upload",
        data={"namespace": "test"},
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_health_endpoint(client):
    with patch("app.services.embedder.OllamaEmbedder.check_health", return_value=True), \
         patch("app.services.vector_store.VectorStore.check_health", return_value=True):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ollama" in data
        assert "pinecone" in data