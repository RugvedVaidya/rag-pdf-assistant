import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaEmbedder:
    """Generates embeddings using Ollama's local embedding models."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_embed_model
        self._client = httpx.Client(timeout=60.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        response = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts in batches."""
        embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            logger.info("embedding_batch", start=i, end=min(i + batch_size, total), total=total)
            for text in batch:
                embeddings.append(self.embed(text))

        return embeddings

    def check_health(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def __del__(self):
        self._client.close()