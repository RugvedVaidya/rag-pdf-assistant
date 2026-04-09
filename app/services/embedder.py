"""
Embedding service using FastEmbed — runs fully locally, no API key needed.
Uses BAAI/bge-small-en-v1.5 which produces 768-dim vectors, same as the
previous nomic-embed-text model, so existing Pinecone indexes are compatible.

FastEmbed downloads the model on first use (~130MB) and caches it locally.
On Render, the model is cached in the Docker image via requirements install.
"""
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    """Local embedding using FastEmbed — zero API cost, zero latency overhead."""

    def __init__(self):
        settings = get_settings()
        self.model_name = settings.embed_model
        self._model = self._load_model()

    def _load_model(self):
        try:
            from fastembed import TextEmbedding
            logger.info("loading_embed_model", model=self.model_name)
            model = TextEmbedding(model_name=self.model_name)
            logger.info("embed_model_loaded", model=self.model_name)
            return model
        except ImportError:
            raise RuntimeError("fastembed not installed. Run: pip install fastembed")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Embed a list of texts.
        FastEmbed handles batching internally and is significantly faster
        than calling embed() in a loop.
        """
        logger.info("embedding_batch", total=len(texts))
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]

    def check_health(self) -> bool:
        try:
            test = self.embed("health check")
            return len(test) > 0
        except Exception:
            return False


# Keep OllamaEmbedder as an alias so any existing imports don't break
OllamaEmbedder = Embedder


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()