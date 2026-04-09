from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "rag-pdf-index"
    pinecone_environment: str = "us-east-1"

    # Groq (LLM inference)
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    # Embeddings — fastembed runs locally, no API needed
    embed_model: str = "BAAI/bge-base-en-v1.5"   # 768-dim, matches existing Pinecone index

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k_results: int = 5

    # Query rewriting
    query_rewrite_enabled: bool = True

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_file_size_mb: int = 50

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()