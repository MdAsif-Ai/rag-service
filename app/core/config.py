import functools
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.
    Reads from environment variables and optionally from a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- APPLICATION ---
    APP_NAME: str = Field(default="LMS RAG Service")
    APP_ENV: Literal["development", "staging", "production"] = Field(default="production")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # --- API ---
    API_PREFIX: str = Field(default="/v1")
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    # --- SUPABASE ---
    SUPABASE_URL: AnyHttpUrl
    SUPABASE_SERVICE_ROLE_KEY: SecretStr
    SUPABASE_STORAGE_BUCKET: str

    # --- QDRANT ---
    QDRANT_URL: str
    QDRANT_API_KEY: SecretStr
    QDRANT_COLLECTION: str

    # --- REDIS ---
    REDIS_URL: str

    # --- CELERY ---
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # --- EMBEDDING ---
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    EMBEDDING_DEVICE: Literal["cpu", "cuda"] = Field(default="cuda")

    # --- RERANKER ---
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
    RERANKER_DEVICE: Literal["cpu", "cuda"] = Field(default="cuda")

        # --- RERANKER ---
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
    RERANKER_DEVICE: Literal["cpu", "cuda"] = Field(default="cuda")
    RERANK_BATCH_SIZE: int = Field(default=16, description="Batch size for model inference")
    RERANK_MAX_CANDIDATES: int = Field(default=30, description="Max candidates to rerank from the fused pool")
    RERANK_FINAL_TOP_K: int = Field(default=5, description="Default number of final results to return if not specified by API")

    # --- RETRIEVAL ---
    DEFAULT_TOP_K: int = Field(default=50)
    RERANK_TOP_K: int = Field(default=10)
    FINAL_TOP_K: int = Field(default=5)

    # --- INGESTION ---
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_FILE_TYPES: list[str] = ["pdf", "docx", "pptx", "txt", "md", "html"]
    # Chunking Configuration (Token-based)
    CHUNK_SIZE_TOKENS: int = Field(default=600, description="Target tokens per chunk (400-700 recommended)")
    CHUNK_OVERLAP_TOKENS: int = Field(default=100, description="Overlap tokens (50-100 recommended)")
    CHUNK_TOKENIZER: str = Field(default="cl100k_base", description="Tiktoken encoder name")
    CHUNK_INCLUDE_CONTEXT_PREFIX: bool = Field(default=True, description="Prepend heading hierarchy to chunk content")

        # --- FUSION ---
    FUSION_RRF_K: int = Field(default=60, description="K parameter for Reciprocal Rank Fusion")
    FUSION_DENSE_WEIGHT: float = Field(default=1.0, description="Weight multiplier for dense retrieval ranks")
    FUSION_SPARSE_WEIGHT: float = Field(default=1.0, description="Weight multiplier for sparse retrieval ranks")


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    Subsequent calls will return the same instance, preventing
    repeated reads from the environment or .env file.
    """
    return Settings()