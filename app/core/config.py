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

    # --- RETRIEVAL ---
    DEFAULT_TOP_K: int = Field(default=50)
    RERANK_TOP_K: int = Field(default=10)
    FINAL_TOP_K: int = Field(default=5)

    # --- INGESTION ---
    MAX_FILE_SIZE_MB: int = Field(default=50)
    CHUNK_SIZE: int = Field(default=1024)
    CHUNK_OVERLAP: int = Field(default=100)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    Subsequent calls will return the same instance, preventing
    repeated reads from the environment or .env file.
    """
    return Settings()