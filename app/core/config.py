import functools
from typing import List, Literal, Optional
from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")

    APP_NAME: str = Field(default="LMS RAG Service")
    APP_ENV: Literal["development", "staging", "production"] = Field(default="production")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    API_PREFIX: str = Field(default="/api/v1")
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    CORS_ORIGINS: List[str] = Field(default=["*"])

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    RAG_SERVICE_API_KEY: SecretStr = Field(...)
    HEALTH_CHECK_API_KEY: Optional[SecretStr] = Field(default=None)

    SUPABASE_URL: AnyHttpUrl
    SUPABASE_SERVICE_ROLE_KEY: SecretStr
    SUPABASE_STORAGE_BUCKET: str = Field(default="lms-rag-documents")

    QDRANT_URL: str
    QDRANT_API_KEY: SecretStr
    QDRANT_COLLECTION: str = Field(default="lms_knowledge_base")
    QDRANT_TIMEOUT: int = Field(default=10)

    REDIS_URL: str = Field(default="redis://redis:6379/0")
    CELERY_BROKER_URL: str = Field(default="redis://redis:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://redis:6379/2")
    CELERY_TASK_SERIALIZER: str = Field(default="json")
    CELERY_RESULT_SERIALIZER: str = Field(default="json")
    CELERY_ACCEPT_CONTENT: List[str] = Field(default=["json"])

    @field_validator("CELERY_ACCEPT_CONTENT", mode="before")
    @classmethod
    def parse_celery_accept(cls, v):
        if isinstance(v, str):
            return [content.strip() for content in v.split(",")]
        return v

    CELERY_TASK_TIME_LIMIT: int = Field(default=7200, description="2 hours hard limit")
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=6900, description="1 hour 55 minutes soft limit")

    CELERY_TASK_ACKS_LATE: bool = Field(default=True)
    CELERY_TASK_REJECT_ON_WORKER_LOST: bool = Field(default=True)
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = Field(default=1)
    CELERY_MAX_RETRIES: int = Field(default=3)
    CELERY_RETRY_DELAY: int = Field(default=60)
    TENACITY_MAX_RETRIES: int = Field(default=3)
    TENACITY_RETRY_DELAY: int = Field(default=2)

    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    EMBEDDING_DEVICE: Literal["cpu", "cuda"] = Field(default="cuda")
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
    RERANKER_DEVICE: Literal["cpu", "cuda"] = Field(default="cuda")
    RERANK_BATCH_SIZE: int = Field(default=16)
    RERANK_MAX_CANDIDATES: int = Field(default=30)
    RERANK_FINAL_TOP_K: int = Field(default=5)

    DEFAULT_TOP_K: int = Field(default=50)
    SPARSE_TOP_K: int = Field(default=50)
    FUSION_TOP_K: int = Field(default=30)
    FINAL_TOP_K: int = Field(default=20)
    FUSION_RRF_K: int = Field(default=60)
    FUSION_DENSE_WEIGHT: float = Field(default=1.0)
    FUSION_SPARSE_WEIGHT: float = Field(default=1.0)
    DENSE_TOP_K: int = Field(default=20)


    MAX_FILE_SIZE_MB: int = Field(default=50)
    CHUNK_SIZE_TOKENS: int = Field(default=600)
    CHUNK_OVERLAP_TOKENS: int = Field(default=100)
    CHUNK_TOKENIZER: str = Field(default="cl100k_base")
    SUPPORTED_FILE_TYPES: List[str] = Field(default=["pdf", "docx", "pptx", "txt", "md", "html", "mp3", "wav", "m4a", "mp4", "mkv", "avi", "png", "jpg", "jpeg"])

    @field_validator("SUPPORTED_FILE_TYPES", mode="before")
    @classmethod
    def parse_supported_types(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v

    HTTP_TIMEOUT: int = Field(default=10)

@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()