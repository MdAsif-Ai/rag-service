import functools
from typing import List, Literal, Optional, Union
from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.
    Reads from environment variables and optionally from a .env file.
    Required production secrets will cause startup failure if missing.
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
    API_PREFIX: str = Field(default="/api/v1")
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    
    # --- CORS ---
    # Accepts comma-separated string in .env (e.g., CORS_ORIGINS="http://localhost,http://api.lms.com")
    CORS_ORIGINS: List[str] = Field(default=["*"])

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # --- SECURITY ---
    RAG_SERVICE_API_KEY: SecretStr = Field(..., description="API key for service-to-service authentication")
    HEALTH_CHECK_API_KEY: Optional[SecretStr] = Field(default=None, description="Optional API key for health endpoints")

    # --- SUPABASE ---
    SUPABASE_URL: AnyHttpUrl
    SUPABASE_SERVICE_ROLE_KEY: SecretStr
    SUPABASE_STORAGE_BUCKET: str = Field(default="lms-rag-documents")

    # --- QDRANT ---
    QDRANT_URL: str
    QDRANT_API_KEY: SecretStr
    QDRANT_COLLECTION: str = Field(default="lms_knowledge_base")
    QDRANT_TIMEOUT: int = Field(default=10, description="Qdrant client timeout in seconds")

    # --- REDIS ---
    REDIS_URL: str = Field(default="redis://redis:6379/0")

    # --- CELERY ---
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

    CELERY_TASK_ACKS_LATE: bool = Field(default=True)
    CELERY_TASK_REJECT_ON_WORKER_LOST: bool = Field(default=True)
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = Field(default=1)
    CELERY_TASK_TIME_LIMIT: int = Field(default=1800, description="30 minutes hard limit")
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=1500, description="25 minutes soft limit")
    
    # --- RETRY LOGIC ---
    CELERY_MAX_RETRIES: int = Field(default=3)
    CELERY_RETRY_DELAY: int = Field(default=60, description="Base delay in seconds for exponential backoff")
    CELERY_RETRY_BACKOFF: bool = Field(default=True)
    CELERY_RETRY_BACKOFF_MAX: int = Field(default=300, description="Max delay in seconds for backoff")
    TENACITY_MAX_RETRIES: int = Field(default=3, description="Retries for sync HTTP operations like Supabase")
    TENACITY_RETRY_DELAY: int = Field(default=2, description="Base delay in seconds for Tenacity backoff")

    # --- EMBEDDING ---
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    EMBEDDING_DEVICE: Literal["cpu", "cuda"] = Field(default="cuda")

    # --- RERANKER ---
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
    RERANKER_DEVICE: Literal["cpu", "cuda"] = Field(default="cuda")
    RERANK_BATCH_SIZE: int = Field(default=16)
    RERANK_MAX_CANDIDATES: int = Field(default=30)
    RERANK_FINAL_TOP_K: int = Field(default=5)

    # --- RETRIEVAL ---
    DEFAULT_TOP_K: int = Field(default=50)
    RERANK_TOP_K: int = Field(default=10)
    FUSION_RRF_K: int = Field(default=60)
    FUSION_DENSE_WEIGHT: float = Field(default=1.0)
    FUSION_SPARSE_WEIGHT: float = Field(default=1.0)

    # --- INGESTION ---
    MAX_FILE_SIZE_MB: int = Field(default=50)
    CHUNK_SIZE_TOKENS: int = Field(default=600)
    CHUNK_OVERLAP_TOKENS: int = Field(default=100)
    CHUNK_TOKENIZER: str = Field(default="cl100k_base")
    CHUNK_INCLUDE_CONTEXT_PREFIX: bool = Field(default=True)
    SUPPORTED_FILE_TYPES: List[str] = Field(default=["pdf", "docx", "pptx", "txt", "md", "html"])

        # --- RETRIEVAL ---
    DENSE_TOP_K: int = Field(default=50, description="Candidate pool size for dense retrieval")
    SPARSE_TOP_K: int = Field(default=50, description="Candidate pool size for sparse retrieval")
    FUSION_TOP_K: int = Field(default=30, description="Number of fused candidates to pass to the reranker")
    FINAL_TOP_K: int = Field(default=5, description="Default number of final results to return if not specified by API")
    FUSION_RRF_K: int = Field(default=60, description="K parameter for Reciprocal Rank Fusion")
    FUSION_DENSE_WEIGHT: float = Field(default=1.0, description="Weight multiplier for dense retrieval ranks")
    FUSION_SPARSE_WEIGHT: float = Field(default=1.0, description="Weight multiplier for sparse retrieval ranks")
    RERANK_MAX_CANDIDATES: int = Field(default=30, description="Max candidates to rerank from the fused pool")
    RERANK_FINAL_TOP_K: int = Field(default=5, description="Default final top_k for reranker")

    @field_validator("SUPPORTED_FILE_TYPES", mode="before")
    @classmethod
    def parse_supported_types(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v

    # --- HTTP CLIENT ---
    HTTP_TIMEOUT: int = Field(default=10, description="Default timeout for outbound HTTP requests")


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    Subsequent calls will return the same instance, preventing
    repeated reads from the environment or .env file.
    """
    return Settings()