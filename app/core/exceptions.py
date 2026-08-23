from typing import Optional, Any


class RAGServiceException(Exception):
    """
    Base exception for all RAG service errors.
    Designed to separate safe, client-facing messages from internal debug details.
    """
    default_message: str = "An internal server error occurred."
    status_code: int = 500

    def __init__(self, message: Optional[str] = None, detail: Optional[Any] = None):
        self.message = message or self.default_message
        # 'detail' is for internal logging/debugging only. 
        # It MUST NOT be sent directly to the client in production responses.
        self.detail = detail
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


# --- 1. Configuration Errors ---
class ConfigurationException(RAGServiceException):
    """Raised when application settings or environment variables are missing or invalid."""
    default_message = "Service configuration error."
    status_code = 500


# --- 2. Validation Errors ---
class ValidationException(RAGServiceException):
    """Raised when input data fails validation (e.g., invalid request payload)."""
    default_message = "Validation failed."
    status_code = 422


# --- 3. Document Processing Errors ---
class DocumentProcessingException(RAGServiceException):
    """Raised when a document fails to parse or process correctly."""
    default_message = "Failed to process the document."
    status_code = 422


class UnsupportedFileException(DocumentProcessingException):
    """Raised when a file type is not supported by the ingestion pipeline."""
    default_message = "Unsupported file type."
    status_code = 415


# --- 4. Embedding Errors ---
class EmbeddingException(RAGServiceException):
    """Raised when the embedding model fails to generate vectors."""
    default_message = "Failed to generate embeddings."
    status_code = 500


# --- 5. Infrastructure Errors ---
class QdrantException(RAGServiceException):
    """Raised for Qdrant database connection, querying, or upserting failures."""
    default_message = "Vector database operation failed."
    status_code = 503


class SupabaseException(RAGServiceException):
    """Raised for Supabase storage or database failures."""
    default_message = "Metadata or file storage operation failed."
    status_code = 503


# --- 6. Pipeline Errors ---
class IngestionException(RAGServiceException):
    """Raised when the general ingestion pipeline encounters an unrecoverable error."""
    default_message = "Ingestion pipeline failed."
    status_code = 500


class RetrievalException(RAGServiceException):
    """Raised when the hybrid retrieval or context construction fails."""
    default_message = "Failed to retrieve context."
    status_code = 500


# --- 7. Job Errors ---
class JobException(RAGServiceException):
    """Raised when a background Celery job fails to execute or queue properly."""
    default_message = "Background job failed."
    status_code = 500