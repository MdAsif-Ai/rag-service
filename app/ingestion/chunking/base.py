import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.ingestion.loaders.base import ParsedSection


class TextChunk(BaseModel):
    """
    Normalized representation of a document chunk ready for embedding.
    """
    chunk_id: str
    document_id: UUID
    course_id: str
    filename: str
    content: str
    page: Optional[int] = None
    section: Optional[str] = None
    chapter: Optional[str] = None
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.
    This makes the system extensible for semantic chunking later.
    """

    @abstractmethod
    def chunk(
        self,
        sections: List[ParsedSection],
        document_id: UUID,
        course_id: str,
        filename: str,
        chapter: Optional[str] = None
    ) -> List[TextChunk]:
        """
        Processes parsed sections and returns a list of TextChunks.
        """
        pass

    @staticmethod
    def generate_deterministic_id(document_id: UUID, chunk_index: int) -> str:
        """
        Generates a deterministic UUID for a chunk based on document_id and chunk_index.
        This ensures that re-indexing the same document overwrites existing vectors
        rather than creating duplicates.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}_{chunk_index}"))