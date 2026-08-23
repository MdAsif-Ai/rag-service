from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, Field

class ParsedSection(BaseModel):
    """Normalized internal representation of a document block from a loader."""
    content: str
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    content_type: str = "text"  # text, table, code, list
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TextChunk(BaseModel):
    """Final chunk representation ready for embedding."""
    chunk_id: str
    document_id: UUID
    course_id: str
    filename: str
    content: str
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    chunk_index: int
    content_type: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)