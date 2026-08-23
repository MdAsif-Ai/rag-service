from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    DELETED = "DELETED"


class DocumentBase(BaseModel):
    course_id: str = Field(..., min_length=1, description="The course this document belongs to")
    filename: str = Field(..., min_length=1)
    source_type: str = Field(..., description="e.g., pdf, docx, youtube, url")
    file_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata extracted from the file or provided by teacher")


class DocumentCreate(DocumentBase):
    pass


class DocumentMetadata(DocumentBase):
    id: UUID = Field(..., alias="document_id")
    status: DocumentStatus
    chunk_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True