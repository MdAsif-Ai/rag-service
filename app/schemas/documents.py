from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

from .jobs import JobMetadata

class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    DELETED = "DELETED"

class DocumentMetadata(BaseModel):
    document_id: UUID = Field(..., alias="id")
    course_id: str
    filename: str
    file_type: str
    file_size: int
    storage_path: str
    checksum: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

class DocumentDetail(DocumentMetadata):
    latest_job: Optional[JobMetadata] = None