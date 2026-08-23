from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobStage(str, Enum):
    UPLOADED = "UPLOADED"
    EXTRACTING = "EXTRACTING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    DONE = "DONE"
    FAILED = "FAILED"


class JobMetadata(BaseModel):
    id: UUID = Field(..., alias="job_id")
    document_id: UUID
    status: JobStatus
    stage: Optional[JobStage] = None
    attempts: int = 0
    error: Optional[str] = Field(default=None, description="Error message if the job failed")
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        populate_by_name = True