from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class JobStage(str, Enum):
    UPLOADED = "UPLOADED"
    DOWNLOADING = "DOWNLOADING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    DONE = "DONE"
    FAILED = "FAILED"

class JobMetadata(BaseModel):
    job_id: UUID
    document_id: UUID
    status: JobStatus
    stage: Optional[JobStage] = None
    attempts: int = 0
    progress: Optional[float] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)