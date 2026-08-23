from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

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
    # Maps the database 'id' column to the API response 'job_id'
    job_id: UUID = Field(..., alias="id")
    document_id: UUID
    status: JobStatus
    stage: Optional[JobStage] = None
    attempts: int = 0
    progress: Optional[float] = Field(default=None, description="Progress percentage (0.0 to 1.0) if available")
    error: Optional[str] = Field(default=None, description="Safe error message if the job failed")
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)