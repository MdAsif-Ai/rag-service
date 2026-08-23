from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .jobs import JobStatus


class IngestionMetadata(BaseModel):
    """
    Metadata accompanying a file upload.
    Sent as form data alongside the file in the multipart request.
    """
    course_id: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    file_metadata: Optional[Dict[str, Any]] = None


class IngestionResponse(BaseModel):
    document_id: UUID
    job_id: UUID
    status: JobStatus