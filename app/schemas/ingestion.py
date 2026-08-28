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
    filename: Optional[str] = Field(default=None)
    chapter: Optional[str] = None
    section: Optional[str] = None
    source_type: Optional[str] = None
    content_format: Optional[str] = Field(
        default="auto", 
        description="Hint for parser: e.g., 'auto', 'handwritten', 'diagram', 'scanned'"
    )
    url: Optional[str] = Field(
        default=None, 
        description="URL for YouTube videos or web pages"
    )

class IngestionResponse(BaseModel):
    document_id: UUID
    job_id: UUID
    status: JobStatus