from typing import Optional, Union
from fastapi import APIRouter, Depends, UploadFile, File, Form, status
from loguru import logger

from app.core.security import verify_api_key
from app.schemas.ingestion import IngestionResponse, IngestionMetadata
from app.services.ingestion import IngestionService

router = APIRouter()

@router.post(
    "/ingest", 
    response_model=IngestionResponse, 
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)]
)
async def ingest_document(
    file: Optional[Union[UploadFile, str]] = File(None, description="The document file to ingest"),
    course_id: str = Form(..., description="The course ID this document belongs to"),
    filename: Optional[str] = Form(None),
    chapter: Optional[str] = Form(None),
    section: Optional[str] = Form(None),
    source_type: Optional[str] = Form(None),
    content_format: Optional[str] = Form("auto"),
    url: Optional[str] = Form(None)
):
    """
    Accepts a document upload OR a URL, stores it, and queues an asynchronous ingestion job.
    """
    # Safely handle Swagger empty string bug
    if isinstance(file, str):
        file = None
        
    metadata = IngestionMetadata(
        course_id=course_id,
        filename=filename,
        chapter=chapter,
        section=section,
        source_type=source_type,
        content_format=content_format,
        url=url
    )
    
    return await IngestionService.process_upload(file=file, metadata=metadata)