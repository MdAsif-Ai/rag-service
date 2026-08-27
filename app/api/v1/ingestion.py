from typing import Optional
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
    file: Optional[UploadFile] = File(None, description="The document file to ingest"),
    course_id: str = Form(..., description="The course ID this document belongs to"),
    filename: str = Form(..., description="The original filename or video title"),
    chapter: Optional[str] = Form(None, description="Optional chapter association"),
    section: Optional[str] = Form(None, description="Optional section association"),
    source_type: Optional[str] = Form(None, description="Optional source type metadata"),
    content_format: Optional[str] = Form("auto", description="Hint: 'auto', 'handwritten', 'diagram', 'audio', 'youtube'"),
    url: Optional[str] = Form(None, description="URL for YouTube videos or web pages")
):
    """
    Accepts a document upload OR a URL, stores it, and queues an asynchronous ingestion job.
    """
    logger.info(f"Received ingestion request for course: {course_id}, file: {filename}")
    
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