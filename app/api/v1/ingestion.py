from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, status
from loguru import logger

from app.core.security import verify_api_key
from app.schemas.ingestion import IngestionResponse
from app.services.ingestion import IngestionService

router = APIRouter()

@router.post(
    "/ingest", 
    response_model=IngestionResponse, 
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)]
)
async def ingest_document(
    file: UploadFile = File(..., description="The document file to ingest"),
    course_id: str = Form(..., description="The course ID this document belongs to"),
    chapter: Optional[str] = Form(None, description="Optional chapter association"),
    section: Optional[str] = Form(None, description="Optional section association"),
    source_type: Optional[str] = Form(None, description="Optional source type metadata")
):
    """
    Accepts a document upload, stores it, and queues an asynchronous ingestion job.
    Requires a valid API key.
    """
    logger.info(f"Received ingestion request for course: {course_id}, file: {file.filename}")
    
    # Delegate entirely to the service layer. 
    # The service handles validation, storage, DB records, and Celery queueing.
    # FastAPI's exception handlers in main.py will catch any HTTPExceptions raised.
    return await IngestionService.process_upload(
        file=file,
        course_id=course_id,
        chapter=chapter,
        section=section,
        source_type=source_type
    )