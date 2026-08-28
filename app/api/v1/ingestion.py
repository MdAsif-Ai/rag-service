from typing import Optional
from fastapi import APIRouter, Depends, Request, UploadFile, Form, status
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
    request: Request,
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
    # Safely extract file to avoid Swagger empty string crash
    form = await request.form()
    file = form.get("file")
    if not isinstance(file, UploadFile):
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