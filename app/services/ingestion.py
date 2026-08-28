import hashlib
import uuid
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from loguru import logger

from app.core.config import get_settings
from app.db.supabase import get_supabase_client
from app.storage.supabase import storage_service
from app.schemas.ingestion import IngestionResponse, IngestionMetadata
from app.schemas.jobs import JobStatus
from app.jobs.ingestion import ingest_document

def clean_param(value: Optional[str]) -> Optional[str]:
    """Helper to clear out Swagger 'string' placeholders and empty strings."""
    if not value or value.strip() == "" or value.lower() == "string":
        return None
    return value

class IngestionService:
    @staticmethod
    async def process_upload(
        file: Optional[UploadFile],
        metadata: IngestionMetadata
    ) -> IngestionResponse:
        settings = get_settings()
        supabase = get_supabase_client()
        
        course_id = metadata.course_id
        
        # Clean up placeholder strings and empty strings from Swagger UI
        filename = clean_param(metadata.filename)
        chapter = clean_param(metadata.chapter)
        section = clean_param(metadata.section)
        source_type = clean_param(metadata.source_type)
        content_format = clean_param(metadata.content_format) or "auto"
        url = clean_param(metadata.url)

        # Smart URL check: Only treat it as a URL if it actually starts with http
        if url and not url.startswith("http"):
            url = None

        # Check if a physical file was actually provided (Swagger sends empty string for file if none selected)
        has_file = file is not None and file.filename and file.filename.strip() != ""

        # Handle Physical File vs URL
        if url:
            file_type = "youtube"
            file_size = 0
            checksum = hashlib.sha256(url.encode()).hexdigest()
            storage_path = url
            if not filename:
                filename = "youtube_video"
        elif has_file:
            # If filename was blank or 'string', use the actual uploaded file's name
            if not filename:
                filename = file.filename or "uploaded_file"
                
            ext = filename.split(".")[-1].lower() if "." in filename else ""
            if ext not in settings.SUPPORTED_FILE_TYPES:
                raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Unsupported file type: .{ext}")

            file_type = ext
            file_bytes = await file.read()
            if len(file_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File size exceeds maximum limit")

            file_size = len(file_bytes)
            checksum = hashlib.sha256(file_bytes).hexdigest()
            storage_path = "" 
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                detail="No file provided and no valid URL provided. You must provide one."
            )

        # 4. Duplicate Detection
        existing = supabase.table("documents").select("id").eq("checksum", checksum).eq("course_id", course_id).execute()
        if existing.data:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document already exists for this course.")

        # 5. Create Database Records
        doc_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        
        try:
            supabase.table("documents").insert({
                "id": doc_id,
                "course_id": course_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": file_size,
                "storage_path": storage_path, 
                "checksum": checksum,
                "status": "PENDING"
            }).execute()
            
            supabase.table("ingestion_jobs").insert({
                "id": job_id,
                "document_id": doc_id,
                "status": JobStatus.QUEUED,
                "stage": "UPLOADED"
            }).execute()
        except Exception as e:
            logger.error(f"Database failure during ingestion: {e}")
            raise HTTPException(status_code=500, detail="Failed to create ingestion records.")

        # 6. Upload to Storage (Only if it's a physical file)
        if has_file and not url:
            try:
                storage_path = storage_service.upload_file(file_bytes, uuid.UUID(doc_id), filename, file.content_type)
                supabase.table("documents").update({"storage_path": storage_path}).eq("id", doc_id).execute()
            except Exception as e:
                logger.error(f"Storage failure for doc {doc_id}: {e}")
                supabase.table("documents").delete().eq("id", doc_id).execute()
                supabase.table("ingestion_jobs").delete().eq("id", job_id).execute()
                raise HTTPException(status_code=500, detail="Failed to upload file to storage.")

        # 7. Enqueue Celery Task
        try:
            ingest_document.delay(doc_id, job_id, content_format, url)
        except Exception as e:
            logger.error(f"Queue failure for job {job_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to queue background ingestion job.")

        return IngestionResponse(
            document_id=uuid.UUID(doc_id),
            job_id=uuid.UUID(job_id),
            status=JobStatus.QUEUED
        )