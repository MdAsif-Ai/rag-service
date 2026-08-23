import hashlib
import uuid
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from loguru import logger

from app.core.config import get_settings
from app.db.supabase import get_supabase_client
from app.storage.supabase import storage_service
from app.schemas.ingestion import IngestionResponse
from app.schemas.jobs import JobStatus
from app.jobs.ingestion import ingest_document

class IngestionService:
    @staticmethod
    async def process_upload(
        file: UploadFile,
        course_id: str,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> IngestionResponse:
        settings = get_settings()
        supabase = get_supabase_client()
        
        # 1. Validate File Extension
        filename = file.filename or "unknown"
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in settings.SUPPORTED_FILE_TYPES:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Unsupported file type: .{ext}")

        # 2. Read file and check size (MUST HAPPEN BEFORE DB CALLS)
        file_bytes = await file.read()
        if len(file_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File size exceeds maximum limit")

        # 3. Calculate Checksum
        checksum = hashlib.sha256(file_bytes).hexdigest()

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
                "file_type": ext,
                "file_size": len(file_bytes),
                "storage_path": "", 
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

        # 6. Upload to Storage
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
            ingest_document.delay(doc_id, job_id)
        except Exception as e:
            logger.error(f"Queue failure for job {job_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to queue background ingestion job.")

        return IngestionResponse(
            document_id=uuid.UUID(doc_id),
            job_id=uuid.UUID(job_id),
            status=JobStatus.QUEUED
        )