import uuid
from typing import Optional
from fastapi import HTTPException, status
from loguru import logger

from app.db.supabase import get_supabase_client
from app.schemas.documents import DocumentDetail
from app.schemas.jobs import JobMetadata, JobStatus, JobStage
from app.schemas.ingestion import IngestionResponse
from app.jobs.ingestion import ingest_document

class DocumentService:
    @staticmethod
    async def get_document(document_id: uuid.UUID) -> DocumentDetail:
        supabase = get_supabase_client()
        
        try:
            doc_res = supabase.table("documents").select("*").eq("id", str(document_id)).limit(1).execute()
            if not doc_res.data:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
            
            # Use copy() to avoid mutating the original dictionary
            doc_data = doc_res.data[0].copy()
            doc_data["document_id"] = doc_data.pop("id")
            
            job_res = supabase.table("ingestion_jobs").select("*").eq("document_id", str(document_id)).order("created_at", desc=True).limit(1).execute()
            
            latest_job = None
            if job_res.data:
                job_data = job_res.data[0].copy()
                job_data["job_id"] = job_data.pop("id")
                latest_job = JobMetadata(**job_data)
            
            return DocumentDetail(**doc_data, latest_job=latest_job)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database error fetching document {document_id}: {e}")
            raise HTTPException(status_code=500, detail="An error occurred while retrieving document.")

    @staticmethod
    async def reindex_document(document_id: uuid.UUID) -> IngestionResponse:
        supabase = get_supabase_client()
        
        try:
            doc_res = supabase.table("documents").select("id, course_id, status").eq("id", str(document_id)).limit(1).execute()
            if not doc_res.data:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
                
            doc_data = doc_res.data[0]
            
            if doc_data["status"] == "PROCESSING":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, 
                    detail="Document is already being processed. Please wait for the current job to finish."
                )
            
            job_id = str(uuid.uuid4())
            supabase.table("ingestion_jobs").insert({
                "id": job_id,
                "document_id": str(document_id),
                "status": JobStatus.QUEUED,
                "stage": JobStage.UPLOADED
            }).execute()
            
            supabase.table("documents").update({
                "status": "PROCESSING"
            }).eq("id", str(document_id)).execute()
            
            ingest_document.delay(str(document_id), job_id)
            
            return IngestionResponse(
                document_id=document_id,
                job_id=uuid.UUID(job_id),
                status=JobStatus.QUEUED
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to start reindex for {document_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to queue reindex job.")