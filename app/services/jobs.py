from uuid import UUID
from fastapi import HTTPException, status
from loguru import logger

from app.db.supabase import get_supabase_client
from app.schemas.jobs import JobMetadata

class JobService:
    @staticmethod
    async def get_job_status(job_id: UUID) -> JobMetadata:
        supabase = get_supabase_client()
        
        try:
            res = supabase.table("ingestion_jobs").select("*").eq("id", str(job_id)).limit(1).execute()
            
            if not res.data:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
                
            job_data = res.data[0]
            # Explicit mapping for public contract
            job_data["job_id"] = job_data.pop("id")
            
            return JobMetadata(**job_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database error fetching job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="An error occurred while retrieving job status."
            )