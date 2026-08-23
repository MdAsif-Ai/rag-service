from uuid import UUID
from fastapi import APIRouter, Depends, status

from app.core.security import verify_api_key
from app.schemas.jobs import JobMetadata
from app.services.jobs import JobService

router = APIRouter()

@router.get(
    "/jobs/{job_id}", 
    response_model=JobMetadata,
    dependencies=[Depends(verify_api_key)]
)
async def get_job_status(job_id: UUID):
    """
    Retrieves the current status and metadata of an ingestion job.
    """
    return await JobService.get_job_status(job_id)