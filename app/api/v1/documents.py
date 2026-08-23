from uuid import UUID
from fastapi import APIRouter, Depends, status

from app.core.security import verify_api_key
from app.schemas.documents import DocumentDetail
from app.schemas.ingestion import IngestionResponse
from app.services.documents import DocumentService

router = APIRouter()

@router.get(
    "/documents/{document_id}", 
    response_model=DocumentDetail,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)]
)
async def get_document(document_id: UUID):
    """
    Retrieves detailed metadata and latest ingestion status for a document.
    """
    return await DocumentService.get_document(document_id)

@router.post(
    "/documents/{document_id}/reindex", 
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)]
)
async def reindex_document(document_id: UUID):
    """
    Triggers a re-ingestion of an existing document without requiring a new upload.
    Uses idempotent upserts (deletes old chunks, inserts new ones).
    """
    return await DocumentService.reindex_document(document_id)