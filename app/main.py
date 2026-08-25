from fastapi import APIRouter, Depends, status, HTTPException
from loguru import logger
from typing import Optional

from app.core.security import verify_api_key
from app.core.exceptions import RetrievalException, ValidationException
from app.schemas.retrieval import RetrievalRequest, APIRetrievalResponse, RetrievedChunk
from app.retrieval.pipeline import RetrievalPipeline, RetrievalResponse
from app.retrieval.models import RetrievalFilters
from app.core.state import app_state  # <--- MUST IMPORT FROM core.state

router = APIRouter()

async def get_pipeline() -> RetrievalPipeline:
    if not app_state.pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Retrieval pipeline is not initialized.")
    return app_state.pipeline

@router.post(
    "/retrieve", 
    response_model=APIRetrievalResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)]
)
async def retrieve_context(
    request: RetrievalRequest,
    pipeline: RetrievalPipeline = Depends(get_pipeline)
):
    logger.info(f"Retrieval request for courses: {request.course_ids}, top_k: {request.top_k}")
    
    try:
        internal_filters: Optional[RetrievalFilters] = None
        if request.filters:
            internal_filters = RetrievalFilters(**request.filters.model_dump())
            
        pipeline_response: RetrievalResponse = await pipeline.retrieve(
            query=request.query,
            course_ids=request.course_ids,
            top_k=request.top_k,
            filters=internal_filters
        )
        
        results = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                course_id=c.course_id,
                filename=c.filename,
                content=c.content,
                page=c.page,
                chapter=c.chapter,
                section=c.section,
                chunk_index=c.chunk_index,
                dense_score=c.dense_score,
                sparse_score=c.sparse_score,
                fusion_score=c.fusion_score,
                rerank_score=c.rerank_score,
                metadata=c.metadata
            ) for c in pipeline_response.results
        ]
        
        return APIRetrievalResponse(
            query=request.query,
            total_candidates=pipeline_response.total_candidates,
            final_count=pipeline_response.final_count,
            results=results
        )
        
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.message)
    except RetrievalException as e:
        logger.error(f"Retrieval pipeline failed: {e.detail}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Retrieval service temporarily unavailable.")
    except Exception as e:
        logger.error(f"Unexpected error during retrieval: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")