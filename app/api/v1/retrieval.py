from fastapi import APIRouter, Depends, status, HTTPException
from loguru import logger

from app.core.security import verify_api_key
from app.core.exceptions import RetrievalException, ValidationException
from app.schemas.retrieval import RetrievalRequest, APIRetrievalResponse, RetrievedChunk
from app.retrieval.pipeline import RetrievalPipeline, RetrievalResponse

router = APIRouter()

# Dependency to provide the pipeline instance
async def get_pipeline() -> RetrievalPipeline:
    # In a real app, this would yield a cached/singleton pipeline
    from app.main import app_state
    return app_state.pipeline

@router.post(
    "/retrieve", 
    response_model=APIRetrievalResponse,
    dependencies=[Depends(verify_api_key)]
)
async def retrieve_context(
    request: RetrievalRequest,
    pipeline: RetrievalPipeline = Depends(get_pipeline)
):
    """
    Retrieves knowledge context from the vector database.
    This endpoint DOES NOT generate LLM answers. It returns chunks for the LMS to send to an LLM.
    """
    logger.info(f"Retrieval request for courses: {request.course_ids}, top_k: {request.top_k}")
    
    try:
        # Execute the retrieval pipeline
        pipeline_response: RetrievalResponse = await pipeline.retrieve(
            query=request.query,
            course_ids=request.course_ids,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Format the response for the API consumer
        return APIRetrievalResponse(
            query=request.query,
            total_candidates=pipeline_response.total_candidates,
            final_count=pipeline_response.final_count,
            results=pipeline_response.results
        )
        
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.message)
    except RetrievalException as e:
        logger.error(f"Retrieval pipeline failed: {e.detail}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Retrieval service temporarily unavailable.")
    except Exception as e:
        logger.error(f"Unexpected error during retrieval: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")