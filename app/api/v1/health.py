from fastapi import APIRouter, Depends, status, Response
from app.core.security import verify_health_api_key
from app.services.health import get_system_health

router = APIRouter()

@router.get(
    "/health",
    dependencies=[Depends(verify_health_api_key)]
)
async def health():
    """
    Liveness probe. Indicates the FastAPI process is alive and routing requests.
    Does NOT check external dependencies.
    """
    return {"status": "ok"}

@router.get(
    "/ready",
    dependencies=[Depends(verify_health_api_key)]
)
async def readiness(response: Response):
    """
    Readiness probe. Indicates whether the service can actually process requests.
    Checks Qdrant, Redis, and Supabase connectivity.
    """
    health_status = await get_system_health()
    
    if health_status["status"] != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
    return health_status