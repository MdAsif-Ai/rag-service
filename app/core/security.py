import secrets
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from loguru import logger

from .config import Settings, get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key_header: Optional[str] = Depends(API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> bool:
    """
    FastAPI dependency to protect internal endpoints.
    Validates the incoming X-API-Key header against the configured RAG_SERVICE_API_KEY.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide it in the 'X-API-Key' header.",
        )

    configured_key = settings.RAG_SERVICE_API_KEY.get_secret_value()

    if not secrets.compare_digest(api_key_header, configured_key):
        logger.warning("Invalid API Key attempt received.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key.",
        )

    return True


async def verify_health_api_key(
    api_key_header: Optional[str] = Depends(API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> bool:
    """
    FastAPI dependency specifically for health endpoints.
    If HEALTH_CHECK_API_KEY is configured, it requires that key.
    If it is NOT configured (None/empty), the endpoint is open for infrastructure load balancers
    and Docker healthchecks to access without authentication.
    """
    # Safely check if the key is None or empty
    health_key_secret = settings.HEALTH_CHECK_API_KEY
    if not health_key_secret:
        return True

    health_key = health_key_secret.get_secret_value()
    if not health_key:
        return True

    # If a health key IS configured, enforce it
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Health API Key.",
        )

    if not secrets.compare_digest(api_key_header, health_key):
        logger.warning("Invalid Health API Key attempt received.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Health API Key.",
        )

    return True