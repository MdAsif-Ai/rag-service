import asyncio
from typing import Dict, Any
import redis.asyncio as aioredis
from loguru import logger

from app.core.config import get_settings
from app.vectorstore.qdrant import get_qdrant_repository
from app.db.supabase import get_supabase_client

async def check_qdrant() -> bool:
    try:
        repo = get_qdrant_repository()
        return await asyncio.wait_for(asyncio.to_thread(repo.health_check), timeout=2.0)
    except Exception as e:
        logger.error(f"Health check: Qdrant failed: {e}")
        return False

async def check_redis() -> bool:
    try:
        settings = get_settings()
        r = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2.0, socket_timeout=2.0)
        pong = await asyncio.wait_for(r.ping(), timeout=2.0)
        await r.aclose()
        return pong
    except Exception as e:
        logger.error(f"Health check: Redis failed: {e}")
        return False

async def check_supabase() -> bool:
    try:
        sb = get_supabase_client()
        def _query():
            return sb.table("documents").select("id").limit(1).execute()
        await asyncio.wait_for(asyncio.to_thread(_query), timeout=3.0)
        return True
    except Exception as e:
        logger.error(f"Health check: Supabase failed: {e}")
        return False

async def get_system_health() -> Dict[str, Any]:
    """Runs all dependency checks concurrently."""
    # return_exceptions=True ensures that if a check raises an exception,
    # it is returned as a result object rather than crashing the gather.
    results = await asyncio.gather(
        check_qdrant(),
        check_redis(),
        check_supabase(),
        return_exceptions=True
    )
    
    # Safely evaluate results: if it's an exception or not True, it's False
    qdrant_ok = results[0] if (not isinstance(results[0], Exception) and results[0] is True) else False
    redis_ok = results[1] if (not isinstance(results[1], Exception) and results[1] is True) else False
    supabase_ok = results[2] if (not isinstance(results[2], Exception) and results[2] is True) else False
    
    is_ready = all([qdrant_ok, redis_ok, supabase_ok])
    
    return {
        "status": "ready" if is_ready else "unavailable",
        "dependencies": {
            "qdrant": qdrant_ok,
            "redis": redis_ok,
            "database": supabase_ok
        }
    }