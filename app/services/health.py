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
        # Wrap sync call in thread and enforce a 2-second timeout
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
        # Lightweight query to verify DB connectivity
        def _query():
            return sb.table("documents").select("id").limit(1).execute()
        await asyncio.wait_for(asyncio.to_thread(_query), timeout=3.0)
        return True
    except Exception as e:
        logger.error(f"Health check: Supabase failed: {e}")
        return False

async def get_system_health() -> Dict[str, Any]:
    """Runs all dependency checks concurrently."""
    qdrant_ok, redis_ok, supabase_ok = await asyncio.gather(
        check_qdrant(),
        check_redis(),
        check_supabase()
    )
    
    is_ready = all([qdrant_ok, redis_ok, supabase_ok])
    
    return {
        "status": "ready" if is_ready else "unavailable",
        "dependencies": {
            "qdrant": qdrant_ok,
            "redis": redis_ok,
            "database": supabase_ok
        }
    }