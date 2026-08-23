from celery import Celery
from loguru import logger
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "rag_service",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.jobs.ingestion"]
)

# Production-safe configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Reliability: Prevent zombie tasks and ensure tasks are re-queued if worker crashes
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Performance: Prefetch 1 task at a time per worker process (prevents long queue starvation)
    worker_prefetch_multiplier=1,
    
    # Timeouts (Seconds) - prevent stuck workers
    task_time_limit=1800,      # 30 minutes hard limit
    task_soft_time_limit=1500, # 25 minutes soft limit
    
    # Retries
    task_default_max_retries=3,
    task_default_retry_delay=60, # 1 minute exponential backoff base
)

@celery_app.task
def heartbeat():
    """Simple task to verify worker connectivity."""
    logger.info("Celery worker heartbeat received.")
    return True