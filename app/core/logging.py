import sys
import logging
import contextvars
import re
from typing import Any, Dict, Optional

from loguru import logger
from .config import get_settings

# Context variables for tracking request and job IDs across async flows
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
task_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("task_id", default=None)

# Sensitive key patterns to scrub from log payloads
SENSITIVE_PATTERNS = re.compile(r"(?i)(password|secret|api_key|apikey|token|authorization|content|text|payload)")
SCRUBBED_VALUE = "***SCRUBBED***"


class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging messages and routes them to Loguru.
    Required for capturing logs from FastAPI, Uvicorn, and Celery internals.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def scrub_sensitive_data(record: Dict[str, Any]) -> None:
    """
    Patches the log record to scrub sensitive data from extra fields.
    Prevents API keys, secrets, and document contents from being logged.
    """
    extra = record.get("extra", {})
    if not isinstance(extra, dict):
        return
        
    for key in list(extra.keys()):
        if SENSITIVE_PATTERNS.search(key):
            extra[key] = SCRUBBED_VALUE


def context_patcher(record: Dict[str, Any]) -> None:
    """
    Attaches contextual information (request_id, task_id, service_name)
    to every log record and applies sensitive data scrubbing.
    """
    record["extra"]["request_id"] = request_id_ctx.get() or "-"
    record["extra"]["task_id"] = task_id_ctx.get() or "-"
    
    # Ensure service name is always present
    if "service_name" not in record["extra"]:
        record["extra"]["service_name"] = "rag-service"
        
    scrub_sensitive_data(record)


def setup_logging() -> None:
    """
    Configures structured JSON logging for production, or pretty logging for development.
    Safely captures FastAPI, Celery, and Uvicorn logs.
    """
    settings = get_settings()
    
    # Remove default Loguru sink
    logger.remove()
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Determine if we should use JSON (production) or pretty (development) format
    is_prod = settings.APP_ENV.lower() == "production"
    log_format = (
        '{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", "level": "{level}", "service": "{extra[service_name]}", "request_id": "{extra[request_id]}", "task_id": "{extra[task_id]}", "message": "{message}"}'
        if is_prod else
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[service_name]}</cyan> | <blue>req:{extra[request_id]}</blue> | <magenta>task:{extra[task_id]}</magenta> | <level>{message}</level>"
    )
    
    # Add sink with context patching
    logger.configure(patcher=context_patcher)
    
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=log_format,
        serialize=is_prod,  # Outputs structured JSON if True
        backtrace=False,    # Do not expose variable values in tracebacks (security)
        diagnose=False,     # Do not expose variable values in exceptions (security)
        enqueue=True,       # Thread-safe logging, non-blocking for FastAPI/Celery
    )

    logger.info("Logging system initialized.", service_name=settings.APP_NAME, env=settings.APP_ENV)    