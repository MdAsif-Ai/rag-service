from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uuid

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.core.config import get_settings
from app.core.logging import setup_logging, request_id_ctx
from app.core.exceptions import RAGServiceException

# Routers
from app.api.v1.health import router as health_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.retrieval import router as retrieval_router
from app.api.v1.documents import router as documents_router


# --- Application State ---
# Used for dependency injection in routes (e.g., RetrievalPipeline)
class AppState:
    pass

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages application startup and shutdown events.
    """
    settings = get_settings()
    
    # 1. Initialize logging
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} in {settings.APP_ENV} mode...")
    
    # NOTE: ML Models (BGE-M3, Reranker) are NOT loaded here.
    # They are loaded lazily by the Celery workers or isolated retrieval services 
    # to prevent OOM errors in the FastAPI event loop.
    
    # 2. Initialize app state (e.g., Retrieval Pipeline if run in-process)
    # In a scaled architecture, this might just be an HTTP client to another service.
    try:
        # Lazy import to prevent loading PyTorch at module level
        from app.retrieval.pipeline import RetrievalPipeline
        from app.retrieval.query_encoder import BGEQueryEncoder
        from app.retrieval.dense import DenseRetriever
        from app.retrieval.sparse import SparseRetriever
        from app.retrieval.fusion import RRFFusion
        from app.retrieval.reranker import BGEReranker
        from app.vectorstore.qdrant import get_qdrant_repository
        from app.embeddings.bge_m3 import get_embedding_service
        
        # Wire up the pipeline for this API process
        # (Model weights will load on first request if not cached by worker)
        qdrant_repo = get_qdrant_repository()
        encoder = BGEQueryEncoder(get_embedding_service())
        
        app_state.pipeline = RetrievalPipeline(
            query_encoder=encoder,
            dense_retriever=DenseRetriever(qdrant_repo),
            sparse_retriever=SparseRetriever(qdrant_repo),
            fusion_service=RRFFusion(k=settings.FUSION_RRF_K),
            reranker=BGEReranker() # Loads on first use
        )
        logger.info("Retrieval pipeline initialized.")
    except Exception as e:
        logger.warning(f"Pipeline initialization deferred or failed: {e}. Endpoints may be unavailable.")
        app_state.pipeline = None

    yield
    
    # Shutdown logic
    logger.info(f"Shutting down {settings.APP_NAME}...")


# --- Middleware ---
class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Extracts or generates a Request ID and stores it in the contextvars
    so it can be attached to all structured logs for this request.
    """
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Set the context variable for loguru
        token = request_id_ctx.set(req_id)
        
        response = await call_next(request)
        
        # Expose the Request ID to the client
        response.headers["X-Request-ID"] = req_id
        
        # Reset context variable
        request_id_ctx.reset(token)
        
        return response


# --- Application Factory ---
def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        description="Production-grade Retrieval-Augmented Generation (RAG) service for the LMS.",
        version="1.0.0",
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
        lifespan=lifespan
    )
    
    # 1. Add Middleware
    app.add_middleware(RequestIDMiddleware)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production via Nginx/LMS domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"]
    )
    
    # 2. Register Exception Handlers
    @app.exception_handler(RAGServiceException)
    async def rag_exception_handler(request: Request, exc: RAGServiceException):
        # Log the internal detail, return the safe message
        logger.error(f"Application error: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )
        
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # Catch-all to ensure NO stack traces are exposed
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected internal server error occurred."},
        )

    # 3. Register Routers
    # Health checks at root level for load balancers
    app.include_router(health_router, tags=["Health"])
    
    # API v1 routes
    api_prefix = settings.API_PREFIX
    app.include_router(ingestion_router, prefix=api_prefix, tags=["Ingestion"])
    app.include_router(jobs_router, prefix=api_prefix, tags=["Jobs"])
    app.include_router(retrieval_router, prefix=api_prefix, tags=["Retrieval"])
    app.include_router(documents_router, prefix=api_prefix, tags=["Documents"])
    
    return app

# Instantiate the app for Uvicorn
app = create_app()