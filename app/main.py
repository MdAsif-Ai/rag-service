from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uuid

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.core.state import app_state
from app.core.config import get_settings
from app.core.logging import setup_logging, request_id_ctx
from app.core.exceptions import RAGServiceException

from app.api.v1.health import router as health_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.retrieval import router as retrieval_router
from app.api.v1.documents import router as documents_router

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} in {settings.APP_ENV} mode...")
    
    try:
        from app.retrieval.pipeline import RetrievalPipeline
        from app.retrieval.query_encoder import BGEQueryEncoder
        from app.retrieval.dense import DenseRetriever
        from app.retrieval.sparse import SparseRetriever
        from app.retrieval.fusion import RRFFusion
        from app.retrieval.reranker import get_reranker_service
        from app.vectorstore.qdrant import get_qdrant_repository
        from app.embeddings.bge_m3 import get_embedding_service
        
        qdrant_repo = get_qdrant_repository()
        if not qdrant_repo.collection_exists():
            qdrant_repo.create_collection()
        encoder = BGEQueryEncoder(get_embedding_service())
        
        app_state.pipeline = RetrievalPipeline(
            query_encoder=encoder,
            dense_retriever=DenseRetriever(qdrant_repo),
            sparse_retriever=SparseRetriever(qdrant_repo),
            fusion_service=RRFFusion(k=settings.FUSION_RRF_K),
            reranker=get_reranker_service()
        )
        logger.info("Retrieval pipeline initialized.")
    except Exception as e:
        logger.warning(f"Pipeline initialization deferred or failed: {e}. Endpoints may be unavailable.")
        app_state.pipeline = None

    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_ctx.set(req_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        request_id_ctx.reset(token)
        return response

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
    
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"]
    )
    
    @app.exception_handler(RAGServiceException)
    async def rag_exception_handler(request: Request, exc: RAGServiceException):
        logger.error(f"Application error: {exc.detail}")
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
        
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "An unexpected internal server error occurred."})

    app.include_router(health_router, tags=["Health"])
    api_prefix = settings.API_PREFIX
    app.include_router(ingestion_router, prefix=api_prefix, tags=["Ingestion"])
    app.include_router(jobs_router, prefix=api_prefix, tags=["Jobs"])
    app.include_router(retrieval_router, prefix=api_prefix, tags=["Retrieval"])
    app.include_router(documents_router, prefix=api_prefix, tags=["Documents"])
    
    return app

app = create_app()