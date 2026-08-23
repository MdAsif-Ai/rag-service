import asyncio
import time
import hashlib
from typing import List, Optional
from loguru import logger

from app.core.config import get_settings
from app.core.exceptions import RetrievalException, ValidationException
from app.retrieval.models import (
    RetrievalCandidate, 
    RetrievalFilters, 
    RetrievalMetrics,
    QueryEmbeddingResult
)
from app.retrieval.interfaces import (
    IQueryEncoder, 
    IRetriever, 
    IFusion, 
    IReranker
)
from pydantic import BaseModel

class RetrievalResponse(BaseModel):
    results: List[RetrievalCandidate]
    total_candidates: int
    final_count: int
    timings: RetrievalMetrics

class RetrievalPipeline:
    """
    Orchestrates the hybrid retrieval pipeline.
    Executes query encoding once, retrieves dense/sparse concurrently, 
    fuses, reranks, and returns the final context.
    """
    
    def __init__(
        self, 
        query_encoder: IQueryEncoder, 
        dense_retriever: IRetriever, 
        sparse_retriever: IRetriever,
        fusion_service: IFusion,
        reranker: IReranker
    ):
        self.encoder = query_encoder
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.fusion = fusion_service
        self.reranker = reranker

    async def retrieve(
        self, 
        query: str, 
        course_ids: List[str], 
        top_k: Optional[int] = None,
        filters: Optional[RetrievalFilters] = None
    ) -> RetrievalResponse:
        """
        Executes the retrieval pipeline asynchronously.
        Uses asyncio.to_thread to prevent blocking FastAPI's event loop during
        synchronous Qdrant and model operations.
        """
        if not query or not query.strip():
            raise ValidationException("Query cannot be empty.")
        if not course_ids:
            raise ValidationException("course_ids must be provided.")

        start_time = time.time()
        settings = get_settings()
        filters = filters or RetrievalFilters()
        q_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
        
        metrics = RetrievalMetrics(query_hash=q_hash)
        
        try:
            # 1. Single Query Encoding (wrapped in thread to avoid blocking)
            t0 = time.time()
            embeddings: QueryEmbeddingResult = await asyncio.to_thread(self.encoder.encode, query)
            metrics.embedding_latency_ms = (time.time() - t0) * 1000
            
            # 2. Concurrent Retrieval (sync Qdrant client isolated in threads)
            t1 = time.time()
            dense_task = asyncio.to_thread(
                self.dense.retrieve, 
                embeddings.dense_vector, 
                course_ids=course_ids, 
                top_k=settings.DENSE_TOP_K, 
                filters=filters
            )
            sparse_task = asyncio.to_thread(
                self.sparse.retrieve, 
                embeddings.sparse_vector, 
                course_ids=course_ids, 
                top_k=settings.SPARSE_TOP_K, 
                filters=filters
            )
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
            
            ret_lat = (time.time() - t1) * 1000
            metrics.dense_latency_ms = ret_lat
            metrics.sparse_latency_ms = ret_lat
            
            metrics.dense_candidates = len(dense_results)
            metrics.sparse_candidates = len(sparse_results)
            
            if not dense_results and not sparse_results:
                logger.info(f"[{q_hash}] No retrieval candidates found.")
                metrics.total_latency_ms = (time.time() - start_time) * 1000
                return RetrievalResponse(results=[], total_candidates=0, final_count=0, timings=metrics)
                
            # 3. Fusion
            t2 = time.time()
            fused = self.fusion.fuse(dense_results, sparse_results, top_k=settings.FUSION_TOP_K)
            metrics.fusion_latency_ms = (time.time() - t2) * 1000
            metrics.fused_candidates = len(fused)
            
            # 4. Reranking
            t3 = time.time()
            final_k = top_k if top_k is not None else settings.FINAL_TOP_K
            
            # Pass final_k to reranker as a hint
            reranked = await asyncio.to_thread(self.reranker.rerank, query, fused, final_k)
            
            # Enforce final top_k constraint at the pipeline level to guarantee API contract
            final_results = reranked[:final_k]
            
            metrics.reranking_latency_ms = (time.time() - t3) * 1000
            metrics.final_candidates = len(final_results)
            
            metrics.total_latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Retrieval metrics: {metrics.model_dump_json()}")
            
            return RetrievalResponse(
                results=final_results,
                total_candidates=metrics.fused_candidates,
                final_count=metrics.final_candidates,
                timings=metrics
            )
            
        except Exception as e:
            logger.error(f"[{q_hash}] Retrieval pipeline failed: {e}")
            if not isinstance(e, (RetrievalException, ValidationException)):
                raise RetrievalException("Pipeline execution failed.", detail=str(e))
            raise