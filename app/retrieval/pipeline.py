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
        top_k: int, 
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
        
        try:
            # 1. Single Query Encoding (wrapped in thread to avoid blocking)
            t0 = time.time()
            embeddings: QueryEmbeddingResult = await asyncio.to_thread(self.encoder.encode, query)
            emb_lat = (time.time() - t0) * 1000
            
            # 2. Concurrent Retrieval (sync Qdrant client isolated in threads)
            t1 = time.time()
            dense_task = asyncio.to_thread(
                self.dense.retrieve, embeddings.dense_vector, course_ids, settings.DEFAULT_TOP_K, filters
            )
            sparse_task = asyncio.to_thread(
                self.sparse.retrieve, embeddings.sparse_vector, course_ids, settings.DEFAULT_TOP_K, filters
            )
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
            ret_lat = (time.time() - t1) * 1000
            
            if not dense_results and not sparse_results:
                logger.info(f"[{q_hash}] No retrieval candidates found.")
                # Return empty response with zero metrics
                metrics = RetrievalMetrics(
                    query_hash=q_hash, embedding_latency_ms=emb_lat, dense_latency_ms=ret_lat/2,
                    sparse_latency_ms=ret_lat/2, fusion_latency_ms=0, reranking_latency_ms=0,
                    total_latency_ms=(time.time()-start_time)*1000, dense_candidates=0,
                    sparse_candidates=0, fused_candidates=0, final_candidates=0
                )
                return RetrievalResponse(results=[], total_candidates=0, final_count=0, timings=metrics)
                
            # 3. Fusion
            t2 = time.time()
            fused = self.fusion.fuse(dense_results, sparse_results)
            fusion_lat = (time.time() - t2) * 1000
            
            # 4. Reranking
            t3 = time.time()
            rerank_candidates = fused[:settings.RERANK_TOP_K]
            reranked = await asyncio.to_thread(self.reranker.rerank, query, rerank_candidates, settings.RERANK_TOP_K)
            rerank_lat = (time.time() - t3) * 1000
            
            # 5. Final slice
            final_results = reranked[:top_k]
            total_lat = (time.time() - start_time) * 1000
            
            metrics = RetrievalMetrics(
                query_hash=q_hash,
                embedding_latency_ms=emb_lat,
                dense_latency_ms=ret_lat / 2,  # approx concurrent split
                sparse_latency_ms=ret_lat / 2,
                fusion_latency_ms=fusion_lat,
                reranking_latency_ms=rerank_lat,
                total_latency_ms=total_lat,
                dense_candidates=len(dense_results),
                sparse_candidates=len(sparse_results),
                fused_candidates=len(fused),
                final_candidates=len(final_results)
            )
            logger.info(f"Retrieval metrics: {metrics.model_dump_json()}")
            
            return RetrievalResponse(
                results=final_results,
                total_candidates=len(fused),
                final_count=len(final_results),
                timings=metrics
            )
            
        except Exception as e:
            logger.error(f"[{q_hash}] Retrieval pipeline failed: {e}")
            if not isinstance(e, (RetrievalException, ValidationException)):
                raise RetrievalException("Pipeline execution failed.", detail=str(e))
            raise