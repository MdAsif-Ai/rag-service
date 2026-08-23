import functools
import threading
from typing import List, Optional
from loguru import logger

from app.core.config import Settings, get_settings
from app.core.exceptions import RetrievalException, ValidationException
from app.retrieval.models import RetrievalCandidate


class BGEReranker:
    """
    Production reranking service using BGE-reranker-v2-m3.
    The model is loaded once per worker process to maximize performance.
    """
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(BGEReranker, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings: Settings):
        if self._initialized:
            return
            
        with BGEReranker._lock:
            if self._initialized:
                return
                
            self.settings = settings
            self.model = None
            self._initialize_model()
            self._initialized = True

    def _initialize_model(self) -> None:
        """Lazy initialization of the BGE Reranker model."""
        try:
            from FlagEmbedding import FlagReranker
            
            model_name = self.settings.RERANKER_MODEL
            device = self.settings.RERANKER_DEVICE
            
            # use_fp16 improves performance significantly on GPU. Fallback to False on CPU.
            use_fp16 = (device == "cuda")
            
            logger.info(f"Initializing BGE Reranker '{model_name}' on device '{device}'...")
            
            self.model = FlagReranker(
                model_name,
                use_fp16=use_fp16,
                device=device
            )
            
            logger.info("BGE Reranker loaded successfully.")
            
        except ImportError:
            raise RetrievalException(
                "FlagEmbedding library not installed. Please install it to use the reranker.",
                detail="Run: pip install FlagEmbedding torch"
            )
        except Exception as e:
            logger.error(f"Failed to load BGE Reranker: {e}")
            
            # CPU Fallback for development if GPU fails
            if device == "cuda":
                logger.warning("GPU load failed. Falling back to CPU.")
                try:
                    from FlagEmbedding import FlagReranker
                    self.model = FlagReranker(model_name, use_fp16=False, device="cpu")
                    self.settings.RERANKER_DEVICE = "cpu" # Update active setting
                    logger.info("BGE Reranker loaded successfully on CPU fallback.")
                except Exception as fallback_e:
                    raise RetrievalException("Failed to load BGE Reranker on both GPU and CPU.", detail=str(fallback_e))
            else:
                raise RetrievalException("Failed to initialize BGE Reranker model.", detail=str(e))

    def rerank(
        self, 
        query: str, 
        candidates: List[RetrievalCandidate], 
        top_k: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """
        Reranks a pool of candidates based on semantic relevance to the query.
        """
        if not query:
            raise ValidationException("Query cannot be empty for reranking.")
            
        if not candidates:
            logger.info("No candidates provided to reranker. Returning empty list.")
            return []

        # 1. Validate candidates and enforce max pool size
        max_pool = self.settings.RERANK_MAX_CANDIDATES
        candidate_pool = candidates[:max_pool]
        
        for c in candidate_pool:
            if not c.content or not c.content.strip():
                raise ValidationException(f"Candidate {c.chunk_id} missing content for reranking.")

        # 2. Prepare pairs for the model
        pairs = [[query, c.content] for c in candidate_pool]
        
        # 3. Execute model inference
        try:
            # FlagReranker handles batching internally via the batch_size parameter
            scores = self.model.compute_score(
                pairs, 
                normalize=True, 
                batch_size=self.settings.RERANK_BATCH_SIZE
            )
            
            # Ensure scores is always a list, even if only 1 pair was passed
            if not isinstance(scores, list):
                scores = [scores]
                
        except Exception as e:
            logger.error(f"Reranker model inference failed: {e}")
            raise RetrievalException("Reranker model inference failed.", detail=str(e))

        # 4. Assign scores and sort deterministically
        for candidate, score in zip(candidate_pool, scores):
            candidate.rerank_score = float(score)
            
        # Sort by rerank_score DESC. Tie-break with chunk_id ASC for determinism.
        candidate_pool.sort(key=lambda x: (-x.rerank_score, x.chunk_id))
        
        # 5. Slice to final top_k
        final_k = top_k if top_k is not None else self.settings.RERANK_FINAL_TOP_K
        return candidate_pool[:final_k]


@functools.lru_cache(maxsize=1)
def get_reranker_service() -> BGEReranker:
    """
    Factory function to get a cached singleton instance of the BGEReranker.
    Ensures the model is loaded exactly once per Uvicorn/Celery worker process.
    """
    settings = get_settings()
    return BGEReranker(settings)