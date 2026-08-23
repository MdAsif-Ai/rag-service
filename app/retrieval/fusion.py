from typing import Dict, List, Optional
from loguru import logger

from app.core.exceptions import ValidationException
from app.retrieval.models import RetrievalCandidate


class RRFFusion:
    """
    Reciprocal Rank Fusion (RRF) service.
    Combines dense and sparse retrieval results by computing a unified score
    based on their reciprocal ranks, rather than averaging raw scores.
    """

    def __init__(self, k: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self, 
        dense_results: List[RetrievalCandidate], 
        sparse_results: List[RetrievalCandidate], 
        top_k: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """
        Fuses two lists of retrieval candidates using RRF.
        
        Args:
            dense_results: Candidates from dense vector search.
            sparse_results: Candidates from sparse lexical search.
            top_k: Optional limit on the number of returned candidates.
            
        Returns:
            A single deduplicated, deterministically sorted list of RetrievalCandidate objects.
        """
        if top_k is not None and top_k <= 0:
            raise ValidationException(f"top_k must be greater than 0, got {top_k}")

        fused_map: Dict[str, RetrievalCandidate] = {}

        def process_results(results: List[RetrievalCandidate], weight: float, score_field: str):
            for rank, candidate in enumerate(results):
                if not candidate.chunk_id:
                    raise ValidationException("Candidate missing chunk_id during fusion")
                
                cid = candidate.chunk_id
                rank_score = weight * (1.0 / (self.k + rank + 1))
                
                if cid not in fused_map:
                    # Deep copy to avoid mutating original objects passed from retrievers
                    fused_map[cid] = candidate.model_copy(deep=True)
                    fused_map[cid].fusion_score = 0.0
                
                # Update specific score and accumulate fusion score
                setattr(fused_map[cid], score_field, getattr(candidate, score_field))
                fused_map[cid].fusion_score = (fused_map[cid].fusion_score or 0.0) + rank_score

        process_results(dense_results, self.dense_weight, "dense_score")
        process_results(sparse_results, self.sparse_weight, "sparse_score")

        fused_list = list(fused_map.values())
        
        # Deterministic sorting: fusion_score DESC, then chunk_id ASC
        fused_list.sort(key=lambda x: (-x.fusion_score, x.chunk_id))
        
        if top_k is not None:
            fused_list = fused_list[:top_k]
            
        logger.debug(f"Fusion complete: {len(dense_results)} dense + {len(sparse_results)} sparse -> {len(fused_list)} fused.")
        return fused_list