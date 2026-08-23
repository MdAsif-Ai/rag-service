# dense.py
from typing import List
from app.vectorstore.qdrant import QdrantRepository
from .models import RetrievalCandidate, RetrievalFilters
from .interfaces import IRetriever

class DenseRetriever(IRetriever):
    def __init__(self, qdrant_repo: QdrantRepository):
        self.qdrant_repo = qdrant_repo

    def retrieve(self, vector: List[float], course_ids: List[str], top_k: int, filters: RetrievalFilters) -> List[RetrievalCandidate]:
        raw_results = self.qdrant_repo.search_dense(
            query_vector=vector,
            course_ids=course_ids,
            top_k=top_k,
            filters=filters.model_dump(exclude_none=True)
        )
        # Map raw results to RetrievalCandidate (omitted mapping logic for brevity, assume it happens in qdrant_repo or here)
        return [RetrievalCandidate(**r, dense_score=r.get("score")) for r in raw_results]
