# sparse.py
from typing import Dict
class SparseRetriever(IRetriever):
    def __init__(self, qdrant_repo: QdrantRepository):
        self.qdrant_repo = qdrant_repo

    def retrieve(self, vector: Dict[int, float], course_ids: List[str], top_k: int, filters: RetrievalFilters) -> List[RetrievalCandidate]:
        raw_results = self.qdrant_repo.search_sparse(
            sparse_vector=vector,
            course_ids=course_ids,
            top_k=top_k,
            filters=filters.model_dump(exclude_none=True)
        )
        return [RetrievalCandidate(**r, sparse_score=r.get("score")) for r in raw_results]