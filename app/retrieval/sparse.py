from typing import Dict, List, Optional, Any
from app.vectorstore.qdrant import QdrantRepository
from app.retrieval.models import RetrievalCandidate, RetrievalFilters
from app.retrieval.interfaces import IRetriever

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
        candidates = []
        for r in raw_results:
            candidates.append(RetrievalCandidate(
                chunk_id=r.get("chunk_id", ""),
                document_id=r.get("document_id", ""),
                course_id=r.get("course_id", ""),
                filename=r.get("filename", ""),
                content=r.get("content", ""),
                page=r.get("page"),
                chapter=r.get("chapter"),
                section=r.get("section"),
                chunk_index=r.get("chunk_index", 0),
                sparse_score=r.get("score"),
                metadata=r.get("metadata", {})
            ))
        return candidates