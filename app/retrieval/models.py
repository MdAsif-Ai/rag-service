from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class QueryEmbeddingResult(BaseModel):
    dense_vector: List[float]
    sparse_vector: Dict[int, float]

class RetrievalFilters(BaseModel):
    """Validated internal filter model to prevent arbitrary Qdrant filters."""
    document_ids: Optional[List[str]] = None
    chapters: Optional[List[str]] = None
    sections: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    pages: Optional[List[int]] = None

class RetrievalCandidate(BaseModel):
    chunk_id: str
    document_id: str
    course_id: str
    filename: str
    content: str
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    chunk_index: int
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalMetrics(BaseModel):
    query_hash: str
    embedding_latency_ms: float
    dense_latency_ms: float
    sparse_latency_ms: float
    fusion_latency_ms: float
    reranking_latency_ms: float
    total_latency_ms: float
    dense_candidates: int
    sparse_candidates: int
    fused_candidates: int
    final_candidates: int