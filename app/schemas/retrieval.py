from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, model_validator

class RetrievalFilters(BaseModel):
    """Validated internal filter model to prevent arbitrary Qdrant filters."""
    document_ids: Optional[List[str]] = None
    chapters: Optional[List[str]] = None
    sections: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    pages: Optional[List[int]] = None

class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The search query")
    course_ids: List[str] = Field(..., min_length=1, description="Restrict search to these course IDs")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of final results to return")
    filters: Optional[RetrievalFilters] = Field(default=None, description="Optional metadata filters")

class RetrievedChunk(BaseModel):
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

class APIRetrievalResponse(BaseModel):
    query: str
    total_candidates: int
    final_count: int
    results: List[RetrievedChunk]