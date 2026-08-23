from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    course_ids: List[str] = Field(..., min_length=1, description="Restrict search to these course IDs")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Override default final top_k results")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional Qdrant payload filters (e.g., {'lesson_id': 'intro'})"
    )


class RetrievedChunk(BaseModel):
    content: str
    document_id: UUID
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int
    score: float = Field(..., description="Final retrieval/reranking score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete payload metadata from Qdrant")


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]