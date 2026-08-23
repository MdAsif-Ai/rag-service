from abc import ABC, abstractmethod
from typing import List
from .models import QueryEmbeddingResult, RetrievalCandidate, RetrievalFilters

class IQueryEncoder(ABC):
    @abstractmethod
    def encode(self, query: str) -> QueryEmbeddingResult: pass

class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, vector, course_ids: List[str], top_k: int, filters: RetrievalFilters) -> List[RetrievalCandidate]: pass

class IFusion(ABC):
    @abstractmethod
    def fuse(self, dense_results: List[RetrievalCandidate], sparse_results: List[RetrievalCandidate]) -> List[RetrievalCandidate]: pass

class IReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrievalCandidate], top_k: int) -> List[RetrievalCandidate]: pass