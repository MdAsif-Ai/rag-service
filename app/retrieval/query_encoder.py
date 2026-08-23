import hashlib
from loguru import logger
from app.core.exceptions import EmbeddingException
from app.embeddings.bge_m3 import BGEEmbeddingService
from .models import QueryEmbeddingResult
from .interfaces import IQueryEncoder

class BGEQueryEncoder(IQueryEncoder):
    def __init__(self, embedding_service: BGEEmbeddingService):
        self.embedding_service = embedding_service

    def encode(self, query: str) -> QueryEmbeddingResult:
        try:
            # Query length logging instead of raw query
            q_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
            logger.info(f"Encoding query (hash: {q_hash}, len: {len(query)})")
            
            result = self.embedding_service.embed_query(query)
            return QueryEmbeddingResult(
                dense_vector=result["dense_vector"],
                sparse_vector=result["sparse_vector"]
            )
        except Exception as e:
            raise EmbeddingException("Query encoding failed.", detail=str(e))