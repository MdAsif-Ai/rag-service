import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings
from app.core.exceptions import QdrantException


class QdrantRepository:
    """
    Production-quality Qdrant repository for LMS knowledge base.
    Supports BGE-M3 dense (1024) and sparse vectors.
    """

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def health_check(self) -> bool:
        """Checks if Qdrant is reachable and responding."""
        try:
            return self.client.get_collection(self.collection_name) is not None
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def collection_exists(self) -> bool:
        """Checks if the configured collection exists."""
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            raise QdrantException("Failed to check collection existence.", detail=str(e))

    def create_collection(self) -> None:
        """
        Creates the Qdrant collection with BGE-M3 compatible configuration.
        BGE-M3 produces 1024-dimensional dense vectors and sparse lexical vectors.
        """
        if self.collection_exists():
            logger.info(f"Collection '{self.collection_name}' already exists.")
            return

        try:
            logger.info(f"Creating Qdrant collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams())
                },
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
            logger.info("Collection created successfully.")
        except Exception as e:
            raise QdrantException("Failed to create Qdrant collection.", detail=str(e))

    def _make_deterministic_id(self, document_id: uuid.UUID, chunk_index: int) -> str:
        """Generates a deterministic UUID for a chunk based on document_id and chunk_index."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}_{chunk_index}"))

    def upsert_points(self, document_id: uuid.UUID, course_id: str, filename: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Upserts a list of processed document chunks into Qdrant.
        Each chunk dict must contain: content, chunk_index, dense_vector, sparse_vector, 
        and optionally: page, section, checksum.
        """
        if not chunks:
            logger.warning(f"No chunks provided for upsertion for document {document_id}")
            return

        points = []
        for chunk in chunks:
            chunk_idx = chunk["chunk_index"]
            point_id = self._make_deterministic_id(document_id, chunk_idx)
            
            payload = {
                "document_id": str(document_id),
                "course_id": course_id,
                "filename": filename,
                "content": chunk["content"],
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "chunk_index": chunk_idx,
                "checksum": chunk.get("checksum"),
            }
            
            vector_data = {
                "dense": chunk["dense_vector"],
                "sparse": chunk["sparse_vector"],
            }
            
            points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully upserted {len(points)} points for document {document_id}")
        except Exception as e:
            raise QdrantException(f"Failed to upsert points for document {document_id}.", detail=str(e))

    def delete_document(self, document_id: uuid.UUID) -> None:
        """
        Deletes all vectors associated with a given document_id.
        Used internally when reindexing a document.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id))
                        )
                    ]
                )
            )
            logger.info(f"Deleted existing points for document {document_id}")
        except Exception as e:
            raise QdrantException(f"Failed to delete points for document {document_id}.", detail=str(e))

    def _build_filter(self, course_ids: List[str], filters: Optional[Dict[str, Any]] = None) -> Filter:
        """Builds a Qdrant filter object for retrieval."""
        conditions = [
            FieldCondition(key="course_id", match=MatchAny(any=course_ids))
        ]
        
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                    
        return Filter(must=conditions)

    def search_dense(self, query_vector: List[float], course_ids: List[str], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a dense vector search.
        """
        qdrant_filter = self._build_filter(course_ids, filters)
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_vector),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            return [self._format_search_result(r) for r in results]
        except Exception as e:
            raise QdrantException("Dense vector search failed.", detail=str(e))

    def search_sparse(self, sparse_vector: Dict[int, float], course_ids: List[str], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a sparse vector (lexical) search.
        """
        qdrant_filter = self._build_filter(course_ids, filters)
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("sparse", sparse_vector),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            return [self._format_search_result(r) for r in results]
        except Exception as e:
            raise QdrantException("Sparse vector search failed.", detail=str(e))

    def _format_search_result(self, result: Any) -> Dict[str, Any]:
        """Formats a Qdrant search result into a standard dictionary."""
        payload = result.payload or {}
        return {
            "chunk_id": payload.get("chunk_id", result.id),
            "document_id": payload.get("document_id"),
            "course_id": payload.get("course_id"),
            "filename": payload.get("filename"),
            "content": payload.get("content"),
            "page": payload.get("page"),
            "chapter": payload.get("chapter"),
            "section": payload.get("section"),
            "chunk_index": payload.get("chunk_index"),
            "score": result.score,
            "metadata": payload
        }



# --- Dependency Injection / Factory ---

def get_qdrant_repository() -> QdrantRepository:
    """
    Factory function to create a QdrantRepository instance.
    Reads configuration from application settings.
    """
    settings = get_settings()
    
    try:
        # Use prefer_grpc=True for faster communication if port 6334 is open
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY.get_secret_value(),
            prefer_grpc=True if settings.QDRANT_URL.startswith("http") else False
        )
        return QdrantRepository(client=client, collection_name=settings.QDRANT_COLLECTION)
    except Exception as e:
        raise QdrantException("Failed to initialize Qdrant client.", detail=str(e))