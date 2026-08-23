import uuid
from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
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
    Production Qdrant repository for LMS knowledge base.
    Enforces strict course_id isolation and supports BGE-M3 dense/sparse vectors.
    """

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def health_check(self) -> bool:
        try:
            return self.client.get_collection(self.collection_name) is not None
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            raise QdrantException("Failed to check collection existence.", detail=str(e))

    def create_collection(self) -> None:
        """
        Creates the Qdrant collection with BGE-M3 compatible configuration.
        Creates payload indexes for course_id and document_id to ensure fast filtering.
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
            
            # Create payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="course_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            logger.info("Collection and indexes created successfully.")
        except Exception as e:
            raise QdrantException("Failed to create Qdrant collection.", detail=str(e))

    def _make_deterministic_id(self, document_id: uuid.UUID, chunk_index: int) -> str:
        """Generates a deterministic UUID for a chunk based on document_id and chunk_index."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}_{chunk_index}"))

    def upsert_points(self, document_id: uuid.UUID, course_id: str, filename: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Batch upserts document chunks into Qdrant.
        Enforces required metadata fields.
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
                "source_type": chunk.get("source_type", "unknown")
            }
            
            vector_data = {
                "dense": chunk["dense_vector"],
                "sparse": chunk["sparse_vector"],
            }
            
            points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))

        try:
            # Qdrant client handles batching internally for large lists
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully upserted {len(points)} points for document {document_id}")
        except Exception as e:
            raise QdrantException(f"Failed to upsert points for document {document_id}.", detail=str(e))

    def delete_document(self, document_id: uuid.UUID) -> None:
        """
        Deletes all vectors associated with a given document_id.
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
        """
        Builds a Qdrant filter object. 
        STRICT REQUIREMENT: course_ids cannot be empty to prevent cross-course data leakage.
        """
        if not course_ids:
            raise QdrantException("course_ids must be provided to prevent cross-course retrieval.")

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
        """Performs a dense vector search with strict course_id filtering."""
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
        """Performs a sparse vector (lexical) search with strict course_id filtering."""
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


def get_qdrant_repository() -> QdrantRepository:
    """Factory function with explicit timeout configuration."""
    settings = get_settings()
    
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY.get_secret_value(),
            timeout=settings.QDRANT_TIMEOUT,
            prefer_grpc=True if settings.QDRANT_URL.startswith("http") else False
        )
        return QdrantRepository(client=client, collection_name=settings.QDRANT_COLLECTION)
    except Exception as e:
        raise QdrantException("Failed to initialize Qdrant client.", detail=str(e))