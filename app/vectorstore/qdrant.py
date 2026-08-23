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
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings
from app.core.exceptions import QdrantException


class QdrantRepository:
    """
    Qdrant repository for the LMS knowledge base.

    Supports:
    - Dense BGE-M3 vectors
    - Sparse vectors
    - Course-level filtering
    - Document deletion
    - Deterministic point IDs
    - INT8 scalar quantization
    - Current Qdrant Python client query_points() API
    """

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            return self.client.get_collection(
                self.collection_name
            ) is not None
        except Exception as exc:
            logger.error("Qdrant health check failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # COLLECTION
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(
                collection_name=self.collection_name
            )
        except Exception as exc:
            raise QdrantException(
                "Failed to check collection existence.",
                detail=str(exc),
            ) from exc

    def create_collection(self) -> None:
        if self.collection_exists():
            logger.info(
                "Collection '{}' already exists.",
                self.collection_name,
            )
            return

        try:
            logger.info(
                "Creating Qdrant collection '{}'.",
                self.collection_name,
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams()
                    )
                },
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="course_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            logger.info(
                "Collection '{}' and indexes created successfully.",
                self.collection_name,
            )

        except Exception as exc:
            raise QdrantException(
                "Failed to create Qdrant collection.",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # POINT ID
    # ------------------------------------------------------------------

    def _make_deterministic_id(
        self,
        document_id: uuid.UUID,
        chunk_index: int,
    ) -> str:
        """
        Generate a deterministic UUID for each document chunk.

        The same document + chunk index always produces the same ID,
        making ingestion idempotent.
        """
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document_id}_{chunk_index}",
            )
        )

    # ------------------------------------------------------------------
    # UPSERT
    # ------------------------------------------------------------------

    def upsert_points(
        self,
        document_id: uuid.UUID,
        course_id: str,
        filename: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        if not chunks:
            logger.warning(
                "No chunks provided for document {}",
                document_id,
            )
            return

        points: List[PointStruct] = []

        for chunk in chunks:
            chunk_index = chunk["chunk_index"]

            point_id = self._make_deterministic_id(
                document_id,
                chunk_index,
            )

            payload = {
                "document_id": str(document_id),
                "course_id": course_id,
                "filename": filename,
                "content": chunk["content"],
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "chunk_index": chunk_index,
                "checksum": chunk.get("checksum"),
                "source_type": chunk.get(
                    "source_type",
                    "unknown",
                ),
            }

            # ----------------------------------------------------------
            # Sparse vector
            # ----------------------------------------------------------

            sparse_dict = chunk.get(
                "sparse_vector",
                {},
            )

            sparse_vector = SparseVector(
                indices=list(sparse_dict.keys()),
                values=list(sparse_dict.values()),
            )

            # ----------------------------------------------------------
            # Named vectors
            # ----------------------------------------------------------

            vector_data = {
                "dense": chunk["dense_vector"],
                "sparse": sparse_vector,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector_data,
                    payload=payload,
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(
                "Successfully upserted {} points for document {}",
                len(points),
                document_id,
            )

        except Exception as exc:
            raise QdrantException(
                f"Failed to upsert points for document {document_id}.",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # DELETE DOCUMENT
    # ------------------------------------------------------------------

    def delete_document(
        self,
        document_id: uuid.UUID,
    ) -> None:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(
                                value=str(document_id)
                            ),
                        )
                    ]
                ),
            )

            logger.info(
                "Deleted existing points for document {}",
                document_id,
            )

        except Exception as exc:
            raise QdrantException(
                f"Failed to delete points for document {document_id}.",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # FILTER
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        course_ids: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Filter:
        """
        Build a mandatory course-level filter.

        This prevents retrieval across courses.
        """

        if not course_ids:
            raise QdrantException(
                "course_ids must be provided to prevent "
                "cross-course retrieval."
            )

        conditions = [
            FieldCondition(
                key="course_id",
                match=MatchAny(any=course_ids),
            )
        ]

        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )

        return Filter(must=conditions)

    # ------------------------------------------------------------------
    # DENSE SEARCH
    # ------------------------------------------------------------------

    def search_dense(
        self,
        query_vector: List[float],
        course_ids: List[str],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(
            course_ids,
            filters,
        )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )

            return [
                self._format_search_result(point)
                for point in response.points
            ]

        except Exception as exc:
            raise QdrantException(
                "Dense vector search failed.",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # SPARSE SEARCH
    # ------------------------------------------------------------------

    def search_sparse(
        self,
        sparse_vector: Dict[int, float],
        course_ids: List[str],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filter(
            course_ids,
            filters,
        )

        try:
            qdrant_sparse = SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values()),
            )

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=qdrant_sparse,
                using="sparse",
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )

            return [
                self._format_search_result(point)
                for point in response.points
            ]

        except Exception as exc:
            raise QdrantException(
                "Sparse vector search failed.",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # RESULT FORMATTING
    # ------------------------------------------------------------------

    def _format_search_result(
        self,
        result: Any,
    ) -> Dict[str, Any]:
        payload = result.payload or {}

        return {
            "chunk_id": payload.get(
                "chunk_id",
                result.id,
            ),
            "document_id": payload.get(
                "document_id"
            ),
            "course_id": payload.get(
                "course_id"
            ),
            "filename": payload.get(
                "filename"
            ),
            "content": payload.get(
                "content"
            ),
            "page": payload.get(
                "page"
            ),
            "chapter": payload.get(
                "chapter"
            ),
            "section": payload.get(
                "section"
            ),
            "chunk_index": payload.get(
                "chunk_index"
            ),
            "score": result.score,
            "metadata": payload,
        }


# ----------------------------------------------------------------------
# REPOSITORY FACTORY
# ----------------------------------------------------------------------

def get_qdrant_repository() -> QdrantRepository:
    settings = get_settings()

    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY.get_secret_value(),
            timeout=settings.QDRANT_TIMEOUT,
            prefer_grpc=True,
        )

        return QdrantRepository(
            client=client,
            collection_name=settings.QDRANT_COLLECTION,
        )

    except Exception as exc:
        raise QdrantException(
            "Failed to initialize Qdrant client.",
            detail=str(exc),
        ) from exc