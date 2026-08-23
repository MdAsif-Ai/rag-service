import pytest
import uuid
from unittest.mock import MagicMock, patch
from qdrant_client.http.models import PointStruct

from app.vectorstore.qdrant import QdrantRepository
from app.core.exceptions import QdrantException

@pytest.fixture
def mock_client():
    return MagicMock()

@pytest.fixture
def repo(mock_client):
    return QdrantRepository(client=mock_client, collection_name="test_collection")

def test_collection_exists(repo, mock_client):
    mock_client.collection_exists.return_value = True
    assert repo.collection_exists() is True

def test_create_collection_with_indexes(repo, mock_client):
    mock_client.collection_exists.return_value = False
    repo.create_collection()
    
    mock_client.create_collection.assert_called_once()
    
    mock_client.create_payload_index.assert_any_call(
        collection_name="test_collection",
        field_name="course_id",
        field_schema="keyword"
    )
    mock_client.create_payload_index.assert_any_call(
        collection_name="test_collection",
        field_name="document_id",
        field_schema="keyword"
    )

def test_upsert_points_generates_deterministic_ids(repo, mock_client):
    doc_id = uuid.uuid4()
    chunks = [{
        "chunk_index": 0,
        "content": "test",
        "dense_vector": [0.1] * 1024,
        "sparse_vector": {1: 0.2}
    }]
    
    repo.upsert_points(doc_id, "course-1", "test.pdf", chunks)
    
    mock_client.upsert.assert_called_once()
    args, kwargs = mock_client.upsert.call_args
    point: PointStruct = kwargs["points"][0]
    
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}_0"))
    assert point.id == expected_id
    assert point.payload["course_id"] == "course-1"

def test_delete_document(repo, mock_client):
    doc_id = uuid.uuid4()
    repo.delete_document(doc_id)
    mock_client.delete.assert_called_once()
    args, kwargs = mock_client.delete.call_args
    assert kwargs["collection_name"] == "test_collection"

def test_search_dense_requires_course_ids(repo):
    with pytest.raises(QdrantException, match="course_ids must be provided"):
        repo.search_dense([0.1], [], top_k=5)

def test_search_dense_success(repo, mock_client):
    mock_result = MagicMock()
    mock_result.id = "chunk-1"
    mock_result.score = 0.95
    mock_result.payload = {"document_id": "doc-1", "course_id": "course-1", "content": "text"}
    
    mock_client.search.return_value = [mock_result]
    
    results = repo.search_dense([0.1], ["course-1"], top_k=5)
    assert len(results) == 1
    assert results[0]["score"] == 0.95
    # The repository maps payload to metadata in the formatted result
    assert results[0]["metadata"]["course_id"] == "course-1"

def test_search_sparse_success(repo, mock_client):
    mock_result = MagicMock()
    mock_result.id = "chunk-2"
    mock_result.score = 12.5
    mock_result.payload = {"document_id": "doc-2", "course_id": "course-1", "content": "text"}
    
    mock_client.search.return_value = [mock_result]
    
    results = repo.search_sparse({1: 0.5}, ["course-1"], top_k=5)
    assert len(results) == 1
    assert results[0]["score"] == 12.5