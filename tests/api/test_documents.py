import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.v1.documents import router
from app.core.security import verify_api_key

# Create a test FastAPI app
app = FastAPI()
app.include_router(router, prefix="/api/v1")

# Override auth dependency
app.dependency_overrides[verify_api_key] = lambda: True

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_doc_data():
    return {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "course_id": "course-1",
        "filename": "test.pdf",
        "file_type": "pdf",
        "file_size": 1024,
        "storage_path": "path/to/test.pdf",
        "checksum": "abc123",
        "status": "INDEXED",
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:05:00Z",
        "chunk_count": 42
    }

@pytest.fixture
def mock_job_data():
    return {
        "id": "123e4567-e89b-12d3-a456-426614174111",
        "document_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "COMPLETED",
        "stage": "DONE",
        "attempts": 1,
        "error": None,
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:05:00Z",
        "completed_at": "2024-01-01T10:05:00Z"
    }

def test_document_retrieval(client, mock_doc_data, mock_job_data):
    with patch("app.services.documents.get_supabase_client") as mock_supabase:
        mock_sb = MagicMock()
        mock_supabase.return_value = mock_sb
        
        # Mock Doc query
        mock_sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[mock_doc_data])
        # Mock Job query (using chaining for the order/limit)
        mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[mock_job_data])
        
        response = client.get(f"/api/v1/documents/{mock_doc_data['id']}")
        
        assert response.status_code == 200
        body = response.json()
        assert body["document_id"] == mock_doc_data["id"]
        assert body["status"] == "INDEXED"
        assert body["latest_job"]["job_id"] == mock_job_data["id"]

def test_missing_document(client):
    with patch("app.services.documents.get_supabase_client") as mock_supabase:
        mock_sb = MagicMock()
        mock_supabase.return_value = mock_sb
        mock_sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        response = client.get("/api/v1/documents/123e4567-e89b-12d3-a456-426614174999")
        assert response.status_code == 404

def test_reindex_success(client, mock_doc_data):
    with patch("app.services.documents.get_supabase_client") as mock_supabase, \
         patch("app.services.documents.ingest_document.delay") as mock_delay:
        
        mock_sb = MagicMock()
        mock_supabase.return_value = mock_sb
        
        # 1. Doc exists and is INDEXED
        mock_sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.side_effect = [
            MagicMock(data=[mock_doc_data]), # First call (get doc)
            MagicMock(data=[mock_job_data])  # Second call (not strictly needed for reindex but good for consistency if we fetched it)
        ]
        # Make insert and update return generic success
        mock_sb.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "new_job"}])
        mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
        
        response = client.post(f"/api/v1/documents/{mock_doc_data['id']}/reindex")
        
        assert response.status_code == 202
        assert response.json()["status"] == "QUEUED"
        mock_delay.assert_called_once()

def test_unauthorized_reindex():
    secure_app = FastAPI()
    secure_app.include_router(router, prefix="/api/v1")
    secure_client = TestClient(secure_app)
    
    response = secure_client.post("/api/v1/documents/123e4567-e89b-12d3-a456-426614174000/reindex")
    assert response.status_code == 401

def test_duplicate_reindex_handling(client, mock_doc_data):
    # Simulate document already PROCESSING
    processing_doc = mock_doc_data.copy()
    processing_doc["status"] = "PROCESSING"
    
    with patch("app.services.documents.get_supabase_client") as mock_supabase:
        mock_sb = MagicMock()
        mock_supabase.return_value = mock_sb
        mock_sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[processing_doc])
        
        response = client.post(f"/api/v1/documents/{processing_doc['id']}/reindex")
        
        assert response.status_code == 409
        assert "already being processed" in response.json()["detail"]

def test_queue_failure(client, mock_doc_data):
    with patch("app.services.documents.get_supabase_client") as mock_supabase, \
         patch("app.services.documents.ingest_document.delay") as mock_delay:
        
        mock_sb = MagicMock()
        mock_supabase.return_value = mock_sb
        mock_sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[mock_doc_data])
        mock_sb.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "new_job"}])
        mock_sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
        
        # Simulate Celery broker down
        mock_delay.side_effect = Exception("Redis connection refused")
        
        response = client.post(f"/api/v1/documents/{mock_doc_data['id']}/reindex")
        
        assert response.status_code == 500
        assert "Failed to queue reindex job" in response.json()["detail"]