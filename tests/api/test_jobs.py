import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.v1.jobs import router
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
def mock_job_data():
    return {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "document_id": "123e4567-e89b-12d3-a456-426614174001",
        "status": "PROCESSING",
        "stage": "EMBEDDING",
        "attempts": 1,
        "progress": 0.5,
        "error": None,
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:05:00Z",
        "completed_at": None
    }

def test_valid_job(client, mock_job_data):
    with patch("app.services.jobs.get_supabase_client") as mock_supabase:
        mock_sb_instance = MagicMock()
        mock_supabase.return_value = mock_sb_instance
        
        # Mock Supabase chain to return data
        mock_sb_instance.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[mock_job_data])
        
        response = client.get(f"/api/v1/jobs/{mock_job_data['id']}")
        
        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == mock_job_data["id"]
        assert body["status"] == "PROCESSING"
        assert body["stage"] == "EMBEDDING"
        assert body["progress"] == 0.5

def test_missing_job(client):
    with patch("app.services.jobs.get_supabase_client") as mock_supabase:
        mock_sb_instance = MagicMock()
        mock_supabase.return_value = mock_sb_instance
        
        # Mock Supabase returning empty list
        mock_sb_instance.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        response = client.get("/api/v1/jobs/123e4567-e89b-12d3-a456-426614174999")
        
        assert response.status_code == 404
        assert response.json()["detail"] == "Job not found."

def test_unauthorized_request():
    # Create a secure app without auth override
    secure_app = FastAPI()
    secure_app.include_router(router, prefix="/api/v1")
    secure_client = TestClient(secure_app)
    
    response = secure_client.get("/api/v1/jobs/123e4567-e89b-12d3-a456-426614174000")
    assert response.status_code == 401

def test_failed_job(client):
    failed_data = {
        "id": "123e4567-e89b-12d3-a456-426614174001",
        "document_id": "123e4567-e89b-12d3-a456-426614174002",
        "status": "FAILED",
        "stage": "FAILED",
        "attempts": 3,
        "progress": None,
        "error": "Unsupported file type.",
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:10:00Z",
        "completed_at": "2024-01-01T10:10:00Z"
    }
    
    with patch("app.services.jobs.get_supabase_client") as mock_supabase:
        mock_sb_instance = MagicMock()
        mock_supabase.return_value = mock_sb_instance
        mock_sb_instance.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[failed_data])
        
        response = client.get(f"/api/v1/jobs/{failed_data['id']}")
        
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "FAILED"
        assert body["error"] == "Unsupported file type."

def test_completed_job(client):
    completed_data = {
        "id": "123e4567-e89b-12d3-a456-426614174003",
        "document_id": "123e4567-e89b-12d3-a456-426614174004",
        "status": "COMPLETED",
        "stage": "DONE",
        "attempts": 1,
        "progress": 1.0,
        "error": None,
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:15:00Z",
        "completed_at": "2024-01-01T10:15:00Z"
    }
    
    with patch("app.services.jobs.get_supabase_client") as mock_supabase:
        mock_sb_instance = MagicMock()
        mock_supabase.return_value = mock_sb_instance
        mock_sb_instance.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[completed_data])
        
        response = client.get(f"/api/v1/jobs/{completed_data['id']}")
        
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "COMPLETED"
        assert body["stage"] == "DONE"
        assert body["progress"] == 1.0