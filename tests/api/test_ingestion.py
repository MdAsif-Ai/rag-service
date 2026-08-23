import pytest
import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.v1.ingestion import router
from app.core.security import verify_api_key

# Create a test FastAPI app
app = FastAPI()
app.include_router(router, prefix="/api/v1")

# Override auth dependency to always pass
app.dependency_overrides[verify_api_key] = lambda: True

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_file():
    return io.BytesIO(b"fake pdf content"), "test.pdf"

def test_successful_upload(client):
    with patch("app.services.ingestion.get_supabase_client") as mock_supabase, \
         patch("app.services.ingestion.storage_service") as mock_storage, \
         patch("app.services.ingestion.ingest_document.delay") as mock_delay:
        
        # Mock Supabase chains
        mock_sb_instance = MagicMock()
        mock_supabase.return_value = mock_sb_instance
        
        # Duplicate check returns empty
        mock_sb_instance.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        # Insert returns success
        mock_sb_instance.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "123"}])
        mock_sb_instance.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
        
        # Mock Storage
        mock_storage.upload_file.return_value = "path/to/file.pdf"
        
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("test.pdf", io.BytesIO(b"fake pdf content"), "application/pdf")},
            data={"course_id": "course123"}
        )
        
        assert response.status_code == 202
        assert "document_id" in response.json()
        assert "job_id" in response.json()
        mock_delay.assert_called_once()

def test_invalid_extension(client):
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.exe", io.BytesIO(b"fake content"), "application/octet-stream")},
        data={"course_id": "course123"}
    )
    assert response.status_code == 415

def test_oversized_file(client):
    # Patch settings to limit file size to ~1 byte for this test
    with patch("app.services.ingestion.get_settings") as mock_settings:
        # 0.000001 MB * 1024 * 1024 = 1.04 bytes. 
        # The uploaded file is 12 bytes, so it will be rejected.
        mock_settings.return_value.MAX_FILE_SIZE_MB = 0.000001
        mock_settings.return_value.SUPPORTED_FILE_TYPES = ["pdf"]
        
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("test.pdf", io.BytesIO(b"fake content"), "application/pdf")},
            data={"course_id": "course123"}
        )
        assert response.status_code == 413

def test_duplicate_document(client):
    with patch("app.services.ingestion.get_supabase_client") as mock_supabase:
        mock_sb_instance = MagicMock()
        mock_supabase.return_value = mock_sb_instance
        
        # Duplicate check returns existing record
        mock_sb_instance.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": "existing_doc"}])
        
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("test.pdf", io.BytesIO(b"fake pdf content"), "application/pdf")},
            data={"course_id": "course123"}
        )
        assert response.status_code == 409

def test_unauthorized_user():
    # Create a new client without the auth override
    secure_app = FastAPI()
    secure_app.include_router(router, prefix="/api/v1")
    secure_client = TestClient(secure_app)
    
    response = secure_client.post(
        "/api/v1/ingest",
        files={"file": ("test.pdf", io.BytesIO(b"fake content"), "application/pdf")},
        data={"course_id": "course123"}
    )
    assert response.status_code == 401 # Missing API Key

def test_storage_failure(client):
    with patch("app.services.ingestion.get_supabase_client") as mock_supabase, \
         patch("app.services.ingestion.storage_service") as mock_storage:
        
        mock_sb_instance = MagicMock()
        mock_supabase.return_value = mock_sb_instance
        mock_sb_instance.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        mock_sb_instance.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "123"}])
        
        # Simulate storage failure
        mock_storage.upload_file.side_effect = Exception("S3 connection lost")
        
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("test.pdf", io.BytesIO(b"fake pdf content"), "application/pdf")},
            data={"course_id": "course123"}
        )
        
        assert response.status_code == 500
        assert "Failed to upload file to storage" in response.json().get("detail", "")
        # Verify compensating DB cleanup was called
        mock_sb_instance.table.return_value.delete.return_value.eq.return_value.execute.assert_called()

def test_queue_failure(client):
    with patch("app.services.ingestion.get_supabase_client") as mock_supabase, \
         patch("app.services.ingestion.storage_service") as mock_storage, \
         patch("app.services.ingestion.ingest_document.delay") as mock_delay:
        
        mock_sb_instance = MagicMock()
        mock_supabase.return_value = mock_sb_instance
        mock_sb_instance.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        mock_sb_instance.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": "123"}])
        
        mock_storage.upload_file.return_value = "path/to/file.pdf"
        # Simulate Celery broker being down
        mock_delay.side_effect = Exception("Redis connection refused")
        
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("test.pdf", io.BytesIO(b"fake pdf content"), "application/pdf")},
            data={"course_id": "course123"}
        )
        
        assert response.status_code == 500
        assert "Failed to queue background ingestion job" in response.json().get("detail", "")