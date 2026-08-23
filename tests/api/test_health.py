import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.v1.health import router
from app.core.security import verify_health_api_key

# Create a test FastAPI app
app = FastAPI()
app.include_router(router)

# Override auth dependency to allow testing without API keys
app.dependency_overrides[verify_health_api_key] = lambda: True

@pytest.fixture
def client():
    return TestClient(app)

def test_health_success(client):
    """Liveness should always return 200 if the process is running."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.services.health.check_qdrant", new_callable=AsyncMock)
@patch("app.services.health.check_redis", new_callable=AsyncMock)
@patch("app.services.health.check_supabase", new_callable=AsyncMock)
def test_readiness_success(mock_sb, mock_redis, mock_qdrant, client):
    """Readiness should return 200 if all dependencies are healthy."""
    mock_qdrant.return_value = True
    mock_redis.return_value = True
    mock_sb.return_value = True
    
    response = client.get("/ready")
    
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["dependencies"]["qdrant"] is True
    assert body["dependencies"]["redis"] is True
    assert body["dependencies"]["database"] is True

@patch("app.services.health.check_qdrant", new_callable=AsyncMock)
@patch("app.services.health.check_redis", new_callable=AsyncMock)
@patch("app.services.health.check_supabase", new_callable=AsyncMock)
def test_qdrant_unavailable(mock_sb, mock_redis, mock_qdrant, client):
    """Should return 503 if Qdrant check fails."""
    mock_qdrant.return_value = False
    mock_redis.return_value = True
    mock_sb.return_value = True
    
    response = client.get("/ready")
    
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["dependencies"]["qdrant"] is False

@patch("app.services.health.check_qdrant", new_callable=AsyncMock)
@patch("app.services.health.check_redis", new_callable=AsyncMock)
@patch("app.services.health.check_supabase", new_callable=AsyncMock)
def test_redis_unavailable(mock_sb, mock_redis, mock_qdrant, client):
    """Should return 503 if Redis check fails."""
    mock_qdrant.return_value = True
    mock_redis.return_value = False
    mock_sb.return_value = True
    
    response = client.get("/ready")
    
    assert response.status_code == 503
    assert response.json()["dependencies"]["redis"] is False

@patch("app.services.health.check_qdrant", new_callable=AsyncMock)
@patch("app.services.health.check_redis", new_callable=AsyncMock)
@patch("app.services.health.check_supabase", new_callable=AsyncMock)
def test_dependency_timeout(mock_sb, mock_redis, mock_qdrant, client):
    """Should return 503 if a dependency times out (raises exception)."""
    import asyncio
    mock_qdrant.return_value = True
    mock_redis.side_effect = asyncio.TimeoutError()
    mock_sb.return_value = True
    
    response = client.get("/ready")
    
    assert response.status_code == 503
    assert response.json()["dependencies"]["redis"] is False