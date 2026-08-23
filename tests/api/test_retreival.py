import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.v1.retrieval import router, get_pipeline
from app.core.security import verify_api_key
from app.core.exceptions import RetrievalException
from app.retrieval.models import RetrievalCandidate, RetrievalMetrics

# Create a test FastAPI app
app = FastAPI()
app.include_router(router, prefix="/api/v1")

# Override auth dependency
app.dependency_overrides[verify_api_key] = lambda: True

@pytest.fixture
def mock_pipeline():
    """Mocks the RetrievalPipeline to avoid actual Qdrant/Model calls."""
    pipeline = MagicMock()
    pipeline.retrieve = AsyncMock()
    
    # Override the dependency to return our mock
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return pipeline

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_candidate():
    return RetrievalCandidate(
        chunk_id="chunk-1",
        document_id="doc-1",
        course_id="course-1",
        filename="test.pdf",
        content="This is the retrieved text.",
        page=1,
        chapter="Intro",
        section="Background",
        chunk_index=0,
        dense_score=0.9,
        sparse_score=12.5,
        fusion_score=0.032,
        rerank_score=0.95,
        metadata={"source_type": "pdf"}
    )

@pytest.fixture
def mock_metrics():
    return RetrievalMetrics(
        query_hash="abcd1234", embedding_latency_ms=10.0, dense_latency_ms=5.0,
        sparse_latency_ms=5.0, fusion_latency_ms=1.0, reranking_latency_ms=20.0,
        total_latency_ms=36.0, dense_candidates=5, sparse_candidates=5,
        fused_candidates=8, final_candidates=1
    )

def test_successful_retrieval(client, mock_pipeline, mock_candidate, mock_metrics):
    # Setup mock pipeline response
    mock_pipeline.retrieve.return_value = MagicMock(
        results=[mock_candidate],
        total_candidates=1,
        final_count=1,
        timings=mock_metrics
    )
    
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "What is Python?",
            "course_ids": ["course-1"],
            "top_k": 5
        }
    )
    
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is Python?"
    assert body["final_count"] == 1
    assert len(body["results"]) == 1
    assert body["results"][0]["chunk_id"] == "chunk-1"
    assert body["results"][0]["rerank_score"] == 0.95
    assert body["results"][0]["content"] == "This is the retrieved text."

def test_missing_query(client):
    response = client.post(
        "/api/v1/retrieve",
        json={
            "course_ids": ["course-1"],
            "top_k": 5
        }
    )
    assert response.status_code == 422

def test_empty_query(client):
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "",
            "course_ids": ["course-1"],
            "top_k": 5
        }
    )
    assert response.status_code == 422

def test_missing_course_ids(client):
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "What is Python?",
            "top_k": 5
        }
    )
    assert response.status_code == 422

def test_empty_course_ids(client):
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "What is Python?",
            "course_ids": [],
            "top_k": 5
        }
    )
    assert response.status_code == 422

def test_invalid_top_k(client):
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "What is Python?",
            "course_ids": ["course-1"],
            "top_k": 100  # Exceeds the 50 limit
        }
    )
    assert response.status_code == 422

def test_unauthorized_request():
    # Create a secure app without auth override
    secure_app = FastAPI()
    secure_app.include_router(router, prefix="/api/v1")
    secure_client = TestClient(secure_app)
    
    response = secure_client.post(
        "/api/v1/retrieve",
        json={
            "query": "What is Python?",
            "course_ids": ["course-1"],
            "top_k": 5
        }
    )
    assert response.status_code == 401

def test_pipeline_failure(client, mock_pipeline):
    # Simulate Qdrant going down inside the pipeline
    mock_pipeline.retrieve.side_effect = RetrievalException("Qdrant connection lost")
    
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "What is Python?",
            "course_ids": ["course-1"],
            "top_k": 5
        }
    )
    
    assert response.status_code == 503
    assert "Retrieval service temporarily unavailable" in response.json().get("detail", "")

def test_empty_results(client, mock_pipeline, mock_metrics):
    mock_pipeline.retrieve.return_value = MagicMock(
        results=[],
        total_candidates=0,
        final_count=0,
        timings=mock_metrics
    )
    
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "Obscure topic not in DB",
            "course_ids": ["course-1"],
            "top_k": 5
        }
    )
    
    assert response.status_code == 200
    body = response.json()
    assert body["final_count"] == 0
    assert len(body["results"]) == 0

def test_response_schema_security(client, mock_pipeline, mock_candidate, mock_metrics):
    # Verify no internal secrets are leaked
    mock_pipeline.retrieve.return_value = MagicMock(
        results=[mock_candidate],
        total_candidates=1,
        final_count=1,
        timings=mock_metrics
    )
    
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "test",
            "course_ids": ["c1"],
            "top_k": 1
        }
    )
    
    body_str = response.text.lower()
    assert "api_key" not in body_str
    assert "qdrant_url" not in body_str
    assert "password" not in body_str