import pytest
from pydantic import ValidationError
from app.schemas.retrieval import RetrievalRequest

def test_valid_retrieval_request():
    req = RetrievalRequest(query="test", course_ids=["c1"], top_k=5)
    assert req.top_k == 5

def test_empty_query_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="", course_ids=["c1"])

def test_empty_course_ids_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="test", course_ids=[])

def test_top_k_limit_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="test", course_ids=["c1"], top_k=100)