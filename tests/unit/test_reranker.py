import pytest
from unittest.mock import MagicMock, patch

from app.retrieval.reranker import BGEReranker
from app.retrieval.models import RetrievalCandidate
from app.core.exceptions import RetrievalException, ValidationException
from app.core.config import Settings

@pytest.fixture
def mock_settings():
    # Create a mock settings object to avoid loading .env during tests
    settings = MagicMock(spec=Settings)
    settings.RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    settings.RERANKER_DEVICE = "cpu"
    settings.RERANK_BATCH_SIZE = 2
    settings.RERANK_MAX_CANDIDATES = 5
    settings.RERANK_FINAL_TOP_K = 3
    return settings

@pytest.fixture
def mock_flag_reranker():
    # Patch the FlagReranker class where it is imported in the module
    with patch("app.retrieval.reranker.FlagReranker", autospec=True) as mock_class:
        # The instance returned when FlagReranker() is called
        mock_instance = mock_class.return_value
        yield mock_instance

@pytest.fixture
def reranker_service(mock_settings, mock_flag_reranker):
    return BGEReranker(mock_settings)

def make_candidate(cid, content="test content"):
    return RetrievalCandidate(
        chunk_id=cid,
        document_id="doc1",
        course_id="course1",
        filename="file.pdf",
        content=content,
        chunk_index=0
    )

def test_empty_candidates_returns_empty(reranker_service):
    result = reranker_service.rerank("query", [])
    assert result == []

def test_empty_query_raises_validation(reranker_service):
    with pytest.raises(ValidationException):
        reranker_service.rerank("", [make_candidate("c1")])

def test_missing_content_raises_validation(reranker_service):
    c1 = make_candidate("c1", content=None)
    with pytest.raises(ValidationException):
        reranker_service.rerank("query", [c1])

def test_normal_reranking_and_sorting(reranker_service, mock_flag_reranker):
    c1 = make_candidate("c1", "good match")
    c2 = make_candidate("c2", "better match")
    
    # Mock the model returning a higher score for c2
    mock_flag_reranker.compute_score.return_value = [0.4, 0.9]
    
    result = reranker_service.rerank("query", [c1, c2], top_k=2)
    
    assert len(result) == 2
    assert result[0].chunk_id == "c2"
    assert result[0].rerank_score == 0.9
    assert result[1].chunk_id == "c1"
    assert result[1].rerank_score == 0.4
    
    # Verify model was called correctly
    mock_flag_reranker.compute_score.assert_called_once()
    args, kwargs = mock_flag_reranker.compute_score.call_args
    assert args[0] == [["query", "good match"], ["query", "better match"]]
    assert kwargs["batch_size"] == 2

def test_max_candidates_enforced(reranker_service, mock_flag_reranker):
    # Pass 10 candidates, but max is 5
    candidates = [make_candidate(f"c{i}") for i in range(10)]
    
    # Return 5 scores
    mock_flag_reranker.compute_score.return_value = [0.1 * i for i in range(5)]
    
    result = reranker_service.rerank("query", candidates, top_k=5)
    
    assert len(result) == 5
    # Verify model only received 5 pairs
    args, _ = mock_flag_reranker.compute_score.call_args
    assert len(args[0]) == 5

def test_top_k_slicing(reranker_service, mock_flag_reranker):
    c1 = make_candidate("c1")
    c2 = make_candidate("c2")
    c3 = make_candidate("c3")
    
    mock_flag_reranker.compute_score.return_value = [0.2, 0.9, 0.5]
    
    # Request top_k=1
    result = reranker_service.rerank("query", [c1, c2, c3], top_k=1)
    assert len(result) == 1
    assert result[0].chunk_id == "c2" # Highest score

def test_model_failure_raises_exception(reranker_service, mock_flag_reranker):
    mock_flag_reranker.compute_score.side_effect = RuntimeError("CUDA OOM")
    
    with pytest.raises(RetrievalException):
        reranker_service.rerank("query", [make_candidate("c1")])

def test_single_candidate_returns_list(reranker_service, mock_flag_reranker):
    c1 = make_candidate("c1")
    
    # FlagReranker returns a float if only 1 pair is passed
    mock_flag_reranker.compute_score.return_value = 0.85
    
    result = reranker_service.rerank("query", [c1], top_k=1)
    
    assert len(result) == 1
    assert result[0].chunk_id == "c1"
    assert result[0].rerank_score == 0.85

def test_deterministic_tie_breaking(reranker_service, mock_flag_reranker):
    c_z = make_candidate("chunk_z", "content z")
    c_a = make_candidate("chunk_a", "content a")
    
    # Both get identical scores
    mock_flag_reranker.compute_score.return_value = [0.5, 0.5]
    
    result = reranker_service.rerank("query", [c_z, c_a], top_k=2)
    
    # Should sort by chunk_id ASC since scores are equal
    assert result[0].chunk_id == "chunk_a"
    assert result[1].chunk_id == "chunk_z"