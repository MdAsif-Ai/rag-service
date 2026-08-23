import pytest
from app.retrieval.fusion import RRFFusion
from app.retrieval.models import RetrievalCandidate
from app.core.exceptions import ValidationException

@pytest.fixture
def fusion_service():
    return RRFFusion(k=60, dense_weight=1.0, sparse_weight=1.0)

def make_candidate(cid, content="text", dense_score=None, sparse_score=None, chunk_idx=0):
    return RetrievalCandidate(
        chunk_id=cid,
        document_id="doc1",
        course_id="course1",
        filename="file.pdf",
        content=content,
        chunk_index=chunk_idx,
        dense_score=dense_score,
        sparse_score=sparse_score,
        metadata={"source": "test"}
    )

def test_dense_only_results(fusion_service):
    dense = [make_candidate("c1", dense_score=0.9), make_candidate("c2", dense_score=0.8)]
    sparse = []
    
    result = fusion_service.fuse(dense, sparse)
    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].dense_score == 0.9
    assert result[0].sparse_score is None

def test_sparse_only_results(fusion_service):
    dense = []
    sparse = [make_candidate("c1", sparse_score=10.5), make_candidate("c2", sparse_score=8.1)]
    
    result = fusion_service.fuse(dense, sparse)
    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].sparse_score == 10.5
    assert result[0].dense_score is None

def test_overlapping_results(fusion_service):
    dense = [make_candidate("c1", dense_score=0.9)]
    sparse = [make_candidate("c1", sparse_score=10.5)]
    
    result = fusion_service.fuse(dense, sparse)
    assert len(result) == 1
    assert result[0].chunk_id == "c1"
    assert result[0].dense_score == 0.9
    assert result[0].sparse_score == 10.5

def test_rrf_score_calculation(fusion_service):
    # Rank 0 in dense: 1.0 / (60 + 0 + 1) = 1/61
    # Rank 0 in sparse: 1.0 / (60 + 0 + 1) = 1/61
    # Expected = 2/61
    dense = [make_candidate("c1")]
    sparse = [make_candidate("c1")]
    
    result = fusion_service.fuse(dense, sparse)
    expected_score = (1.0 / 61) + (1.0 / 61)
    assert abs(result[0].fusion_score - expected_score) < 1e-9

def test_weighted_rrf():
    service = RRFFusion(k=60, dense_weight=2.0, sparse_weight=1.0)
    dense = [make_candidate("c1")]
    sparse = [make_candidate("c1")]
    
    result = service.fuse(dense, sparse)
    expected_score = (2.0 * (1.0 / 61)) + (1.0 * (1.0 / 61))
    assert abs(result[0].fusion_score - expected_score) < 1e-9

def test_deterministic_ordering(fusion_service):
    # Two candidates with identical RRF scores should break ties by chunk_id ASC
    c1 = make_candidate("chunk_b")
    c2 = make_candidate("chunk_a")
    
    # Put both at rank 0 in respective lists to give them identical fusion scores
    dense = [c1]
    sparse = [c2]
    
    result = fusion_service.fuse(dense, sparse)
    assert result[0].chunk_id == "chunk_a"
    assert result[1].chunk_id == "chunk_b"
    assert result[0].fusion_score == result[1].fusion_score

def test_empty_results(fusion_service):
    result = fusion_service.fuse([], [])
    assert result == []

def test_invalid_top_k(fusion_service):
    with pytest.raises(ValidationException):
        fusion_service.fuse([], [], top_k=0)
    with pytest.raises(ValidationException):
        fusion_service.fuse([], [], top_k=-5)

def test_metadata_preservation(fusion_service):
    c1 = make_candidate("c1", content="Specific content", chunk_idx=5)
    dense = [c1]
    sparse = []
    
    result = fusion_service.fuse(dense, sparse)
    assert result[0].content == "Specific content"
    assert result[0].chunk_index == 5
    assert result[0].metadata == {"source": "test"}

def test_missing_chunk_id_raises(fusion_service):
    c1 = make_candidate("c1")
    c1.chunk_id = None  # Simulate malformed data
    
    with pytest.raises(ValidationException):
        fusion_service.fuse([c1], [])