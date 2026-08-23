import pytest
import asyncio
from unittest.mock import MagicMock, patch

from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.models import RetrievalCandidate, RetrievalFilters, QueryEmbeddingResult
from app.core.exceptions import RetrievalException, ValidationException

@pytest.fixture
def mock_components():
    encoder = MagicMock()
    encoder.encode.return_value = QueryEmbeddingResult(dense_vector=[0.1, 0.2], sparse_vector={1: 0.5})
    
    dense_retriever = MagicMock()
    sparse_retriever = MagicMock()
    
    fusion = MagicMock()
    reranker = MagicMock()
    
    return encoder, dense_retriever, sparse_retriever, fusion, reranker

@pytest.fixture
def pipeline(mock_components):
    enc, dense, sparse, fusion, reranker = mock_components
    return RetrievalPipeline(enc, dense, sparse, fusion, reranker)

@pytest.mark.asyncio
async def test_query_encoded_once(pipeline, mock_components):
    enc, dense, sparse, _, _ = mock_components
    dense.retrieve.return_value = []
    sparse.retrieve.return_value = []
    
    await pipeline.retrieve("test query", ["course1"], 5)
    enc.encode.assert_called_once_with("test query")

@pytest.mark.asyncio
async def test_dense_and_sparse_receive_precomputed_vectors(pipeline, mock_components):
    enc, dense, sparse, _, _ = mock_components
    dense.retrieve.return_value = []
    sparse.retrieve.return_value = []
    
    await pipeline.retrieve("test query", ["course1"], 5)
    
    dense.retrieve.assert_called_once()
    args, _ = dense.retrieve.call_args
    assert args[0] == [0.1, 0.2]  # Dense vector
    
    sparse.retrieve.assert_called_once()
    args, _ = sparse.retrieve.call_args
    assert args[0] == {1: 0.5}  # Sparse vector

@pytest.mark.asyncio
async def test_course_filters_passed_correctly(pipeline, mock_components):
    _, dense, sparse, _, _ = mock_components
    dense.retrieve.return_value = []
    sparse.retrieve.return_value = []
    
    await pipeline.retrieve("test query", ["courseA", "courseB"], 5)
    
    _, kwargs = dense.retrieve.call_args
    assert kwargs["course_ids"] == ["courseA", "courseB"]
    
    _, kwargs = sparse.retrieve.call_args
    assert kwargs["course_ids"] == ["courseA", "courseB"]

@pytest.mark.asyncio
async def test_pipeline_flow_and_top_k(pipeline, mock_components):
    _, dense, sparse, fusion, reranker = mock_components
    
    c1 = RetrievalCandidate(chunk_id="1", document_id="d", course_id="c", filename="f", content="a", chunk_index=0)
    c2 = RetrievalCandidate(chunk_id="2", document_id="d", course_id="c", filename="f", content="b", chunk_index=1)
    
    dense.retrieve.return_value = [c1]
    sparse.retrieve.return_value = [c2]
    fusion.fuse.return_value = [c1, c2]
    reranker.rerank.return_value = [c2, c1]  # Reranked order
    
    with patch('app.retrieval.pipeline.get_settings') as mock_settings:
        mock_settings.return_value.DEFAULT_TOP_K = 50
        mock_settings.return_value.RERANK_TOP_K = 10
        
        response = await pipeline.retrieve("test query", ["course1"], 1)  # Request top_k=1
        
        assert response.final_count == 1
        assert response.results[0].chunk_id == "2"  # Top reranked item
        assert response.total_candidates == 2

@pytest.mark.asyncio
async def test_empty_results_returns_empty_response(pipeline, mock_components):
    _, dense, sparse, _, _ = mock_components
    dense.retrieve.return_value = []
    sparse.retrieve.return_value = []
    
    response = await pipeline.retrieve("test query", ["course1"], 5)
    assert response.results == []
    assert response.final_count == 0

@pytest.mark.asyncio
async def test_infrastructure_failure_raises_exception(pipeline, mock_components):
    _, dense, _, _, _ = mock_components
    dense.retrieve.side_effect = Exception("Qdrant connection refused")
    
    with pytest.raises(RetrievalException):
        await pipeline.retrieve("test query", ["course1"], 5)

@pytest.mark.asyncio
async def test_invalid_query_raises_validation(pipeline):
    with pytest.raises(ValidationException):
        await pipeline.retrieve("", ["course1"], 5)

@pytest.mark.asyncio
async def test_invalid_course_ids_raises_validation(pipeline):
    with pytest.raises(ValidationException):
        await pipeline.retrieve("test query", [], 5)