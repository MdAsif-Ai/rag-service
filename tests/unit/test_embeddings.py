import pytest
from unittest.mock import MagicMock, patch

from app.embeddings.bge_m3 import BGEEmbeddingService, EmbeddingResult
from app.core.exceptions import EmbeddingException

@pytest.fixture
def mock_bge_model():
    # Patch at the source library because it is imported lazily inside the class
    with patch("FlagEmbedding.BGEM3FlagModel") as mock:
        instance = MagicMock()
        mock.return_value = instance
        
        instance.encode.return_value = {
            "dense_vecs": [[0.1, 0.2, 0.3]],
            "lexical_weights": [{1: 0.5, 42: 0.8}]
        }
        yield instance

def test_embed_query_returns_typed_model(mock_bge_model, mock_settings):
    BGEEmbeddingService._instance = None
    service = BGEEmbeddingService(mock_settings)
    result = service.embed_query("test query")
    
    assert isinstance(result, EmbeddingResult)
    assert result.dense_vector == [0.1, 0.2, 0.3]
    assert result.sparse_vector == {1: 0.5, 42: 0.8}

def test_embed_documents_returns_typed_list(mock_bge_model, mock_settings):
    BGEEmbeddingService._instance = None
    service = BGEEmbeddingService(mock_settings)
    
    mock_bge_model.encode.return_value = {
        "dense_vecs": [[0.1, 0.2], [0.3, 0.4]],
        "lexical_weights": [{1: 0.5}, {2: 0.8}]
    }
    
    texts = ["doc one", "doc two"]
    results = service.embed_documents(texts)
    
    assert len(results) == 2
    assert all(isinstance(r, EmbeddingResult) for r in results)
    assert results[0].dense_vector == [0.1, 0.2]
    assert results[1].sparse_vector == {2: 0.8}

def test_empty_query_raises_exception(mock_bge_model, mock_settings):
    BGEEmbeddingService._instance = None
    service = BGEEmbeddingService(mock_settings)
    
    with pytest.raises(EmbeddingException):
        service.embed_query("")

def test_empty_documents_returns_empty_list(mock_bge_model, mock_settings):
    BGEEmbeddingService._instance = None
    service = BGEEmbeddingService(mock_settings)
    
    results = service.embed_documents([])
    assert results == []
    mock_bge_model.encode.assert_not_called()

def test_cpu_fallback_on_cuda_failure(mock_settings):
    BGEEmbeddingService._instance = None
    
    with patch("FlagEmbedding.BGEM3FlagModel") as mock_model_class:
        mock_cuda_instance = MagicMock()
        mock_cuda_instance.encode.side_effect = Exception("CUDA OOM")
        
        mock_cpu_instance = MagicMock()
        mock_cpu_instance.encode.return_value = {
            "dense_vecs": [[0.1]],
            "lexical_weights": [{1: 0.5}]
        }
        
        mock_model_class.side_effect = [Exception("CUDA OOM"), mock_cpu_instance]
        
        mock_settings.EMBEDDING_DEVICE = "cuda"
        
        service = BGEEmbeddingService(mock_settings)
        
        assert mock_settings.EMBEDDING_DEVICE == "cpu"
        
        res = service.embed_query("test")
        assert res.dense_vector == [0.1]