import pytest
from unittest.mock import MagicMock, patch
from app.embeddings.bge_m3 import BGEEmbeddingService

@pytest.fixture
def mock_bge_model():
    with patch("app.embeddings.bge_m3.BGEM3FlagModel") as mock:
        instance = MagicMock()
        mock.return_value = instance
        instance.encode.return_value = {
            "dense_vecs": [[0.1, 0.2]],
            "lexical_weights": [{1: 0.5}]
        }
        yield instance

def test_embed_query_returns_dense_and_sparse(mock_bge_model, mock_settings):
    service = BGEEmbeddingService(mock_settings)
    result = service.embed_query("test query")
    
    assert "dense_vector" in result
    assert "sparse_vector" in result
    assert result["dense_vector"] == [0.1, 0.2]
    assert result["sparse_vector"] == {1: 0.5}