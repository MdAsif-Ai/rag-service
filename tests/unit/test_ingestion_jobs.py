import pytest
from unittest.mock import MagicMock, patch
from celery.exceptions import Ignore

from app.jobs.ingestion import ingest_document
from app.core.exceptions import DocumentProcessingException, QdrantException

@pytest.fixture
def mock_dependencies():
    with patch("app.jobs.ingestion.get_supabase_client") as mock_supabase, \
         patch("app.jobs.ingestion.storage_service") as mock_storage, \
         patch("app.jobs.ingestion.get_qdrant_repository") as mock_qdrant, \
         patch("app.jobs.ingestion.get_embedding_service") as mock_embed, \
         patch("app.jobs.ingestion.StructureAwareChunker") as mock_chunker, \
         patch("app.jobs.ingestion.DocumentNormalizer") as mock_normalizer, \
         patch("app.jobs.ingestion.PDFLoader") as mock_loader, \
         patch("app.jobs.ingestion.update_job_status") as mock_status:
        
        # Configure Supabase mock
        supabase_instance = MagicMock()
        mock_supabase.return_value = supabase_instance
        
        # Configure Storage mock
        mock_storage.download_file.return_value = b"fake pdf bytes"
        
        # Configure Loader mock
        loader_instance = MagicMock()
        mock_loader.return_value = loader_instance
        loader_instance._safe_load.return_value = [MagicMock(content="text", page=1, section="s")]
        
        # Configure Normalizer mock
        normalizer_instance = MagicMock()
        mock_normalizer.return_value = normalizer_instance
        normalizer_instance.normalize.return_value = [MagicMock(content="normalized text", page=1, section="s")]
        
        # Configure Chunker mock
        chunker_instance = MagicMock()
        mock_chunker.return_value = chunker_instance
        mock_chunk = MagicMock()
        mock_chunk.content = "chunk content"
        mock_chunk.chunk_index = 0
        mock_chunk.page = 1
        mock_chunk.section = "s"
        chunker_instance.chunk.return_value = [mock_chunk]
        
        # Configure Embedder mock
        embed_instance = MagicMock()
        mock_embed.return_value = embed_instance
        embed_instance.embed_documents.return_value = [{"dense_vector": [0.1], "sparse_vector": {1: 0.2}}]
        
        # Configure Qdrant mock
        qdrant_instance = MagicMock()
        mock_qdrant.return_value = qdrant_instance
        
        yield {
            "supabase": supabase_instance,
            "storage": mock_storage,
            "loader": loader_instance,
            "normalizer": normalizer_instance,
            "chunker": chunker_instance,
            "embedder": embed_instance,
            "qdrant": qdrant_instance,
            "status": mock_status
        }

def test_successful_ingestion(mock_dependencies):
    mocks = mock_dependencies
    # Mock DB returns document metadata
    mocks["supabase"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"id": "doc1", "file_type": "pdf", "storage_path": "path", "filename": "f.pdf", "course_id": "c1"}
    )
    
    # Celery context mock
    task = MagicMock()
    task.retry = MagicMock(side_effect=Exception("Should not retry"))
    
    ingest_document(task, "doc1", "job1")
    
    mocks["storage"].download_file.assert_called_once_with("path")
    mocks["loader"]._safe_load.assert_called_once()
    mocks["qdrant"].delete_document.assert_called_once_with("doc1")
    mocks["qdrant"].upsert_points.assert_called_once()
    mocks["status"].assert_any_call("job1", "COMPLETED", "DONE")

def test_permanent_failure_unsupported_file(mock_dependencies):
    mocks = mock_dependencies
    mocks["supabase"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"id": "doc1", "file_type": "xyz", "storage_path": "path", "filename": "f.xyz", "course_id": "c1"}
    )
    
    task = MagicMock()
    
    with pytest.raises(Ignore):
        ingest_document(task, "doc1", "job1")
        
    mocks["status"].assert_any_call("job1", "FAILED", "FAILED", error="Unsupported file type.")

def test_transient_failure_qdrant_down(mock_dependencies):
    mocks = mock_dependencies
    mocks["supabase"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"id": "doc1", "file_type": "pdf", "storage_path": "path", "filename": "f.pdf", "course_id": "c1"}
    )
    mocks["qdrant"].upsert_points.side_effect = QdrantException("Connection lost")
    
    task = MagicMock()
    task.retry = MagicMock(side_effect=Exception("Retried"))
    
    with pytest.raises(Exception, match="Retried"):
        ingest_document(task, "doc1", "job1")
        
    task.retry.assert_called_once()
    mocks["status"].assert_any_call("job1", "FAILED", "FAILED", error="Transient infrastructure error. Retrying.")