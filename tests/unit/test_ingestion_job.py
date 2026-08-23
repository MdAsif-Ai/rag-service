import pytest
from unittest.mock import MagicMock, patch
from celery.exceptions import Ignore

from app.jobs.ingestion import ingest_document, TransientDBError
from app.core.exceptions import DocumentProcessingException, QdrantException, UnsupportedFileException

@pytest.fixture
def mock_dependencies():
    with patch("app.jobs.ingestion.get_supabase_client") as mock_supabase, \
         patch("app.jobs.ingestion.storage_service") as mock_storage, \
         patch("app.jobs.ingestion.get_qdrant_repository") as mock_qdrant, \
         patch("app.jobs.ingestion.get_embedding_service") as mock_embed, \
         patch("app.jobs.ingestion.StructureAwareChunker") as mock_chunker, \
         patch("app.jobs.ingestion.DocumentNormalizer") as mock_normalizer, \
         patch("app.jobs.ingestion.get_loader") as mock_loader, \
         patch("app.jobs.ingestion.update_job_status") as mock_status:
        
        supabase_instance = MagicMock()
        mock_supabase.return_value = supabase_instance
        mock_storage.download_file.return_value = b"fake pdf bytes"
        
        loader_instance = MagicMock()
        mock_loader.return_value = loader_instance
        loader_instance._safe_load.return_value = [MagicMock(content="text", page=1, section="s", source_type="pdf")]
        
        normalizer_instance = MagicMock()
        mock_normalizer.return_value = normalizer_instance
        normalizer_instance.normalize.return_value = [MagicMock(content="normalized text", page=1, section="s", source_type="pdf")]
        
        chunker_instance = MagicMock()
        mock_chunker.return_value = chunker_instance
        mock_chunk = MagicMock()
        mock_chunk.content = "chunk content"
        mock_chunk.chunk_index = 0
        mock_chunk.page = 1
        mock_chunk.section = "s"
        mock_chunk.source_type = "pdf"
        chunker_instance.chunk.return_value = [mock_chunk]
        
        embed_instance = MagicMock()
        mock_embed.return_value = embed_instance
        from app.embeddings.bge_m3 import EmbeddingResult
        embed_instance.embed_documents.return_value = [EmbeddingResult(dense_vector=[0.1], sparse_vector={1: 0.2})]
        
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
    mocks["supabase"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"id": "doc1", "file_type": "pdf", "storage_path": "path", "filename": "f.pdf", "course_id": "c1"}
    )
    
    task = MagicMock()
    task.request.retries = 0
    task.max_retries = 3
    task.retry = MagicMock(side_effect=Exception("Should not retry"))
    
    # Use .run to invoke the bound task function directly
    ingest_document.run(task, "doc1", "job1")
    
    task.retry.assert_not_called()
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
    
    mocks["loader"]._safe_load.side_effect = UnsupportedFileException("Unsupported file type: xyz")
    
    task = MagicMock()
    task.request.retries = 0
    task.max_retries = 3
    task.retry = MagicMock(side_effect=Exception("Should not retry"))
    
    with pytest.raises(Ignore):
        ingest_document.run(task, "doc1", "job1")
        
    task.retry.assert_not_called()
    mocks["status"].assert_any_call("job1", "FAILED", "FAILED", error="Unsupported file type: xyz")

def test_transient_failure_qdrant_down(mock_dependencies):
    mocks = mock_dependencies
    mocks["supabase"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"id": "doc1", "file_type": "pdf", "storage_path": "path", "filename": "f.pdf", "course_id": "c1"}
    )
    mocks["qdrant"].upsert_points.side_effect = QdrantException("Connection lost")
    
    task = MagicMock()
    task.request.retries = 0
    task.max_retries = 3
    task.retry = MagicMock(side_effect=Exception("Retried"))
    
    with pytest.raises(Exception, match="Retried"):
        ingest_document.run(task, "doc1", "job1")
        
    task.retry.assert_called_once()
    mocks["status"].assert_any_call("job1", "PROCESSING", "RETRYING", error="Attempt 1 failed. Retrying...")

def test_max_retries_exceeded_marks_failed(mock_dependencies):
    mocks = mock_dependencies
    mocks["supabase"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"id": "doc1", "file_type": "pdf", "storage_path": "path", "filename": "f.pdf", "course_id": "c1"}
    )
    mocks["qdrant"].upsert_points.side_effect = QdrantException("Connection lost")
    
    task = MagicMock()
    task.request.retries = 3
    task.max_retries = 3
    task.retry = MagicMock(side_effect=Exception("Should not retry"))
    
    with pytest.raises(Ignore):
        ingest_document.run(task, "doc1", "job1")
        
    task.retry.assert_not_called()
    mocks["status"].assert_any_call("job1", "FAILED", "FAILED", error="Max retries exceeded. Transient infrastructure error.")

def test_job_status_update_failure_triggers_transient_retry(mock_dependencies):
    mocks = mock_dependencies
    mocks["supabase"].table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
        data={"id": "doc1", "file_type": "pdf", "storage_path": "path", "filename": "f.pdf", "course_id": "c1"}
    )
    
    mocks["status"].side_effect = TransientDBError("Supabase connection lost")
    
    task = MagicMock()
    task.request.retries = 0
    task.max_retries = 3
    task.retry = MagicMock(side_effect=Exception("Retried"))
    
    with pytest.raises(Exception, match="Retried"):
        ingest_document.run(task, "doc1", "job1")
        
    task.retry.assert_called_once()