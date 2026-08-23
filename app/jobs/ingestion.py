import tempfile
import os
import traceback
from typing import Dict, Any
from loguru import logger
from celery import current_task
from celery.exceptions import Ignore

from app.jobs.celery import celery_app
from app.core.config import get_settings
from app.core.exceptions import (
    DocumentProcessingException, 
    UnsupportedFileException, 
    EmbeddingException,
    QdrantException,
    SupabaseException,
    RAGServiceException
)
from app.db.supabase import get_supabase_client
from app.storage.supabase import storage_service
from app.vectorstore.qdrant import get_qdrant_repository
from app.embeddings.bge_m3 import get_embedding_service
from app.ingestion.chunking import StructureAwareChunker, ParsedSection
from app.ingestion.normalizer import DocumentNormalizer
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.loaders.docx import DOCXLoader
from app.ingestion.loaders.pptx import PPTXLoader
from app.ingestion.loaders.text import TextLoader

# Map file extensions to loader classes
LOADER_MAP: Dict[str, type[DocumentLoader]] = {
    "pdf": PDFLoader,
    "docx": DOCXLoader,
    "pptx": PPTXLoader,
    "txt": TextLoader,
}

def update_job_status(job_id: str, status: str, stage: str, error: str = None):
    """Helper to update job progress in Supabase without excessive writes."""
    try:
        supabase = get_supabase_client()
        update_data = {"status": status, "stage": stage, "updated_at": "now()"}
        if error:
            update_data["error"] = error
            update_data["completed_at"] = "now()"
        elif status == "COMPLETED":
            update_data["completed_at"] = "now()"
            
        supabase.table("ingestion_jobs").update(update_data).eq("id", job_id).execute()
        logger.info(f"Job {job_id} status updated: {status} / {stage}")
    except Exception as e:
        logger.error(f"Failed to update job status in DB for {job_id}: {e}")

@celery_app.task(
    bind=True, 
    name="ingest_document",
    autoretry_for=(SupabaseException, QdrantException, EmbeddingException, ConnectionError),
    retry_backoff=True, 
    retry_backoff_max=300,
    max_retries=3
)
def ingest_document(self, document_id: str, job_id: str):
    """
    Main ingestion pipeline executed by Celery workers.
    """
    logger.info(f"Starting ingestion for document {document_id}, job {job_id}")
    settings = get_settings()
    
    try:
        update_job_status(job_id, "PROCESSING", "DOWNLOADING")
        supabase = get_supabase_client()
        
        # 1. Load Document Metadata
        doc_res = supabase.table("documents").select("*").eq("id", document_id).single().execute()
        if not doc_res.data:
            raise DocumentProcessingException(f"Document metadata not found for {document_id}")
        doc_metadata = doc_res.data
        
        file_type = doc_metadata.get("file_type", "").lower()
        storage_path = doc_metadata["storage_path"]
        filename = doc_metadata["filename"]
        course_id = doc_metadata["course_id"]
        
        # 2. Retrieve Source File
        file_bytes = storage_service.download_file(storage_path)
        
        # Save to temp file for parsers
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
            
        try:
            # 3. Select Loader
            loader_class = LOADER_MAP.get(file_type)
            if not loader_class:
                raise UnsupportedFileException(f"Unsupported file type: {file_type}")
                
            # 4. Parse
            update_job_status(job_id, "PROCESSING", "PARSING")
            loader = loader_class()
            raw_sections = loader._safe_load(tmp_file_path)
            
            # 5. Normalize
            normalizer = DocumentNormalizer()
            normalized_sections = normalizer.normalize(raw_sections)
            
            # 6. Chunk
            update_job_status(job_id, "PROCESSING", "CHUNKING")
            chunker = StructureAwareChunker(
                chunk_size_tokens=settings.CHUNK_SIZE_TOKENS,
                chunk_overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
                tokenizer_name=settings.CHUNK_TOKENIZER
            )
            chunks = chunker.chunk(
                sections=normalized_sections,
                document_id=document_id,
                course_id=course_id,
                filename=filename
            )
            
            if not chunks:
                raise DocumentProcessingException("Document yielded no processable chunks.")
            
            # 7. Embed
            update_job_status(job_id, "PROCESSING", "EMBEDDING")
            embedding_service = get_embedding_service()
            texts = [c.content for c in chunks]
            embeddings = embedding_service.embed_documents(texts)
            
            # 8. Prepare for Upsert
            upsert_data = []
            for chunk, emb in zip(chunks, embeddings):
                upsert_data.append({
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "page": chunk.page,
                    "section": chunk.section,
                    "dense_vector": emb["dense_vector"],
                    "sparse_vector": emb["sparse_vector"],
                })
            
            # 9. Upsert to Qdrant
            update_job_status(job_id, "PROCESSING", "INDEXING")
            qdrant_repo = get_qdrant_repository()
            
            # Idempotency: Delete old chunks first to prevent stale data on re-index
            qdrant_repo.delete_document(document_id)
            qdrant_repo.upsert_points(
                document_id=document_id,
                course_id=course_id,
                filename=filename,
                chunks=upsert_data
            )
            
            # 10. Mark Completed
            supabase.table("documents").update({
                "status": "INDEXED", 
                "updated_at": "now()",
                "chunk_count": len(chunks)
            }).eq("id", document_id).execute()
            
            update_job_status(job_id, "COMPLETED", "DONE")
            logger.info(f"Successfully ingested document {document_id}")
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                
    except (UnsupportedFileException, DocumentProcessingException) as e:
        # Permanent failures: do not retry
        logger.error(f"Permanent ingestion failure for {document_id}: {e}")
        update_job_status(job_id, "FAILED", "FAILED", error=str(e.default_message))
        supabase.table("documents").update({"status": "FAILED"}).eq("id", document_id).execute()
        raise Ignore()
        
    except Exception as e:
        # Transient failures: retry
        logger.error(f"Transient ingestion failure for {document_id}: {e}\n{traceback.format_exc()}")
        update_job_status(job_id, "FAILED", "FAILED", error="Transient infrastructure error. Retrying.")
        raise self.retry(exc=e)