import tempfile
import os
import traceback
from typing import Optional, Dict, Any, cast
from loguru import logger
from celery.exceptions import Ignore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.jobs.celery import celery_app
from app.core.config import get_settings
from app.core.exceptions import (
    DocumentProcessingException, 
    UnsupportedFileException, 
    QdrantException,
    SupabaseException
)
from app.db.supabase import get_supabase_client
from app.storage.supabase import storage_service
from app.vectorstore.qdrant import get_qdrant_repository
from app.embeddings.bge_m3 import get_embedding_service
from app.ingestion.chunking import StructureAwareChunker
from app.ingestion.normalizer import DocumentNormalizer
from app.ingestion.loaders import get_loader

class TransientDBError(Exception):
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TransientDBError),
    reraise=True
)
def update_job_status(job_id: str, status: str, stage: str, error: Optional[str] = None):
    try:
        supabase = get_supabase_client()
        update_data: Dict[str, Any] = {"status": status, "stage": stage, "updated_at": "now()"}
        if error:
            update_data["error"] = error
            update_data["completed_at"] = "now()"
        elif status == "COMPLETED":
            update_data["completed_at"] = "now()"
            
        supabase.table("ingestion_jobs").update(update_data).eq("id", job_id).execute()
        logger.info(f"Job {job_id} status updated: {status} / {stage}")
    except Exception as e:
        logger.error(f"Failed to update job status in DB for {job_id}: {e}")
        raise TransientDBError(str(e))

@celery_app.task(bind=True, name="ingest_document")
def ingest_document(self, document_id: str, job_id: str):
    logger.info(f"Starting ingestion for document {document_id}, job {job_id}")
    settings = get_settings()
    
    try:
        update_job_status(job_id, "PROCESSING", "DOWNLOADING")
        supabase = get_supabase_client()
        
        doc_res = supabase.table("documents").select("*").eq("id", document_id).single().execute()
        if not doc_res.data:
            raise DocumentProcessingException(f"Document metadata not found for {document_id}")
        
        # Cast to dict for mypy
        doc_metadata = cast(Dict[str, Any], doc_res.data)
        
        file_type = str(doc_metadata.get("file_type", "")).lower()
        storage_path = str(doc_metadata["storage_path"])
        filename = str(doc_metadata["filename"])
        course_id = str(doc_metadata["course_id"])
        
        file_bytes = storage_service.download_file(storage_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
            
        try:
            update_job_status(job_id, "PROCESSING", "PARSING")
            loader = get_loader(file_type)
            raw_sections = loader._safe_load(tmp_file_path)
            
            update_job_status(job_id, "PROCESSING", "NORMALIZING")
            normalizer = DocumentNormalizer()
            normalized_sections = normalizer.normalize(raw_sections)
            
            update_job_status(job_id, "PROCESSING", "CHUNKING")
            chunker = StructureAwareChunker(
                chunk_size_tokens=settings.CHUNK_SIZE_TOKENS,
                chunk_overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
                tokenizer_name=settings.CHUNK_TOKENIZER
            )
            
            # Convert string UUID to actual UUID object for chunker
            import uuid
            doc_uuid = uuid.UUID(document_id)
            
            chunks = chunker.chunk(
                sections=normalized_sections,
                document_id=doc_uuid,
                course_id=course_id,
                filename=filename
            )
            
            if not chunks:
                raise DocumentProcessingException("Document yielded no processable chunks.")
            
            update_job_status(job_id, "PROCESSING", "EMBEDDING")
            embedding_service = get_embedding_service()
            texts = [c.content for c in chunks]
            embeddings = embedding_service.embed_documents(texts)
            
            upsert_data = []
            for chunk, emb in zip(chunks, embeddings):
                upsert_data.append({
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "page": chunk.page,
                    "section": chunk.section,
                    "source_type": chunk.source_type,
                    "dense_vector": emb.dense_vector,
                    "sparse_vector": emb.sparse_vector,
                })
            
            update_job_status(job_id, "PROCESSING", "INDEXING")
            qdrant_repo = get_qdrant_repository()
            
            qdrant_repo.delete_document(doc_uuid)
            qdrant_repo.upsert_points(
                document_id=doc_uuid,
                course_id=course_id,
                filename=filename,
                chunks=upsert_data
            )
            
            supabase.table("documents").update({
                "status": "INDEXED", 
                "updated_at": "now()",
                "chunk_count": len(chunks)
            }).eq("id", document_id).execute()
            
            update_job_status(job_id, "COMPLETED", "DONE")
            logger.info(f"Successfully ingested document {document_id}")
            
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                
    except (UnsupportedFileException, DocumentProcessingException) as e:
        logger.error(f"Permanent ingestion failure for {document_id}: {e}")
        try:
            update_job_status(job_id, "FAILED", "FAILED", error=e.message)
            supabase.table("documents").update({"status": "FAILED"}).eq("id", document_id).execute()
        except Exception as db_err:
            logger.error(f"Failed to mark job {job_id} as FAILED in DB: {db_err}")
        raise Ignore()
        
    except Exception as e:
        logger.error(f"Transient ingestion failure for {document_id}: {e}\n{traceback.format_exc()}")
        
        if self.request.retries >= self.max_retries:
            logger.error(f"Max retries exceeded for job {job_id}. Marking as FAILED.")
            try:
                update_job_status(job_id, "FAILED", "FAILED", error="Max retries exceeded. Transient infrastructure error.")
                supabase.table("documents").update({"status": "FAILED"}).eq("id", document_id).execute()
            except Exception as db_err:
                logger.error(f"Failed to mark job {job_id} as FAILED in DB: {db_err}")
            raise Ignore()
            
        try:
            update_job_status(job_id, "PROCESSING", "RETRYING", error=f"Attempt {self.request.retries + 1} failed. Retrying...")
        except Exception as db_err:
            logger.error(f"Failed to update job status to RETRYING: {db_err}")
            
        raise self.retry(exc=e)