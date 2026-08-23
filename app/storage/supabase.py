import re
from typing import Any, Dict, Optional
from uuid import UUID

from loguru import logger

from app.core.config import get_settings
from app.core.exceptions import SupabaseException
from app.db.supabase import get_supabase_client


class SupabaseStorageService:
    """
    Service for handling original document files in Supabase Storage.
    Ensures files are stored safely using their document_id and never logged.
    """

    def __init__(self):
        self.client = get_supabase_client()
        self.bucket_name = get_settings().SUPABASE_STORAGE_BUCKET

    @staticmethod
    def generate_storage_path(document_id: UUID, filename: str) -> str:
        """
        Generates a safe, unique storage path based on the document ID.
        Format: {document_id}/{sanitized_filename}
        """
        # Sanitize filename to remove potentially dangerous characters
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        return f"{document_id}/{safe_filename}"

    def _get_file_info(self, storage_path: str) -> Optional[Dict[str, Any]]:
        """
        Internal method to list files in the storage path's directory and find the exact file.
        Used for both existence checks and metadata retrieval.
        """
        path_parts = storage_path.split('/')
        prefix = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else ''
        file_name = path_parts[-1]

        try:
            files = self.client.storage.from_(self.bucket_name).list(path=prefix)
            for f in files:
                if f.get('name') == file_name:
                    return f
            return None
        except Exception as e:
            logger.error(f"Failed to query Supabase storage metadata for path: {storage_path}")
            raise SupabaseException("Failed to query file metadata.", detail=str(e))

    def upload_file(self, file_data: bytes, document_id: UUID, filename: str, content_type: str = "application/octet-stream") -> str:
        """
        Uploads a file to Supabase Storage.
        Returns the storage path.
        """
        storage_path = self.generate_storage_path(document_id, filename)

        try:
            # Do not log file_data
            logger.info(f"Uploading file to Supabase storage at path: {storage_path}")
            
            self.client.storage.from_(self.bucket_name).upload(
                path=storage_path,
                file=file_data,
                file_options={"content-type": content_type, "upsert": "false"}
            )
            logger.info(f"Successfully uploaded file to {storage_path}")
            return storage_path
            
        except Exception as e:
            logger.error(f"Failed to upload file to Supabase storage at {storage_path}. Size: {len(file_data)} bytes.")
            raise SupabaseException("Failed to upload file to storage.", detail=str(e))

    def download_file(self, storage_path: str) -> bytes:
        """
        Downloads a file from Supabase Storage.
        """
        try:
            logger.info(f"Downloading file from Supabase storage path: {storage_path}")
            
            res = self.client.storage.from_(self.bucket_name).download(storage_path)
            
            if not res:
                raise SupabaseException("File not found or empty response from storage.", detail=f"Path: {storage_path}")
                
            logger.info(f"Successfully downloaded file from {storage_path}")
            return res
            
        except SupabaseException:
            raise
        except Exception as e:
            logger.error(f"Failed to download file from Supabase storage at {storage_path}")
            raise SupabaseException("Failed to download file from storage.", detail=str(e))

    def file_exists(self, storage_path: str) -> bool:
        """
        Checks if a file exists in Supabase Storage.
        """
        info = self._get_file_info(storage_path)
        return info is not None

    def get_file_metadata(self, storage_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves file metadata (size, last modified, mimetype) from Supabase Storage.
        """
        return self._get_file_info(storage_path)


# Singleton instance for easy access
storage_service = SupabaseStorageService()