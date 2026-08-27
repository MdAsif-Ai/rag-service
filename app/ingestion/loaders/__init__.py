from typing import Any, Type
from app.core.exceptions import UnsupportedFileException
from .base import DocumentLoader, ParsedSection
from .docling_loader import DoclingLoader

LOADER_MAP: dict[str, Type[Any]] = {
    "pdf": DoclingLoader,
    "docx": DoclingLoader,
    "pptx": DoclingLoader,
    "html": DoclingLoader,
    "htm": DoclingLoader,
    "md": DoclingLoader,
    "markdown": DoclingLoader,
    "txt": DoclingLoader,
    "png": DoclingLoader,
    "jpg": DoclingLoader,
    "jpeg": DoclingLoader,
}

def get_loader(file_type: str) -> DocumentLoader:
    file_type = file_type.lower()
    loader_class = LOADER_MAP.get(file_type)
    if not loader_class:
        raise UnsupportedFileException(f"Unsupported file type: {file_type}")
    return loader_class()

__all__ = ["DocumentLoader", "ParsedSection", "get_loader", "DoclingLoader"]