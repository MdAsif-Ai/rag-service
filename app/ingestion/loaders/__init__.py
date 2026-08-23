from typing import Any, Type
from app.core.exceptions import UnsupportedFileException
from .base import DocumentLoader, ParsedSection
from .pdf import PDFLoader
from .docx import DOCXLoader
from .pptx import PPTXLoader
from .xlsx import XLSXLoader
from .html import HTMLLoader
from .markdown import MarkdownLoader
from .text import TextLoader

LOADER_MAP: dict[str, Type[Any]] = {
    "pdf": PDFLoader,
    "docx": DOCXLoader,
    "pptx": PPTXLoader,
    "xlsx": XLSXLoader,
    "html": HTMLLoader,
    "htm": HTMLLoader,
    "md": MarkdownLoader,
    "markdown": MarkdownLoader,
    "txt": TextLoader,
}

def get_loader(file_type: str) -> DocumentLoader:
    file_type = file_type.lower()
    loader_class = LOADER_MAP.get(file_type)
    if not loader_class:
        raise UnsupportedFileException(f"Unsupported file type: {file_type}")
    return loader_class()

__all__ = ["DocumentLoader", "ParsedSection", "get_loader"]