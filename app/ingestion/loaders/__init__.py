from typing import Any, Type, Optional
from app.core.exceptions import UnsupportedFileException
from .base import DocumentLoader, ParsedSection
from .docling_loader import DoclingLoader
from .multimodal_loaders import GeminiVisionLoader, GroqAudioLoader, VideoLoader

LOADER_MAP: dict[str, Type[Any]] = {
    "pdf": DoclingLoader,
    "docx": DoclingLoader,
    "pptx": DoclingLoader,
    "html": DoclingLoader,
    "md": DoclingLoader,
    "txt": DoclingLoader,
    
    # Audio formats
    "mp3": GroqAudioLoader,
    "wav": GroqAudioLoader,
    "m4a": GroqAudioLoader,
    
    # Video formats
    "mp4": VideoLoader,
    "mkv": VideoLoader,
    "avi": VideoLoader,
    
    # Image formats
    "png": GeminiVisionLoader,
    "jpg": GeminiVisionLoader,
    "jpeg": GeminiVisionLoader,
}

def get_loader(file_type: str, content_format: str = "auto", url: str = None) -> DocumentLoader:
    file_type = file_type.lower()
    
    # 1. Handle YouTube URLs
    if url and ("youtube" in url or "youtu.be" in url):
        return GroqAudioLoader(url=url)
        
    # 2. Intelligent Routing for PDFs based on LMS hint
    if file_type == "pdf" and content_format in ["handwritten", "diagram", "scanned"]:
        return GeminiVisionLoader(is_pdf=True)
        
    # 3. Intelligent Routing for Images based on LMS hint
    if file_type in ["png", "jpg", "jpeg"] and content_format in ["handwritten", "diagram"]:
        return GeminiVisionLoader(is_pdf=False)
        
    # 4. Default routing based on file extension
    loader_class = LOADER_MAP.get(file_type)
    if not loader_class:
        raise UnsupportedFileException(f"Unsupported file type: {file_type}")
    return loader_class()

__all__ = ["DocumentLoader", "ParsedSection", "get_loader"]