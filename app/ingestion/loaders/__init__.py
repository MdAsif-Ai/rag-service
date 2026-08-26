from typing import Type

from app.core.exceptions import UnsupportedFileException

from .base import DocumentLoader, ParsedSection
from .docling_loader import DoclingLoader


# Docling is the single extraction backend.
#
# Supported formats are intentionally explicit rather than accepting
# arbitrary extensions and failing later inside the ingestion pipeline.
LOADER_MAP: dict[str, Type[DocumentLoader]] = {
    # Documents
    "pdf": DoclingLoader,
    "docx": DoclingLoader,

    # Presentations
    "pptx": DoclingLoader,

    # Web / markup
    "html": DoclingLoader,
    "htm": DoclingLoader,
    "md": DoclingLoader,
    "markdown": DoclingLoader,

    # Plain text
    "txt": DoclingLoader,

    # Images
    "png": DoclingLoader,
    "jpg": DoclingLoader,
    "jpeg": DoclingLoader,
}


def get_loader(file_type: str) -> DocumentLoader:
    """
    Return the appropriate document loader.

    Args:
        file_type:
            File extension such as 'pdf', '.pdf', 'pptx', etc.

    Returns:
        DocumentLoader instance.

    Raises:
        UnsupportedFileException:
            If the extension is not supported.
    """
    normalized_type = file_type.lower().strip().lstrip(".")

    loader_class = LOADER_MAP.get(normalized_type)

    if loader_class is None:
        raise UnsupportedFileException(
            f"Unsupported file type: {normalized_type}"
        )

    return loader_class()


__all__ = [
    "DocumentLoader",
    "ParsedSection",
    "DoclingLoader",
    "get_loader",
]