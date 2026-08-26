from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from docling.document_converter import DocumentConverter

from app.core.exceptions import DocumentProcessingException
from app.ingestion.loaders.base import DocumentLoader, ParsedSection


logger = logging.getLogger(__name__)


class DoclingLoader(DocumentLoader):
    """
    Universal document loader powered by Docling.

    Docling is used as the single extraction backend for supported
    document formats such as:

        - PDF
        - DOCX
        - PPTX
        - HTML
        - Markdown
        - TXT
        - common image formats

    The loader converts the source document into structured Markdown,
    preserving useful document structure such as headings, tables and
    document ordering as much as Docling supports.
    """

    _converter: Optional[DocumentConverter] = None

    @classmethod
    def _get_converter(cls) -> DocumentConverter:
        """
        Create the Docling converter once per worker process.

        Celery workers should therefore reuse the same converter rather
        than downloading/loading models for every document.
        """
        if cls._converter is None:
            logger.info("Initializing Docling DocumentConverter")

            cls._converter = DocumentConverter()

            logger.info("Docling DocumentConverter initialized")

        return cls._converter

    def load(self, file_path: str) -> List[ParsedSection]:
        """
        Extract structured content from a document.

        Args:
            file_path: Local path to the document.

        Returns:
            A list containing the extracted document content.

        Raises:
            DocumentProcessingException:
                If the file does not exist or Docling cannot process it.
        """
        path = Path(file_path)

        if not path.exists():
            raise DocumentProcessingException(
                f"Document does not exist: {file_path}"
            )

        if not path.is_file():
            raise DocumentProcessingException(
                f"Path is not a file: {file_path}"
            )

        try:
            converter = self._get_converter()

            logger.info(
                "Extracting document with Docling: %s",
                path.name,
            )

            result = converter.convert(str(path))

            document = result.document

            markdown_content = document.export_to_markdown()

            if not markdown_content or not markdown_content.strip():
                raise DocumentProcessingException(
                    f"Docling extracted no usable content from "
                    f"'{path.name}'"
                )

            markdown_content = markdown_content.strip()

            metadata = {
                "filename": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "parser": "docling",
            }

            section = ParsedSection(
                content=markdown_content,
                section=path.stem,
                content_type="document",
                source_type="docling",
                metadata=metadata,
            )

            logger.info(
                "Successfully extracted %d characters from %s",
                len(markdown_content),
                path.name,
            )

            return [section]

        except DocumentProcessingException:
            raise

        except Exception as exc:
            logger.exception(
                "Docling failed to process document: %s",
                file_path,
            )

            raise DocumentProcessingException(
                f"Docling failed to process '{path.name}': {exc}"
            ) from exc