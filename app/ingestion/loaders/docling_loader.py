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
    Universal document loader using Docling.

    Docling handles supported formats such as:

    - PDF
    - DOCX
    - PPTX
    - HTML
    - Markdown
    - TXT
    - Images

    The extracted document is converted to Markdown so the rest of
    the RAG pipeline can operate on one normalized representation.
    """

    _converter: Optional[DocumentConverter] = None

    @classmethod
    def _get_converter(cls) -> DocumentConverter:
        """
        Create the Docling converter once per worker process.

        Celery workers reuse the converter instead of initializing it
        for every document.
        """

        if cls._converter is None:
            logger.info("Initializing Docling DocumentConverter...")

            cls._converter = DocumentConverter()

            logger.info("Docling DocumentConverter initialized")

        return cls._converter

    def load(self, file_path: str) -> List[ParsedSection]:
        """
        Extract content from a document using Docling.

        Args:
            file_path: Path to the uploaded document.

        Returns:
            List of ParsedSection objects.
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

            # ---------------------------------------------------------
            # IMPORTANT:
            # Depending on the installed Docling version/API,
            # convert() may return:
            #
            #   ConversionResult
            #
            # or an iterator/generator of ConversionResult objects.
            #
            # Handle both forms.
            # ---------------------------------------------------------

            conversion = converter.convert(str(path))

            if hasattr(conversion, "document"):
                # Single ConversionResult
                result = conversion

            else:
                # Generator / iterator
                try:
                    result = next(iter(conversion))
                except StopIteration:
                    raise DocumentProcessingException(
                        f"Docling returned no conversion result for "
                        f"'{path.name}'"
                    )

            document = result.document

            if document is None:
                raise DocumentProcessingException(
                    f"Docling returned no document for '{path.name}'"
                )

            # ---------------------------------------------------------
            # Export the structured document to Markdown.
            #
            # Markdown preserves useful RAG structure including:
            # headings
            # paragraphs
            # tables
            # lists
            # ordering
            # ---------------------------------------------------------

            markdown_content = document.export_to_markdown()

            if not markdown_content:
                raise DocumentProcessingException(
                    f"Docling extracted no content from '{path.name}'"
                )

            markdown_content = markdown_content.strip()

            if not markdown_content:
                raise DocumentProcessingException(
                    f"Docling extracted empty content from '{path.name}'"
                )

            # ---------------------------------------------------------
            # Metadata
            # ---------------------------------------------------------

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