import logging
from pathlib import Path
from typing import Any, List, Union
from docling.document_converter import DocumentConverter

from app.ingestion.loaders.base import DocumentLoader, ParsedSection

logger = logging.getLogger(__name__)

class DoclingLoader(DocumentLoader):
    """
    Universal document loader using Docling.
    Handles PDF, DOCX, PPTX, HTML, and images with automatic OCR and table extraction.
    """
    
    _converter: Union[DocumentConverter, None] = None

    def __init__(self) -> None:
        if DoclingLoader._converter is None:
            logger.info("Initializing Docling DocumentConverter...")
            DoclingLoader._converter = DocumentConverter()
        self.converter = DoclingLoader._converter

    def load(self, file_path: str) -> List[ParsedSection]:
        """
        Parse a document with Docling and convert the resulting document
        into ParsedSection objects used by the RAG ingestion pipeline.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Document path is not a file: {path}")

        try:
            result = self._convert(path)
            document = self._extract_document(result)

            if document is None:
                raise RuntimeError(f"Docling returned no document for '{path.name}'")

            markdown = document.export_to_markdown()

            if not markdown or not markdown.strip():
                raise RuntimeError(f"Docling extracted no text from '{path.name}'")

            return self._markdown_to_sections(
                markdown=markdown,
                filename=path.name,
            )

        except Exception as exc:
            raise RuntimeError(f"Docling failed to process '{path.name}': {exc}") from exc

    def _convert(self, path: Path) -> Any:
        """
        Handle both public Docling return styles encountered across
        Docling v2 installations (ConversionResult or Generator).
        """
        result = self.converter.convert(str(path))

        # Normal public ConversionResult.
        if hasattr(result, "document"):
            return result

        # Generator / iterator returned by the installed Docling build.
        if hasattr(result, "__next__"):
            try:
                return next(result)
            except StopIteration as exc:
                raise RuntimeError(f"Docling returned an empty conversion iterator for '{path.name}'") from exc

        # Other iterable implementations.
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict, list, tuple)):
            iterator = iter(result)
            try:
                return next(iterator)
            except StopIteration as exc:
                raise RuntimeError(f"Docling returned an empty conversion iterator for '{path.name}'") from exc

        raise TypeError(f"Unsupported Docling conversion result type: {type(result).__name__}")

    @staticmethod
    def _extract_document(result: Any) -> Any:
        """Extract Docling's DoclingDocument from a normalized ConversionResult."""
        document = getattr(result, "document", None)
        if document is not None:
            return document
        raise AttributeError(
            f"Docling conversion result does not contain a 'document' attribute. Received: {type(result).__name__}"
        )

    @staticmethod
    def _markdown_to_sections(markdown: str, filename: str) -> List[ParsedSection]:
        """Convert Docling Markdown into ParsedSection objects."""
        lines = markdown.splitlines()
        sections: List[ParsedSection] = []
        
        chapter = ""
        section = ""
        subsection = ""
        
        current_lines: List[str] = []
        current_heading = ""

        def flush() -> None:
            nonlocal current_lines
            content = "\n".join(current_lines).strip()
            if not content:
                current_lines = []
                return
            
            sections.append(ParsedSection(
                content=content,
                page=None,
                chapter=chapter,
                section=section or subsection,
                content_type="text",
                source_type="docling",
                metadata={
                    "filename": filename,
                    "chapter": chapter,
                    "section": section,
                    "subsection": subsection,
                    "heading": current_heading,
                },
            ))
            current_lines = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("### "):
                flush()
                subsection = stripped[4:].strip()
                current_heading = subsection
                current_lines.append(line)
                continue

            if stripped.startswith("## "):
                flush()
                section = stripped[3:].strip()
                subsection = ""
                current_heading = section
                current_lines.append(line)
                continue

            if stripped.startswith("# "):
                flush()
                chapter = stripped[2:].strip()
                section = ""
                subsection = ""
                current_heading = chapter
                current_lines.append(line)
                continue

            current_lines.append(line)

        flush()
        return sections