import logging
from typing import List
from docling.document_converter import DocumentConverter
from app.ingestion.loaders.base import DocumentLoader, ParsedSection

logger = logging.getLogger(__name__)

class DoclingLoader(DocumentLoader):
    """
    Universal document loader using Docling.
    Handles PDF, DOCX, PPTX, HTML, and images with automatic OCR and table extraction.
    """
    
    # Initialize the converter once per worker process
    _converter = None

    def __init__(self):
        if DoclingLoader._converter is None:
            logger.info("Initializing Docling DocumentConverter...")
            # Docling will automatically download its models to the HF cache on first run
            DoclingLoader._converter = DocumentConverter()

    def load(self, file_path: str) -> List[ParsedSection]:
        # Docling converts the document into a structured format
        result = DoclingLoader._converter.convert(file_path)
        
        # Docling can export to Markdown, which perfectly preserves tables and headers
        markdown_content = result.document.export_to_markdown()
        
        sections = []
        current_section = "Default"
        current_text = []
        
        # Parse the Markdown to extract sections and text
        for line in markdown_content.split("\n"):
            if line.startswith("#"):
                # Save the previous section
                if current_text:
                    sections.append(ParsedSection(
                        content="\n".join(current_text).strip(),
                        section=current_section,
                        source_type="docling"
                    ))
                    current_text = []
                # Update current section title
                current_section = line.lstrip("#").strip()
            else:
                if line.strip():
                    current_text.append(line)
        
        # Add the final section
        if current_text:
            sections.append(ParsedSection(
                content="\n".join(current_text).strip(),
                section=current_section,
                source_type="docling"
            ))
            
        if not sections:
            # Fallback if no headers were found
            sections.append(ParsedSection(content=markdown_content, source_type="docling"))
            
        return sections