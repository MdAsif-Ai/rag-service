from typing import List
import fitz  # PyMuPDF

from app.core.exceptions import DocumentProcessingException
from .base import DocumentLoader, ParsedSection


class PDFLoader(DocumentLoader):
    """Loads PDF files using PyMuPDF, preserving page numbers."""

    def load(self, file_path: str) -> List[ParsedSection]:
        sections: List[ParsedSection] = []
        
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    sections.append(
                        ParsedSection(
                            content=text.strip(),
                            page=page_num,
                            metadata={"char_count": len(text)}
                        )
                    )
                    
        if not sections:
            raise DocumentProcessingException("PDF contained no extractable text.")
            
        return sections