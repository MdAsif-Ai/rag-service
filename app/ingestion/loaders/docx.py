from typing import List
from docx import Document

from app.core.exceptions import DocumentProcessingException
from .base import DocumentLoader, ParsedSection


class DOCXLoader(DocumentLoader):
    """Loads DOCX files, grouping text by heading styles."""

    def load(self, file_path: str) -> List[ParsedSection]:
        doc = Document(file_path)
        sections: List[ParsedSection] = []
        
        current_section = "Default"
        current_text = []

        for para in doc.paragraphs:
            if not para.text.strip():
                continue
                
            # Check if paragraph is a heading
            if para.style.name.startswith("Heading"):
                if current_text:
                    sections.append(
                        ParsedSection(
                            content="\n".join(current_text),
                            section=current_section
                        )
                    )
                    current_text = []
                current_section = para.text.strip()
            else:
                current_text.append(para.text)

        if current_text:
            sections.append(
                ParsedSection(
                    content="\n".join(current_text),
                    section=current_section
                )
            )

        if not sections:
            raise DocumentProcessingException("DOCX contained no extractable text.")
            
        return sections