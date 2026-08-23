from typing import List
from docx import Document
from app.ingestion.loaders.base import DocumentLoader, ParsedSection

class DOCXLoader(DocumentLoader):
    def load(self, file_path: str) -> List[ParsedSection]:
        doc = Document(file_path)
        sections = []
        current_section = "Default"
        current_text = []

        for para in doc.paragraphs:
            if not para.text.strip():
                continue
            if para.style.name.startswith("Heading"):
                if current_text:
                    sections.append(ParsedSection(
                        content="\n".join(current_text),
                        section=current_section,
                        source_type="docx"
                    ))
                    current_text = []
                current_section = para.text.strip()
            else:
                current_text.append(para.text)

        if current_text:
            sections.append(ParsedSection(
                content="\n".join(current_text),
                section=current_section,
                source_type="docx"
            ))
        return sections