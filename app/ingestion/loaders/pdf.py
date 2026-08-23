from typing import List
import fitz  # PyMuPDF
from app.ingestion.loaders.base import DocumentLoader, ParsedSection

class PDFLoader(DocumentLoader):
    def load(self, file_path: str) -> List[ParsedSection]:
        sections = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    sections.append(ParsedSection(
                        content=text.strip(),
                        page=page_num,
                        source_type="pdf"
                    ))
        return sections