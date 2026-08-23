from typing import List

from app.core.exceptions import DocumentProcessingException
from .base import DocumentLoader, ParsedSection


class TextLoader(DocumentLoader):
    """Loads plain text files efficiently line by line."""

    def load(self, file_path: str) -> List[ParsedSection]:
        sections: List[ParsedSection] = []
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        if not content.strip():
            raise DocumentProcessingException("Text file is empty.")
            
        sections.append(ParsedSection(content=content))
        return sections