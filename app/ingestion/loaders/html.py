from typing import List
from bs4 import BeautifulSoup

from app.core.exceptions import DocumentProcessingException
from .base import DocumentLoader, ParsedSection


class HTMLLoader(DocumentLoader):
    """Loads HTML files, extracting text and basic structure."""

    def load(self, file_path: str) -> List[ParsedSection]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        sections: List[ParsedSection] = []
        current_section = "Default"
        
        # Attempt to split by header tags
        for elem in soup.find_all(["h1", "h2", "h3", "p", "li", "table"]):
            text = elem.get_text(separator=" ", strip=True)
            if not text:
                continue
                
            if elem.name in ["h1", "h2", "h3"]:
                current_section = text
            else:
                sections.append(
                    ParsedSection(
                        content=text,
                        section=current_section
                    )
                )

        # Fallback if no structure found
        if not sections:
            text = soup.get_text(separator="\n", strip=True)
            if text:
                sections.append(ParsedSection(content=text))

        if not sections:
            raise DocumentProcessingException("HTML contained no extractable text.")
            
        return sections