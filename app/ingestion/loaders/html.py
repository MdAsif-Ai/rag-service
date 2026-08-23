from typing import List
from bs4 import BeautifulSoup
from app.ingestion.loaders.base import DocumentLoader, ParsedSection

class HTMLLoader(DocumentLoader):
    def load(self, file_path: str) -> List[ParsedSection]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")

        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        sections = []
        current_section = "Default"
        
        for elem in soup.find_all(["h1", "h2", "h3", "p", "li", "table"]):
            text = elem.get_text(separator=" ", strip=True)
            if not text:
                continue
            if elem.name in ["h1", "h2", "h3"]:
                current_section = text
            else:
                sections.append(ParsedSection(
                    content=text,
                    section=current_section,
                    source_type="html",
                    content_type="table" if elem.name == "table" else "text"
                ))
        
        if not sections:
            text = soup.get_text(separator="\n", strip=True)
            if text:
                sections.append(ParsedSection(content=text, source_type="html"))
                
        return sections