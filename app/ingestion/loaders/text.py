from typing import List
from app.ingestion.loaders.base import DocumentLoader, ParsedSection

class TextLoader(DocumentLoader):
    def load(self, file_path: str) -> List[ParsedSection]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [ParsedSection(content=content, source_type="txt")]