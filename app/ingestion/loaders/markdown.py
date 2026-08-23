import re
from typing import List
from app.ingestion.loaders.base import DocumentLoader, ParsedSection

class MarkdownLoader(DocumentLoader):
    def load(self, file_path: str) -> List[ParsedSection]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        lines = content.split("\n")
        sections = []
        current_section = "Default"
        current_text = []

        for line in lines:
            if not line.strip():
                continue
            header_match = re.match(r'^(#{1,6})\s+(.*)', line)
            if header_match:
                if current_text:
                    sections.append(ParsedSection(
                        content="\n".join(current_text),
                        section=current_section,
                        source_type="md"
                    ))
                    current_text = []
                current_section = header_match.group(2).strip()
            else:
                current_text.append(line)

        if current_text:
            sections.append(ParsedSection(
                content="\n".join(current_text),
                section=current_section,
                source_type="md"
            ))
        return sections