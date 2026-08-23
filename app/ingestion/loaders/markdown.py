import re
from typing import List

from app.core.exceptions import DocumentProcessingException
from .base import DocumentLoader, ParsedSection


class MarkdownLoader(DocumentLoader):
    """Loads Markdown files, splitting by header sections."""

    def load(self, file_path: str) -> List[ParsedSection]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            raise DocumentProcessingException("Markdown file is empty.")

        # Split by markdown headers (e.g., # Header, ## Header)
        lines = content.split("\n")
        sections: List[ParsedSection] = []
        
        current_section = "Default"
        current_text = []

        for line in lines:
            if not line.strip():
                continue
                
            header_match = re.match(r'^(#{1,6})\s+(.*)', line)
            if header_match:
                if current_text:
                    sections.append(
                        ParsedSection(
                            content="\n".join(current_text),
                            section=current_section
                        )
                    )
                    current_text = []
                current_section = header_match.group(2).strip()
            else:
                current_text.append(line)

        if current_text:
            sections.append(
                ParsedSection(
                    content="\n".join(current_text),
                    section=current_section
                )
            )

        return sections