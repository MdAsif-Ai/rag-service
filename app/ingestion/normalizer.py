import re
import unicodedata
from typing import List

from app.ingestion.loaders.base import ParsedSection


class DocumentNormalizer:
    """
    Cleans and normalizes text extracted from documents.
    Designed to be conservative to preserve technical, legal, and educational meaning.
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""

        # 1. Unicode Normalization (NFKC)
        # Normalizes compatibility characters (e.g., full-width to half-width, ligatures)
        # without destroying the underlying meaning of the text.
        text = unicodedata.normalize('NFKC', text)

        # 2. Remove common parser artifacts and invisible characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # BOM / Zero-width no-break space
        text = text.replace('\xad', '')    # Soft hyphen (often breaks words unnecessarily)
        text = text.replace('\xa0', ' ')   # Non-breaking space to standard space

        # 3. Standardize whitespace
        # Standardize Windows/old Mac newlines to Unix
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Replace multiple spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove leading/trailing whitespace on every line
        text = re.sub(r' *\n *', '\n', text)

        # 4. Remove useless repeated blank lines
        # Collapses 3+ newlines down to 2 (preserving single paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def normalize(self, sections: List[ParsedSection]) -> List[ParsedSection]:
        """
        Normalizes the content of a list of ParsedSection objects.
        Preserves all structural metadata (page, section, etc.).
        """
        normalized_sections: List[ParsedSection] = []

        for section in sections:
            cleaned_content = self._normalize_text(section.content)
            
            # Only keep sections that actually have text after cleaning
            if cleaned_content:
                normalized_sections.append(
                    ParsedSection(
                        content=cleaned_content,
                        page=section.page,
                        section=section.section,
                        metadata=section.metadata
                    )
                )

        return normalized_sections