from __future__ import annotations

import traceback

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import DocumentProcessingException


class ParsedSection(BaseModel):
    """
    Normalized representation of content extracted by a loader.
    """

    content: str

    page: Optional[int] = None
    section: Optional[str] = None
    chapter: Optional[str] = None

    content_type: str = "text"
    source_type: str = "unknown"

    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


class DocumentLoader(ABC):

    @abstractmethod
    def load(self, file_path: str) -> List[ParsedSection]:
        """
        Load and extract content from a document.
        """
        raise NotImplementedError

    def _safe_load(self, file_path: str) -> List[ParsedSection]:
        try:
            sections = self.load(file_path)

            if not sections:
                raise DocumentProcessingException(
                    "No content extracted from document."
                )

            valid_sections = [
                section
                for section in sections
                if section.content and section.content.strip()
            ]

            if not valid_sections:
                raise DocumentProcessingException(
                    "Document was processed but contained no usable text."
                )

            return valid_sections

        except DocumentProcessingException:
            raise

        except Exception as exc:
            raise DocumentProcessingException(
                f"Failed to parse file '{file_path}' "
                f"with {self.__class__.__name__}",
                detail=traceback.format_exc(),
            ) from exc