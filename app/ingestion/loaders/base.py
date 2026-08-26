import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import DocumentProcessingException


class ParsedSection(BaseModel):
    """Normalized internal representation of a document block from a loader."""
    content: str
    page: Optional[int] = None
    section: Optional[str] = None
    chapter: Optional[str] = None
    content_type: str = "text"
    source_type: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[ParsedSection]:
        pass

    def _safe_load(self, file_path: str) -> List[ParsedSection]:
        try:
            sections = self.load(file_path)
            if not sections:
                raise DocumentProcessingException("No content extracted from document.")
            return sections
        except DocumentProcessingException:
            raise
        except Exception as e:
            raise DocumentProcessingException(
                f"Failed to parse file {file_path} with {self.__class__.__name__}",
                detail=traceback.format_exc()
            )