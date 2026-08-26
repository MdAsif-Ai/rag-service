from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import traceback

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import DocumentProcessingException


class ParsedSection(BaseModel):
    """
    Normalized internal representation of content extracted from a document.

    Every loader must return one or more ParsedSection objects.
    """

    content: str

    page: Optional[int] = None
    section: Optional[str] = None
    chapter: Optional[str] = None

    content_type: str = "text"
    source_type: str = "unknown"

    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DocumentLoader(ABC):
    """
    Base interface for all document loaders.
    """

    @abstractmethod
    def load(self, file_path: str) -> List[ParsedSection]:
        """
        Extract content from a file.

        Args:
            file_path: Path to the source document.

        Returns:
            List of ParsedSection objects.
        """
        raise NotImplementedError

    def _safe_load(self, file_path: str) -> List[ParsedSection]:
        """
        Execute the loader with consistent error handling.
        """
        try:
            sections = self.load(file_path)

            if not sections:
                raise DocumentProcessingException(
                    "No content extracted from document."
                )

            return sections

        except DocumentProcessingException:
            raise

        except Exception as exc:
            raise DocumentProcessingException(
                f"Failed to parse file '{file_path}' "
                f"with {self.__class__.__name__}",
                detail=traceback.format_exc(),
            ) from exc