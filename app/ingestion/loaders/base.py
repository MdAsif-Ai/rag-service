from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from app.core.exceptions import DocumentProcessingException


class ParsedSection(BaseModel):
    """Normalized internal representation of a document chunk before embedding."""
    content: str
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class DocumentLoader(ABC):
    """
    Common interface for all document loaders.
    Implementations must handle parsing without crashing the worker on malformed input.
    """

    @abstractmethod
    def load(self, file_path: str) -> List[ParsedSection]:
        """Parses the document and returns a list of normalized sections."""
        pass

    def _safe_load(self, file_path: str) -> List[ParsedSection]:
        """Wrapper to catch library-specific exceptions and normalize them."""
        try:
            return self.load(file_path)
        except DocumentProcessingException:
            raise
        except Exception as e:
            raise DocumentProcessingException(
                f"Failed to parse file {file_path} with {self.__class__.__name__}",
                detail=str(e)
            )