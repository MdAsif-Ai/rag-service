from abc import ABC, abstractmethod
from typing import List
from uuid import UUID
from .models import ParsedSection, TextChunk

class BaseChunker(ABC):
    @abstractmethod
    def chunk(
        self,
        sections: List[ParsedSection],
        document_id: UUID,
        course_id: str,
        filename: str
    ) -> List[TextChunk]:
        pass