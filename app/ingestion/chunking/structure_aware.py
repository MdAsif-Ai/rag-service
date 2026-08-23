import re
from typing import List, Optional
from uuid import UUID

from app.core.config import get_settings
from app.ingestion.loaders.base import ParsedSection
from .base import BaseChunker, TextChunk


class StructureAwareChunker(BaseChunker):
    """
    Chunking strategy that respects document structure (sections, pages).
    Falls back to recursive sentence/word splitting when structural blocks
    exceed the target chunk size.
    """

    def __init__(self, target_chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Args:
            target_chunk_size: Target size in characters (approx. 400-800 tokens).
            chunk_overlap: Overlap in characters to preserve context across boundaries.
        """
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        sections: List[ParsedSection],
        document_id: UUID,
        course_id: str,
        filename: str,
        chapter: Optional[str] = None
    ) -> List[TextChunk]:
        """
        Processes sections into chunks. If a section is smaller than target_chunk_size,
        it is kept intact. Otherwise, it is recursively split.
        """
        settings = get_settings()
        # Fallback to settings if not explicitly provided during instantiation
        target_size = self.target_chunk_size or (settings.CHUNK_SIZE * 4) # approx 4 chars per token
        overlap = self.chunk_overlap or (settings.CHUNK_OVERLAP * 4)

        chunks: List[TextChunk] = []
        chunk_idx = 0

        for section in sections:
            section_text = section.content.strip()
            if not section_text:
                continue

            # Prepend section title to content to avoid separating headings from text
            if section.section and not section_text.startswith(section.section):
                section_text = f"{section.section}\n{section_text}"

            # If the section is already within limits, keep it as a single chunk
            if len(section_text) <= target_size:
                chunks.append(self._create_chunk(
                    content=section_text,
                    section=section,
                    document_id=document_id,
                    course_id=course_id,
                    filename=filename,
                    chapter=chapter,
                    chunk_idx=chunk_idx
                ))
                chunk_idx += 1
                continue

            # Otherwise, recursively split the section
            split_texts = self._recursive_split(section_text, target_size, overlap)
            
            for split_text in split_texts:
                chunks.append(self._create_chunk(
                    content=split_text,
                    section=section,
                    document_id=document_id,
                    course_id=course_id,
                    filename=filename,
                    chapter=chapter,
                    chunk_idx=chunk_idx
                ))
                chunk_idx += 1

        return chunks

    def _create_chunk(
        self,
        content: str,
        section: ParsedSection,
        document_id: UUID,
        course_id: str,
        filename: str,
        chapter: Optional[str],
        chunk_idx: int
    ) -> TextChunk:
        return TextChunk(
            chunk_id=self.generate_deterministic_id(document_id, chunk_idx),
            document_id=document_id,
            course_id=course_id,
            filename=filename,
            content=content,
            page=section.page,
            section=section.section,
            chapter=chapter,
            chunk_index=chunk_idx,
            metadata=section.metadata
        )

    def _recursive_split(self, text: str, target_size: int, overlap: int) -> List[str]:
        """
        Recursively splits text by paragraphs, then sentences, then words.
        """
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",    # Exclamatory sentences
            "? ",    # Questions
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters (fallback)
        ]
        
        return self._split_text(text, separators, target_size, overlap)

    def _split_text(self, text: str, separators: List[str], target_size: int, overlap: int) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = separators[:-1]

        # Find the first separator that exists in the text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i+1:]
                break

        # Split the text
        if separator:
            splits = text.split(separator)
        else:
            # Character fallback
            splits = [text[i:i+target_size] for i in range(0, len(text), target_size)]

        # Merge splits respecting target_size and overlap
        current_chunk = ""
        for split in splits:
            split = split.strip()
            if not split:
                continue
                
            prospective_chunk = current_chunk + separator + split if current_chunk else split
            
            if len(prospective_chunk) <= target_size:
                current_chunk = prospective_chunk
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    # Handle overlap by taking the tail of the current chunk
                    if overlap > 0:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + separator + split
                    else:
                        current_chunk = split
                else:
                    # If the single split is still larger than target_size, recurse with a finer separator
                    if new_separators:
                        final_chunks.extend(self._split_text(split, new_separators, target_size, overlap))
                    else:
                        final_chunks.append(split) # Absolute fallback
        
        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks