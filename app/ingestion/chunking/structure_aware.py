from typing import List, Optional
from uuid import UUID
from loguru import logger

from .base import BaseChunker
from .models import ParsedSection, TextChunk
from .recursive import RecursiveTokenSplitter
from .utils import generate_deterministic_id, validate_chunk

class StructureAwareChunker(BaseChunker):
    def __init__(
        self, 
        chunk_size_tokens: int = 600, 
        chunk_overlap_tokens: int = 100,
        tokenizer_name: str = "cl100k_base",
        include_contextual_prefix: bool = True
    ):
        self.chunk_size = chunk_size_tokens
        self.chunk_overlap = chunk_overlap_tokens
        self.include_contextual_prefix = include_contextual_prefix
        self.recursive_splitter = RecursiveTokenSplitter(
            chunk_size_tokens, 
            chunk_overlap_tokens, 
            tokenizer_name
        )
        self.encoder = self.recursive_splitter.encoder

    def _build_contextual_prefix(self, section: ParsedSection) -> str:
        if not self.include_contextual_prefix:
            return ""
        parts = []
        if section.chapter: parts.append(f"Chapter: {section.chapter}")
        if section.section: parts.append(f"Section: {section.section}")
        if section.subsection: parts.append(f"Subsection: {section.subsection}")
        return "\n".join(parts) + "\n\n" if parts else ""

    def _split_table(self, table_text: str) -> List[str]:
        """Intelligently splits oversized tables while keeping headers."""
        lines = table_text.strip().split("\n")
        if len(lines) < 2:
            return self.recursive_splitter.split_text(table_text)
            
        header = lines[0]
        # Assuming markdown table format with separator on line 2
        separator = lines[1] if len(lines) > 1 and "---" in lines[1] else ""
        header_block = f"{header}\n{separator}" if separator else header
        
        chunks = []
        current_chunk = header_block
        
        for line in lines[2 if separator else 1:]:
            prospective_chunk = f"{current_chunk}\n{line}"
            if len(self.encoder.encode(prospective_chunk)) <= self.chunk_size:
                current_chunk = prospective_chunk
            else:
                if current_chunk != header_block:
                    chunks.append(current_chunk)
                current_chunk = f"{header_block}\n{line}" if header_block else line
                
        if current_chunk and current_chunk != header_block:
            chunks.append(current_chunk)
            
        return chunks if chunks else [table_text]

    def chunk(
        self,
        sections: List[ParsedSection],
        document_id: UUID,
        course_id: str,
        filename: str
    ) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        chunk_idx = 0
        
        for section in sections:
            content = section.content.strip()
            if not content:
                continue
                
            prefix = self._build_contextual_prefix(section)
            full_content = f"{prefix}{content}"
            
            token_count = len(self.encoder.encode(full_content))
            
            text_blocks: List[str] = []
            
            if token_count <= self.chunk_size:
                text_blocks.append(full_content)
            else:
                # Strip prefix for structural splitting, re-add later
                if section.content_type == "table":
                    text_blocks = self._split_table(content)
                else:
                    text_blocks = self.recursive_splitter.split_text(content)
                    
            for block in text_blocks:
                final_content = f"{prefix}{block}" if prefix else block
                
                is_valid, msg = validate_chunk(final_content, self.chunk_size, self.encoder)
                if not is_valid:
                    if "Empty" in msg:
                        continue
                    logger.warning(f"Chunk validation failed for doc {document_id} idx {chunk_idx}: {msg}")
                    # Force split if still too large after recursive fallback
                    if "Exceeds" in msg:
                        final_content = final_content[:self.chunk_size * 4] # Hard char limit fallback
                
                chunk_id = generate_deterministic_id(document_id, chunk_idx, final_content)
                
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    course_id=course_id,
                    filename=filename,
                    content=final_content,
                    page=section.page,
                    chapter=section.chapter,
                    section=section.section,
                    subsection=section.subsection,
                    chunk_index=chunk_idx,
                    content_type=section.content_type,
                    metadata=section.metadata
                ))
                chunk_idx += 1
                
        return chunks