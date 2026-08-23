from typing import List, Optional
from uuid import UUID
from loguru import logger

from app.ingestion.loaders.base import ParsedSection
from .base import BaseChunker
from .models import TextChunk
from .recursive import RecursiveTokenSplitter
from .utils import generate_deterministic_id, validate_chunk, get_token_encoder

class StructureAwareChunker(BaseChunker):
    def __init__(
        self, 
        chunk_size_tokens: int = 600, 
        chunk_overlap_tokens: int = 100,
        tokenizer_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size_tokens
        self.chunk_overlap = chunk_overlap_tokens
        self.encoder = get_token_encoder(tokenizer_name)
        self.recursive_splitter = RecursiveTokenSplitter(
            chunk_size_tokens, 
            chunk_overlap_tokens, 
            tokenizer_name
        )

    def chunk(
        self,
        sections: List[ParsedSection],
        document_id: UUID,
        course_id: str,
        filename: str,
        chapter: Optional[str] = None
    ) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        chunk_idx = 0
        
        for section in sections:
            content = section.content.strip()
            if not content:
                continue
                
            token_count = len(self.encoder.encode(content))
            
            text_blocks: List[str] = []
            
            # If section fits in chunk size, keep it whole to preserve structure
            if token_count <= self.chunk_size:
                text_blocks.append(content)
            else:
                # Fallback to recursive splitting for oversized blocks
                text_blocks = self.recursive_splitter.split_text(content)
                
            for block in text_blocks:
                if not block.strip():
                    continue
                    
                is_valid, msg = validate_chunk(block, self.chunk_size, self.encoder)
                if not is_valid and "Empty" not in msg:
                    logger.warning(f"Chunk validation failed for doc {document_id} idx {chunk_idx}: {msg}")
                
                chunk_id = generate_deterministic_id(document_id, chunk_idx, block)
                
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    course_id=course_id,
                    filename=filename,
                    content=block,
                    page=section.page,
                    chapter=chapter if chapter else section.chapter,
                    section=section.section,
                    chunk_index=chunk_idx,
                    content_type=section.content_type,
                    source_type=section.source_type,
                    metadata=section.metadata
                ))
                chunk_idx += 1
                
        return chunks