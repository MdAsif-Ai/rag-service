import pytest
from uuid import uuid4
from app.ingestion.chunking import StructureAwareChunker, ParsedSection

@pytest.fixture
def chunker():
    return StructureAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)

def test_deterministic_chunk_ids(chunker):
    doc_id = uuid4()
    sections = [ParsedSection(content="A" * 50, page=1, source_type="pdf")]
    
    chunks1 = chunker.chunk(sections, doc_id, "c1", "f.pdf")
    chunks2 = chunker.chunk(sections, doc_id, "c1", "f.pdf")
    
    assert len(chunks1) == 1
    assert chunks1[0].chunk_id == chunks2[0].chunk_id

def test_content_change_changes_id(chunker):
    doc_id = uuid4()
    sections1 = [ParsedSection(content="A" * 50, page=1)]
    sections2 = [ParsedSection(content="B" * 50, page=1)]
    
    chunks1 = chunker.chunk(sections1, doc_id, "c1", "f.pdf")
    chunks2 = chunker.chunk(sections2, doc_id, "c1", "f.pdf")
    
    assert chunks1[0].chunk_id != chunks2[0].chunk_id

def test_metadata_preservation(chunker):
    sections = [ParsedSection(content="Test content", page=5, section="Intro", source_type="docx")]
    chunks = chunker.chunk(sections, uuid4(), "c1", "f.pdf")
    
    assert chunks[0].page == 5
    assert chunks[0].section == "Intro"
    assert chunks[0].source_type == "docx"

def test_small_section_not_split(chunker):
    # 10 tokens is well under the 100 token limit
    sections = [ParsedSection(content="word " * 10, page=1)]
    chunks = chunker.chunk(sections, uuid4(), "c1", "f.pdf")
    assert len(chunks) == 1

def test_large_section_is_split(chunker):
    # 500 tokens will be split into multiple chunks
    sections = [ParsedSection(content="word " * 500, page=1)]
    chunks = chunker.chunk(sections, uuid4(), "c1", "f.pdf")
    assert len(chunks) > 1