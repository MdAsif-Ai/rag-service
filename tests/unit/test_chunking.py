from uuid import uuid4
from app.ingestion.chunking import StructureAwareChunker, ParsedSection

def test_deterministic_chunk_ids():
    chunker = StructureAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)
    doc_id = uuid4()
    sections = [ParsedSection(content="A" * 50, page=1)]
    
    chunks1 = chunker.chunk(sections, doc_id, "c1", "f.pdf")
    chunks2 = chunker.chunk(sections, doc_id, "c1", "f.pdf")
    
    assert len(chunks1) == 1
    assert chunks1[0].chunk_id == chunks2[0].chunk_id

def test_metadata_preservation():
    chunker = StructureAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)
    sections = [ParsedSection(content="Test content", page=5, section="Intro")]
    chunks = chunker.chunk(sections, uuid4(), "c1", "f.pdf")
    
    assert chunks[0].page == 5
    assert chunks[0].section == "Intro"