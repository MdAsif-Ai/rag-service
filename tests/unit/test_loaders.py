import pytest
import os
from unittest.mock import MagicMock, patch

from app.ingestion.loaders import get_loader
from app.ingestion.loaders.base import ParsedSection
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.html import HTMLLoader
from app.ingestion.loaders.text import TextLoader
from app.core.exceptions import UnsupportedFileException

def test_unsupported_file_type():
    with pytest.raises(UnsupportedFileException):
        get_loader("exe")

def test_text_loader(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello World")
    
    loader = TextLoader()
    sections = loader._safe_load(str(file_path))
    
    assert len(sections) == 1
    assert sections[0].content == "Hello World"
    assert sections[0].source_type == "txt"

def test_markdown_loader_preserves_headings(tmp_path):
    file_path = tmp_path / "test.md"
    file_path.write_text("# Chapter 1\n\nSome text here\n\n## Section 2\n\nMore text")
    
    loader = MarkdownLoader()
    sections = loader._safe_load(str(file_path))
    
    assert len(sections) == 2
    assert sections[0].section == "Chapter 1"
    assert sections[0].content == "Some text here"
    assert sections[1].section == "Section 2"
    assert sections[1].source_type == "md"

def test_html_loader_preserves_structure(tmp_path):
    file_path = tmp_path / "test.html"
    file_path.write_text("<html><body><h1>Title</h1><p>Content</p></body></html>")
    
    loader = HTMLLoader()
    sections = loader._safe_load(str(file_path))
    
    assert len(sections) == 1
    assert sections[0].section == "Title"
    assert sections[0].content == "Content"
    assert sections[0].source_type == "html"

@patch("app.ingestion.loaders.docx.Document")
def test_docx_loader_metadata_mapping(mock_docx_class):
    # Mock the python-docx Document object
    mock_doc = MagicMock()
    mock_para1 = MagicMock(text="Heading 1", style=MagicMock(name="Heading 1"))
    mock_para2 = MagicMock(text="Paragraph text", style=MagicMock(name="Normal"))
    mock_doc.paragraphs = [mock_para1, mock_para2]
    mock_docx_class.return_value = mock_doc
    
    loader = get_loader("docx")
    sections = loader._safe_load("fake_path.docx")
    
    assert len(sections) == 1
    assert sections[0].section == "Heading 1"
    assert sections[0].content == "Paragraph text"
    assert sections[0].source_type == "docx"