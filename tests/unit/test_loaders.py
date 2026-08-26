import pytest
from unittest.mock import MagicMock, patch

from app.ingestion.loaders import get_loader, DoclingLoader
from app.ingestion.loaders.base import ParsedSection
from app.core.exceptions import UnsupportedFileException

# This fixture automatically mocks the DocumentConverter for ALL tests in this file
# It prevents Docling from downloading 1GB+ of models during tests
@pytest.fixture(autouse=True)
def mock_docling_converter():
    with patch("app.ingestion.loaders.docling_loader.DocumentConverter") as mock_converter_class:
        mock_instance = MagicMock()
        mock_converter_class.return_value = mock_instance
        
        # Set the class variable so __init__ skips real initialization
        DoclingLoader._converter = mock_instance
        yield mock_instance
        
        # Cleanup after tests
        DoclingLoader._converter = None

def test_unsupported_file_type():
    with pytest.raises(UnsupportedFileException):
        get_loader("exe")

def test_get_loader_returns_docling(mock_docling_converter):
    # Verify all formats route to DoclingLoader without triggering real initialization
    loader_pdf = get_loader("pdf")
    assert isinstance(loader_pdf, DoclingLoader)
    
    loader_docx = get_loader("docx")
    assert isinstance(loader_docx, DoclingLoader)

def test_docling_loader_parses_markdown(mock_docling_converter):
    # Setup mock for the conversion result
    mock_result = MagicMock()
    mock_document = MagicMock()
    # Simulate Docling exporting a Markdown string
    mock_document.export_to_markdown.return_value = "# Chapter 1\n\nSome text here\n\n## Section 2\n\nMore text"
    mock_result.document = mock_document
    mock_docling_converter.convert.return_value = mock_result
    
    loader = DoclingLoader()
    sections = loader._safe_load("fake_path.pdf")
    
    assert len(sections) == 2
    assert sections[0].section == "Chapter 1"
    assert sections[0].content == "Some text here"
    assert sections[0].source_type == "docling"
    
    assert sections[1].section == "Section 2"
    assert sections[1].content == "More text"