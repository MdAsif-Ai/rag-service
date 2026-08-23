from app.ingestion.normalizer import DocumentNormalizer
from app.ingestion.chunking.models import ParsedSection

def test_whitespace_normalization():
    norm = DocumentNormalizer()
    section = ParsedSection(content="Hello    \n\n\n   World \r\n")
    result = norm.normalize([section])
    assert result[0].content == "Hello\n\nWorld"

def test_unicode_normalization():
    norm = DocumentNormalizer()
    # Full-width comma to standard comma (NFKC)
    section = ParsedSection(content="Hello，World")
    result = norm.normalize([section])
    # NFKC converts ， to , but does not add a space
    assert result[0].content == "Hello,World"