from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import get_token_encoder

class RecursiveTokenSplitter:
    """Wrapper around LangChain's RecursiveCharacterTextSplitter for token-aware splitting."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int, tokenizer_name: str = "cl100k_base"):
        self.encoder = get_token_encoder(tokenizer_name)
        
        # Prioritize structural boundaries
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",    # Exclamatory
            "? ",    # Questions
            ", ",    # Clauses
            " ",     # Words
            ""       # Character fallback
        ]
        
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=tokenizer_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)