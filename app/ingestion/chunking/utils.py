import hashlib
from uuid import UUID
from typing import List, Tuple
import tiktoken
from loguru import logger

def get_token_encoder(model_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Initializes a tiktoken encoder."""
    try:
        return tiktoken.get_encoding(model_name)
    except Exception:
        logger.warning(f"Unknown tiktoken model {model_name}. Falling back to cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")

def generate_deterministic_id(document_id: UUID, chunk_index: int, content: str) -> str:
    """Generates a deterministic SHA256 hash for the chunk."""
    hash_input = f"{document_id}:{chunk_index}:{content}".encode("utf-8")
    return hashlib.sha256(hash_input).hexdigest()

def validate_chunk(chunk_content: str, max_tokens: int, encoder: tiktoken.Encoding) -> Tuple[bool, str]:
    """Validates chunk content and size."""
    if not chunk_content or not chunk_content.strip():
        return False, "Empty content"
    token_count = len(encoder.encode(chunk_content))
    if token_count > max_tokens:
        return False, f"Exceeds max tokens ({token_count} > {max_tokens})"
    return True, "Valid"