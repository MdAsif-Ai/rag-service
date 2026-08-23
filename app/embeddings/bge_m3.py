import functools
import threading
from typing import List, Dict, Any, Optional

from loguru import logger
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingException


class EmbeddingResult(BaseModel):
    """Typed return model for embedding results."""
    dense_vector: List[float]
    sparse_vector: Dict[int, float]


class BGEEmbeddingService:
    """
    Production embedding service using BGE-M3.
    Supports dense and sparse(lexical) vector generation.
    The model is loaded once per worker process to maximize performance.
    """
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(BGEEmbeddingService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, settings: Settings):
        if self._initialized:
            return
            
        with BGEEmbeddingService._lock:
            if self._initialized:
                return
                
            self.settings = settings
            self.model = None
            self._initialize_model()
            self._initialized = True

    def _initialize_model(self) -> None:
        """Lazy initialization of the BGE-M3 model."""
        try:
            from FlagEmbedding import BGEM3FlagModel
            
            model_name = self.settings.EMBEDDING_MODEL
            device = self.settings.EMBEDDING_DEVICE
            
            use_fp16 = (device == "cuda")
            
            logger.info(f"Initializing BGE-M3 model '{model_name}' on device '{device}'...")
            
            self.model = BGEM3FlagModel(
                model_name,
                use_fp16=use_fp16,
                device=device
            )
            
            logger.info("BGE-M3 model loaded successfully.")
            
        except ImportError:
            raise EmbeddingException(
                "FlagEmbedding library not installed. Please install it to use BGE-M3.",
                detail="Run: pip install FlagEmbedding torch"
            )
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            
            if device == "cuda":
                logger.warning("GPU load failed. Falling back to CPU.")
                try:
                    from FlagEmbedding import BGEM3FlagModel
                    self.model = BGEM3FlagModel(model_name, use_fp16=False, device="cpu")
                    self.settings.EMBEDDING_DEVICE = "cpu"
                    logger.info("BGE-M3 model loaded successfully on CPU fallback.")
                except Exception as fallback_e:
                    raise EmbeddingException("Failed to load BGE-M3 on both GPU and CPU.", detail=str(fallback_e))
            else:
                raise EmbeddingException("Failed to initialize BGE-M3 model.", detail=str(e))

    def embed_documents(self, texts: List[str]) -> List[EmbeddingResult]:
        if not texts:
            return []

        if not self.model:
            raise EmbeddingException("Embedding model is not initialized.")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.settings.RERANK_BATCH_SIZE,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            results = []
            dense_vecs = embeddings["dense_vecs"]
            lexical_weights = embeddings["lexical_weights"]
            
            for i in range(len(texts)):
                # Use list() instead of .tolist() to support both NumPy arrays and plain lists
                dense = list(dense_vecs[i])
                
                sparse = {}
                if i in lexical_weights:
                    sparse = {int(k): float(v) for k, v in lexical_weights[i].items()}
                
                results.append(EmbeddingResult(
                    dense_vector=dense,
                    sparse_vector=sparse
                ))
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {e}")
            raise EmbeddingException("Failed to generate document embeddings.", detail=str(e))

    def embed_query(self, text: str) -> EmbeddingResult:
        if not text or not text.strip():
            raise EmbeddingException("Cannot embed an empty query.")

        if not self.model:
            raise EmbeddingException("Embedding model is not initialized.")

        try:
            embeddings = self.model.encode(
                [text],
                batch_size=1,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            # Use list() instead of .tolist() to support both NumPy arrays and plain lists
            dense = list(embeddings["dense_vecs"][0])
            lexical_weights = embeddings["lexical_weights"]
            
            sparse = {}
            if 0 in lexical_weights:
                sparse = {int(k): float(v) for k, v in lexical_weights[0].items()}
                
            return EmbeddingResult(
                dense_vector=dense,
                sparse_vector=sparse
            )
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise EmbeddingException("Failed to generate query embedding.", detail=str(e))


@functools.lru_cache(maxsize=1)
def get_embedding_service() -> BGEEmbeddingService:
    settings = get_settings()
    return BGEEmbeddingService(settings)