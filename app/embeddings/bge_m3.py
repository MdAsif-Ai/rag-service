import functools
from typing import Any, Dict, List

from loguru import logger

from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingException


class BGEEmbeddingService:
    """
    Production embedding service using BGE-M3.
    Supports dense and sparse(lexical) vector generation.
    The model is loaded once per worker process to maximize performance.
    """

    def __init__(self, settings: Settings):
        try:
            # Imported here to prevent loading PyTorch globally at module import time
            from FlagEmbedding import BGEM3FlagModel

            self.settings = settings
            model_name = settings.EMBEDDING_MODEL
            device = settings.EMBEDDING_DEVICE
            
            # use_fp16 improves performance on GPU. Fallback to False on CPU.
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
            # Handle CUDA out of memory or model download failures
            logger.error(f"Failed to load BGE-M3 model: {e}")
            
            # CPU Fallback for development if GPU fails
            if device == "cuda":
                logger.warning("GPU load failed. Falling back to CPU.")
                try:
                    from FlagEmbedding import BGEM3FlagModel
                    self.model = BGEM3FlagModel(model_name, use_fp16=False, device="cpu")
                    self.settings.EMBEDDING_DEVICE = "cpu" # Update active setting
                    logger.info("BGE-M3 model loaded successfully on CPU fallback.")
                except Exception as fallback_e:
                    raise EmbeddingException("Failed to load BGE-M3 on both GPU and CPU.", detail=str(fallback_e))
            else:
                raise EmbeddingException("Failed to initialize BGE-M3 model.", detail=str(e))

    def embed_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Embeds a list of document chunks.
        Returns a list of dictionaries containing 'dense' and 'sparse' vectors.
        """
        if not texts:
            return []

        try:
            # BGE-M3 encode returns a dict with 'dense_vecs' and 'lexical_weights'
            # normalize_embeddings=True is crucial for Cosine similarity in Qdrant
            embeddings = self.model.encode(
                texts,
                batch_size=12, # Adjust based on GPU VRAM (12 is safe for 8GB-24GB)
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            results = []
            dense_vecs = embeddings["dense_vecs"]
            lexical_weights = embeddings["lexical_weights"]
            
            for i in range(len(texts)):
                # Convert numpy array to list for JSON serialization/Qdrant upsert
                dense = dense_vecs[i].tolist()
                
                # BGE-M3 returns sparse weights as {token_id: weight}
                # Qdrant expects {int: float}
                sparse = {}
                if i in lexical_weights:
                    sparse = {int(k): float(v) for k, v in lexical_weights[i].items()}
                
                results.append({
                    "dense_vector": dense,
                    "sparse_vector": sparse
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {e}")
            raise EmbeddingException("Failed to generate document embeddings.", detail=str(e))

    def embed_query(self, text: str) -> Dict[str, Any]:
        """
        Embeds a single search query.
        Returns a dictionary containing 'dense_vector' and 'sparse_vector'.
        """
        try:
            embeddings = self.model.encode(
                [text],
                batch_size=1,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            dense = embeddings["dense_vecs"][0].tolist()
            lexical_weights = embeddings["lexical_weights"]
            
            sparse = {}
            if 0 in lexical_weights:
                sparse = {int(k): float(v) for k, v in lexical_weights[0].items()}
                
            return {
                "dense_vector": dense,
                "sparse_vector": sparse
            }
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise EmbeddingException("Failed to generate query embedding.", detail=str(e))


@functools.lru_cache(maxsize=1)
def get_embedding_service() -> BGEEmbeddingService:
    """
    Factory function to get a cached singleton instance of the BGEEmbeddingService.
    This ensures the model is loaded exactly once per Uvicorn/Celery worker process,
    rather than on every API request or background job.
    """
    settings = get_settings()
    return BGEEmbeddingService(settings)