import os
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()
if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import EMBEDDING_MODEL_NAME


class LlamaIndexEmbedder:
    """Embedding Generator using LlamaIndex HuggingFaceEmbedding abstraction (intfloat/multilingual-e5-base)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading LlamaIndex HuggingFaceEmbedding: {model_name}...")
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self.model = HuggingFaceEmbedding(
                model_name=model_name,
                query_instruction="query: ",
                text_instruction="passage: ",
                normalize=True,
            )
            self._use_llamaindex = True
        except Exception as exc:
            print(f"LlamaIndex embedding loading note ({exc}), falling back to SentenceTransformer...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._use_llamaindex = False

    def get_document_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed document passages with L2 normalization."""
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        if self._use_llamaindex:
            passages = [f"passage: {t}" for t in texts]
            raw_embs = self.model.get_text_embedding_batch(passages, show_progress=True)
            arr = np.array(raw_embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        else:
            passages = [f"passage: {t}" for t in texts]
            embeddings = self.model.encode(
                passages,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            return np.array(embeddings, dtype=np.float32)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Embed search query with L2 normalization."""
        if self._use_llamaindex:
            formatted_query = f"query: {query}"
            raw_emb = self.model.get_query_embedding(formatted_query)
            arr = np.array([raw_emb], dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        else:
            formatted_query = f"query: {query}"
            embedding = self.model.encode(
                [formatted_query],
                normalize_embeddings=True,
            )
            return np.array(embedding, dtype=np.float32)


# Backward compatibility alias
MultilingualE5Embedder = LlamaIndexEmbedder
