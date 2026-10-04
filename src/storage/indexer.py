import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load environment variables (.env)
load_dotenv(BASE_DIR / ".env")
if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

from src.core.config import EMBEDDING_MODEL_NAME, EMBED_HARD_MAX_TOKENS, EMBED_TARGET_TOKENS
from src.core.utils import remove_vietnamese_accents


class LegalTokenizer:
    """Tokenizer bọc mô hình thực tế để kiểm soát độ dài embedding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._tokenizer = None
        self._is_rust_tokenizer = False

        # Ưu tiên 1: Dùng thư viện tokenizers (Rust, siêu nhanh, không phụ thuộc nặng vào PyTorch)
        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_pretrained(model_name)
            self._is_rust_tokenizer = True
        except Exception:
            # Ưu tiên 2: Dùng transformers AutoTokenizer
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self._is_rust_tokenizer = False
            except Exception:
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._is_rust_tokenizer = False
                except Exception as exc:
                    print(f"[LegalTokenizer Ghi chú] Không thể nạp AutoTokenizer ({exc}). Sử dụng fallback ước lượng...")
                    self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Đếm số lượng tokens thực tế."""
        if not text:
            return 0
        if self._tokenizer:
            try:
                if self._is_rust_tokenizer:
                    return len(self._tokenizer.encode(text).ids)
                else:
                    return len(self._tokenizer.encode(text, add_special_tokens=True))
            except Exception:
                pass
        # Fallback ước tính tiếng Việt (1 từ ≈ 1.4 token + 2 special tokens)
        words = text.split()
        return int(len(words) * 1.4) + 2

    def count_final_embedding_tokens(self, content: str) -> int:
        """Quy tắc 2: Đếm token trên final text thực tế đưa vào embedding (bao gồm 'passage: ')."""
        final_text = f"passage: {content}"
        return self.count_tokens(final_text)


def process_chunks_token_control(
    chunks: List[Dict[str, Any]],
    tokenizer: Optional[LegalTokenizer] = None,
    hard_max_tokens: int = EMBED_HARD_MAX_TOKENS,
    target_tokens: int = EMBED_TARGET_TOKENS,
) -> List[Dict[str, Any]]:
    """Kiểm soát kích thước embedding bằng tokenizer thật trên final embedding string:
    - Nếu final text (passage: + breadcrumb + intro + content) <= hard_max_tokens: giữ nguyên.
    - Nếu > hard_max_tokens:
      - Tạo các subchunks mang source_chunk_id, subchunk_index, parent_chunk_id = original_chunk_id
      - Đặt subchunks is_embeddable = True
      - Đặt original chunk is_embeddable = False (vẫn lưu full trên PostgreSQL).
    """
    if tokenizer is None:
        tokenizer = LegalTokenizer()

    final_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        # Tính token count thực tế trên chuỗi cuối cùng
        chunk_final_tokens = tokenizer.count_final_embedding_tokens(chunk["content"])
        chunk["token_count"] = chunk_final_tokens

        # Nếu chunk không phải là đối tượng embed (ví dụ ArticleChunk hoặc Clause có Point con)
        if not chunk.get("is_embeddable", True):
            final_chunks.append(chunk)
            continue

        # Nếu không vượt ngưỡng giới hạn cứng: giữ nguyên
        if chunk_final_tokens <= hard_max_tokens:
            final_chunks.append(chunk)
            continue

        # Chunk vượt ngưỡng: chia thành các subchunks
        content_lines = chunk["content"].split("\n", 1)
        header = content_lines[0] if len(content_lines) > 1 else ""
        body = content_lines[1] if len(content_lines) > 1 else content_lines[0]

        # Tách body thành các phân đoạn (ưu tiên ; . \n, sau đó dấu phẩy, và cuối cùng là từ)
        major_segments = [s.strip() for s in re.split(r"(?<=[;.\n])\s+", body) if s.strip()]
        atomic_segments: List[str] = []
        for seg in major_segments:
            seg_tokens = tokenizer.count_final_embedding_tokens(f"{header}\n{seg}")
            if seg_tokens > hard_max_tokens:
                comma_splits = [cs.strip() for cs in re.split(r"(?<=[,])\s+", seg) if cs.strip()]
                for cs in comma_splits:
                    cs_tokens = tokenizer.count_final_embedding_tokens(f"{header}\n{cs}")
                    if cs_tokens > hard_max_tokens:
                        # Tách theo từ nếu phân đoạn có dấu phẩy vẫn vượt ngưỡng
                        w_curr: List[str] = []
                        for w in cs.split():
                            w_test = " ".join(w_curr + [w])
                            if tokenizer.count_final_embedding_tokens(f"{header}\n{w_test}") <= hard_max_tokens or not w_curr:
                                w_curr.append(w)
                            else:
                                atomic_segments.append(" ".join(w_curr))
                                w_curr = [w]
                        if w_curr:
                            atomic_segments.append(" ".join(w_curr))
                    else:
                        atomic_segments.append(cs)
            else:
                atomic_segments.append(seg)

        sub_batches: List[str] = []
        current_batch: List[str] = []

        for s in atomic_segments:
            test_batch = current_batch + [s]
            test_content = " ".join(test_batch)
            test_tokens = tokenizer.count_final_embedding_tokens(f"{header}\n{test_content}")

            if test_tokens <= hard_max_tokens:
                current_batch.append(s)
            else:
                if current_batch:
                    sub_batches.append(" ".join(current_batch))
                    current_batch = [s]
                else:
                    sub_batches.append(s)
                    current_batch = []

        if current_batch:
            sub_batches.append(" ".join(current_batch))

        # Chunk gốc chuyển thành container (is_embeddable = False cho Qdrant, vẫn giữ trọn vẹn trong Postgres)
        orig_copy = dict(chunk)
        orig_copy["is_embeddable"] = False
        final_chunks.append(orig_copy)

        # Tạo các subchunks (Quy tắc 3)
        for idx, sub_text in enumerate(sub_batches):
            sub_id = f"{chunk['id']}_sub{idx:02d}"
            sub_content = f"{header}\n{sub_text}".strip()
            sub_token_count = tokenizer.count_final_embedding_tokens(sub_content)

            subchunk = {
                "id": sub_id,
                "document_id": chunk["document_id"],
                "version_id": chunk["version_id"],
                "chapter_id": chunk.get("chapter_id"),
                "article_id": chunk["article_id"],
                "clause_id": chunk.get("clause_id"),
                "point_id": chunk.get("point_id"),
                "parent_chunk_id": chunk["id"],  # Parent là original chunk
                "root_article_chunk_id": chunk.get("root_article_chunk_id"),
                "source_chunk_id": chunk["id"],   # Quy tắc 3: source_chunk_id
                "subchunk_index": idx,            # Quy tắc 3: subchunk_index
                "is_embeddable": True,            # Quy tắc 3: Subchunk được embed vào Qdrant
                "chunk_type": "subchunk",
                "chunk_index": chunk.get("chunk_index", 1),
                "content": sub_content,
                "normalized_content": remove_vietnamese_accents(sub_content),
                "token_count": sub_token_count,
                "metadata": dict(chunk.get("metadata", {})),
                "validity": dict(chunk.get("validity", {})),
                "source": dict(chunk.get("source", {})),
                "search": dict(chunk.get("search", {})),
            }
            subchunk["search"]["qdrant_point_id"] = sub_id
            final_chunks.append(subchunk)

    return final_chunks


class LlamaIndexEmbedder:
    """Embedding Generator using LlamaIndex HuggingFaceEmbedding abstraction with token control support."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading LlamaIndex HuggingFaceEmbedding: {model_name}...")
        self.tokenizer = LegalTokenizer(model_name)
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
