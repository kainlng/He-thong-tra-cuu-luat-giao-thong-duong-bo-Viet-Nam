import os
import re
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()
if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

import numpy as np
import faiss

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.fts_store import SQLiteFTSStore, FTS_DB_PATH

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VECTOR_DB_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_DB_DIR / "metadata.pkl"


def clean_legal_markdown(text: str) -> str:
    """Strip running gazette headers, footers, and redundant page titles from extracted markdown."""
    # 1. Remove CÔNG BÁO running headers
    text = re.sub(r"(?im)^\s*CÔNG BÁO/[^\n]*\n?", "", text)
    # 2. Remove VĂN BẢN QUY PHẠM PHÁP LUẬT
    text = re.sub(r"(?im)^\s*VĂN BẢN QUY PHẠM PHÁP LUẬT\s*\n?", "", text)
    # 3. Remove (Tiếp theo Công báo...)
    text = re.sub(r"(?im)^\s*\(?Tiếp theo Công báo[^\n]*\)?\s*\n?", "", text)
    # 4. Remove CHỦ TỊCH NƯỚC - QUỐC HỘI header banners
    text = re.sub(r"(?im)^\s*#+\s*(?:CHỦ TỊCH NƯỚC\s*-\s*QUỐC HỘI|QUỐC HỘI|CÔNG BÁO).*?\n", "", text)
    # 5. Normalize multiple consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class LlamaIndexEmbedder:
    """Embedding Generator using LlamaIndex HuggingFaceEmbedding abstraction (intfloat/multilingual-e5-base)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
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


class FAISSVectorStore:
    """FAISS Vector Database wrapper using Inner Product (Cosine Similarity)."""

    def __init__(self, embedding_dim: int):
        self.dim = embedding_dim
        # IndexFlatIP (Inner Product) for Cosine Similarity after L2 Normalization
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []

    def add_documents(self, embeddings: np.ndarray, docs_metadata: list[dict]):
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}")
        self.index.add(embeddings)
        self.metadata.extend(docs_metadata)

    def save(self, index_path: Path, metadata_path: Path):
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"\nSaved FAISS index ({self.index.ntotal} vectors) to {index_path}")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path):
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        store = cls(embedding_dim=index.d)
        store.index = index
        store.metadata = metadata
        return store

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({"score": float(score), "metadata": self.metadata[idx]})
        return results


def parse_legal_chunks(markdown_file: Path) -> list[dict]:
    """Split legal markdown file into structured article chunks with exact document position metadata."""
    raw_text = markdown_file.read_text(encoding="utf-8")
    text = clean_legal_markdown(raw_text)

    pattern = r"(?:^|\n)#*\s*(Điều\s+\d+\.\s+[^\n]+)"
    parts = re.split(pattern, text)

    chunks = []
    chunk_index = 1
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            raw_header = parts[i].strip()
            header = re.sub(r"^#+\s*", "", raw_header).strip()
            raw_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            content = clean_legal_markdown(raw_content)
            full_chunk_text = f"{header}\n{content}"
            if len(full_chunk_text.strip()) > 20:
                article_match = re.search(r"Điều\s+(\d+)", header)
                article_num = article_match.group(1) if article_match else ""

                chunks.append({
                    "source": markdown_file.name,
                    "header": header,
                    "article_num": article_num,
                    "chunk_index": chunk_index,
                    "location": f"Văn bản: {markdown_file.name} > {header} (Vị trí #{chunk_index})",
                    "text": full_chunk_text[:1500],
                })
                chunk_index += 1
    else:
        lines = text.split("\n\n")
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) > 800:
                chunks.append({
                    "source": markdown_file.name,
                    "header": "Nội dung chung",
                    "article_num": "",
                    "chunk_index": chunk_index,
                    "location": f"Văn bản: {markdown_file.name} > Vị trí #{chunk_index}",
                    "text": current_chunk.strip(),
                })
                chunk_index += 1
                current_chunk = line + "\n\n"
            else:
                current_chunk += line + "\n\n"
        if current_chunk.strip():
            chunks.append({
                "source": markdown_file.name,
                "header": "Nội dung chung",
                "article_num": "",
                "chunk_index": chunk_index,
                "location": f"Văn bản: {markdown_file.name} > Vị trí #{chunk_index}",
                "text": current_chunk.strip(),
            })

    return chunks


def build_index():
    """Build FAISS Vector Index and SQLite FTS5 BM25 Index from processed markdown documents."""
    md_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not md_files:
        print(f"No processed markdown files found in {PROCESSED_DIR}")
        return

    print(f"Indexing {len(md_files)} markdown files from {PROCESSED_DIR}")
    all_chunks = []
    for md in md_files:
        chunks = parse_legal_chunks(md)
        all_chunks.extend(chunks)
        print(f"Extracted {len(chunks)} chunks from {md.name}")

    if not all_chunks:
        print("No valid text chunks found.")
        return

    print(f"Total chunks to index: {len(all_chunks)}")

    # 1. Build & Save SQLite FTS5 BM25 Index
    print("\nBuilding SQLite FTS5 BM25 index...")
    fts_store = SQLiteFTSStore(FTS_DB_PATH)
    fts_store.reset_db()
    fts_store.add_documents(all_chunks)

    # 2. Build & Save FAISS Dense Vector Index using LlamaIndex Embedder
    embedder = LlamaIndexEmbedder(model_name="intfloat/multilingual-e5-base")
    sample_texts = [c["text"] for c in all_chunks]

    print("\nGenerating embeddings with LlamaIndex HuggingFaceEmbedding (intfloat/multilingual-e5-base)...")
    embeddings = embedder.get_document_embeddings(sample_texts, batch_size=16)
    print(f"Embeddings shape: {embeddings.shape}")

    embedding_dim = embeddings.shape[1]
    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    vector_store.add_documents(embeddings, all_chunks)
    vector_store.save(INDEX_PATH, METADATA_PATH)


if __name__ == "__main__":
    build_index()
