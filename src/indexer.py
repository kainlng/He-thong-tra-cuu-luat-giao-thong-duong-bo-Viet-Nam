import os
import re
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env) before importing HuggingFace libraries
load_dotenv()
if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VECTOR_DB_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_DB_DIR / "metadata.pkl"


class MultilingualE5Embedder:
    """Embedding Generator using intfloat/multilingual-e5-base."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        print(f"Loading E5 model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def get_document_embeddings(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed document passages by prefixing with 'passage: '."""
        passages = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Embed search query by prefixing with 'query: '."""
        formatted_query = f"query: {query}"
        embedding = self.model.encode(
            [formatted_query],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)


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
    """Split legal markdown file into structured article chunks (Điều x)."""
    text = markdown_file.read_text(encoding="utf-8")
    pattern = r"(Điều\s+\d+\.\s+[^\n]+)"
    parts = re.split(pattern, text)

    chunks = []
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            full_chunk_text = f"{header}\n{content}"
            if len(full_chunk_text.strip()) > 20:
                chunks.append({
                    "source": markdown_file.name,
                    "header": header,
                    "text": full_chunk_text[:1500],
                })
    else:
        lines = text.split("\n\n")
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) > 800:
                chunks.append({"source": markdown_file.name, "header": "General Text", "text": current_chunk.strip()})
                current_chunk = line + "\n\n"
            else:
                current_chunk += line + "\n\n"
        if current_chunk.strip():
            chunks.append({"source": markdown_file.name, "header": "General Text", "text": current_chunk.strip()})

    return chunks


def build_index():
    md_files = list(PROCESSED_DIR.glob("*.md"))
    if not md_files:
        print(f"No processed markdown files found in {PROCESSED_DIR}")
        return

    print(f"Found {len(md_files)} markdown files in {PROCESSED_DIR}")
    all_chunks = []
    for md in md_files:
        chunks = parse_legal_chunks(md)
        all_chunks.extend(chunks)
        print(f"Extracted {len(chunks)} chunks from {md.name}")

    if not all_chunks:
        print("No valid text chunks found.")
        return

    print(f"Total chunks to embed: {len(all_chunks)}")

    embedder = MultilingualE5Embedder(model_name="intfloat/multilingual-e5-base")
    sample_texts = [c["text"] for c in all_chunks]

    print("\nGenerating embeddings with intfloat/multilingual-e5-base...")
    embeddings = embedder.get_document_embeddings(sample_texts, batch_size=16)
    print(f"Embeddings shape: {embeddings.shape}")

    embedding_dim = embeddings.shape[1]
    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    vector_store.add_documents(embeddings, all_chunks)
    vector_store.save(INDEX_PATH, METADATA_PATH)


if __name__ == "__main__":
    build_index()
