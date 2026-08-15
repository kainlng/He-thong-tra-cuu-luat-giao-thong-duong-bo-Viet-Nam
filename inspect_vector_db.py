import faiss
import pickle
import sqlite3
import numpy as np
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
INDEX_PATH = VECTOR_DB_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_DB_DIR / "metadata.pkl"
FTS_DB_PATH = VECTOR_DB_DIR / "legal_fts.db"


def inspect_vector_store():
    print("=" * 60)
    print("🔍 KIỂM TRA CẤU TRÚC VECTOR DATABASE (LlamaIndex + FAISS + FTS5)")
    print("=" * 60)

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        print(f"❌ Không tìm thấy file index tại {VECTOR_DB_DIR}")
        return

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print("\n📦 1. FAISS DENSE VECTOR STORE (faiss_index.bin & metadata.pkl)")
    print(f" - Tổng số vector (ntotal)    : {index.ntotal}")
    print(f" - Số chiều vector (dimension): {index.d} (LlamaIndex intfloat/multilingual-e5-base)")
    print(f" - Loại Index FAISS           : {type(index).__name__} (Inner Product / Cosine)")
    print(f" - Tổng số bản ghi metadata   : {len(metadata)}")

    # Display First 2 Vectors with their metadata
    print("\n" + "-" * 60)
    print("📋 CHI TIẾT VECTOR MẪU ĐƯỢC LƯU TRỮ:")
    print("-" * 60)

    for i in range(min(2, index.ntotal)):
        vector = index.reconstruct(i)
        meta = metadata[i]

        print(f"\n🔹 Vector ID #{i}:")
        print(f"   • Vector shape  : {vector.shape} (mảng 768 số thực float32)")
        print(f"   • 10 giá trị đầu: {vector[:10].round(6).tolist()} ...")
        print(f"   • Chuẩn L2 norm : {np.linalg.norm(vector):.4f} (đã chuẩn hóa cosine)")
        print(f"   • Tiêu đề Điều  : {meta.get('header')}")
        print(f"   • File nguồn    : {meta.get('source')}")
        print(f"   • Vị trí        : {meta.get('location')}")
        print(f"   • Đoạn văn bản  :\n     \"{meta.get('text', '')[:180]}...\"")

    # 2. Inspect SQLite FTS5 BM25 Store
    if FTS_DB_PATH.exists():
        conn = sqlite3.connect(str(FTS_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM legal_fts")
        fts_count = cursor.fetchone()[0]

        cursor.execute("SELECT source, header, article_num, chunk_index, location FROM legal_fts LIMIT 2")
        rows = cursor.fetchall()

        print("\n" + "=" * 60)
        print(f"🗄️ 2. SQLITE FTS5 BM25 STORE (legal_fts.db) - {fts_count} bản ghi")
        print("=" * 60)
        for row in rows:
            print(f" - [Index #{row['chunk_index']}] {row['header']} (Nguồn: {row['source']})")


if __name__ == "__main__":
    inspect_vector_store()
