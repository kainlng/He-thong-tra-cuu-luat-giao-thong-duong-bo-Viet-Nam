import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.storage.postgres_manager import PostgresManager
from src.storage.qdrant_manager import QdrantManager


def inspect_databases():
    print("=" * 70)
    print("🔍 KIỂM TRA CƠ SỞ DỮ LIỆU PHÁP LUẬT PHÂN CẤP (PostgreSQL + Qdrant)")
    print("=" * 70)

    # 1. Inspect PostgreSQL Source of Truth
    pg_mgr = PostgresManager()
    stats = pg_mgr.get_database_stats()

    print("\n🏛️ 1. POSTGRESQL RELATIONAL STORE (Source of Truth - 7 Bảng):")
    print(f" - Số lượng Văn bản (Documents) : {stats.get('documents', 0)}")
    print(f" - Số lượng Phiên bản (Versions): {stats.get('versions', 0)}")
    print(f" - Số lượng Chương (Chapters)   : {stats.get('chapters', 0)}")
    print(f" - Số lượng Điều luật (Articles): {stats.get('articles', 0)}")
    print(f" - Số lượng Khoản (Clauses)     : {stats.get('clauses', 0)}")
    print(f" - Số lượng Điểm (Points)       : {stats.get('points', 0)}")
    print(f" - Số lượng Chunks (Full Text)  : {stats.get('chunks', 0)}")

    all_chunks = pg_mgr.get_all_chunks()
    if all_chunks:
        print("\n📋 MẪU CHUNK ĐƯỢC LƯU TRONG POSTGRESQL:")
        sample = all_chunks[0]
        print(f"🔹 Chunk ID   : {sample['id']}")
        print(f"   • Loại Chunk : {sample.get('chunk_type')}")
        print(f"   • Metadata   : {sample.get('metadata')}")
        print(f"   • Hiệu lực   : {sample.get('validity')}")
        print(f"   • Trích đoạn : \"{sample.get('content', '')[:160]}...\"")

    # 2. Inspect Qdrant Vector Search Index
    print("\n" + "-" * 70)
    print("🌌 2. QDRANT VECTOR SEARCH INDEX (Minimal Payload + Embeddings):")
    qdrant_mgr = QdrantManager()
    if qdrant_mgr.client:
        try:
            coll_info = qdrant_mgr.client.get_collection(qdrant_mgr.collection_name)
            print(f" - Collection Name : '{qdrant_mgr.collection_name}'")
            print(f" - Vector Dimension: {qdrant_mgr.vector_dim} (intfloat/multilingual-e5-base)")
            print(f" - Metric Distance : Cosine")
            print(f" - Points Count    : {coll_info.points_count}")
        except Exception as exc:
            print(f" - Qdrant Info Note: {exc}")
    print("=" * 70)


if __name__ == "__main__":
    inspect_databases()
