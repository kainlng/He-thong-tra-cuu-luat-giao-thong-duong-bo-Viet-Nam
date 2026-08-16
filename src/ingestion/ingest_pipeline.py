import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.config import PROCESSED_DIR, RAW_DIR
from src.ingestion.legal_parser import HierarchicalLegalParser, calculate_file_hash
from src.ingestion.hierarchical_chunker import HierarchicalLegalChunker
from src.storage.postgres_manager import PostgresManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.indexer import LlamaIndexEmbedder


def run_ingestion(force_reindex: bool = False):
    """Pipeline nạp dữ liệu hoàn chỉnh:
    PDF/Markdown -> Parser Cấu trúc Cây -> Hierarchical Chunker -> Embedder -> PostgreSQL -> Qdrant.
    """
    print("=" * 70)
    print("🚀 KHỞI ĐỘNG PIPELINE INGESTION PHÁP LUẬT PHÂN CẤP (PostgreSQL + Qdrant)")
    print("=" * 70)

    # 1. Khởi tạo Managers
    pg_mgr = PostgresManager()
    if force_reindex:
        pg_mgr.reset_db()
    else:
        pg_mgr.init_db()

    qdrant_mgr = QdrantManager()
    parser = HierarchicalLegalParser()
    chunker = HierarchicalLegalChunker()
    embedder = LlamaIndexEmbedder()

    md_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not md_files:
        print(f"[Ingest Warning] Không tìm thấy file markdown nào trong {PROCESSED_DIR}.")
        print("Vui lòng chạy 'python src/ingestion/data_loader.py' trước để trích xuất PDF sang Markdown.")
        return

    total_chunks_indexed = 0

    for md_file in md_files:
        print(f"\n📂 Đang xử lý tài liệu: {md_file.name}...")
        file_hash = calculate_file_hash(md_file)

        if not force_reindex and pg_mgr.check_document_exists_by_hash(file_hash):
            print(f"⏩ Tài liệu {md_file.name} (hash: {file_hash[:12]}...) đã tồn tại trong PostgreSQL. Bỏ qua.")
            continue

        raw_content = md_file.read_text(encoding="utf-8")

        # 2. Phân tích cấu trúc phân cấp
        print("  -> Bóc tách cây phân cấp (Văn bản -> Chương -> Điều -> Khoản -> Điểm)...")
        parsed_tree = parser.parse_markdown(raw_content, file_name=md_file.name, file_hash=file_hash)

        # 3. Tạo Chunks phân cấp
        print("  -> Tạo các Semantic Chunks và liên kết parent_chunk_id...")
        chunks = chunker.create_chunks(parsed_tree)
        print(f"  -> Đã tạo {len(chunks)} chunks từ {len(parsed_tree['articles'])} Điều luật.")

        # 4. Sinh Embeddings cho các chunks
        print(f"  -> Đang sinh Vector Embeddings ({embedder.model.__class__.__name__})...")
        chunk_texts = [c["content"] for c in chunks]
        embeddings = embedder.get_document_embeddings(chunk_texts, batch_size=16)

        # 5. Lưu toàn bộ cây phân cấp và chunks vào PostgreSQL (Source of Truth)
        print("  -> Đang lưu trữ vào 7 bảng PostgreSQL...")
        pg_mgr.insert_legal_tree(parsed_tree, chunks)

        # 6. Upsert Vector và Payload tối giản vào Qdrant
        print("  -> Đang đồng bộ Vector và Metadata Filter vào Qdrant...")
        qdrant_mgr.upsert_chunks(chunks, embeddings)

        total_chunks_indexed += len(chunks)

    # Thống kê tổng kết
    stats = pg_mgr.get_database_stats()
    print("\n" + "=" * 70)
    print("HOÀN TẤT INGESTION PHÁP LUẬT!")
    print(f"- Tổng số Văn bản: {stats.get('documents', 0)}")
    print(f"- Tổng số Phiên bản: {stats.get('versions', 0)}")
    print(f"- Tổng số Chương: {stats.get('chapters', 0)}")
    print(f"- Tổng số Điều: {stats.get('articles', 0)}")
    print(f"- Tổng số Khoản: {stats.get('clauses', 0)}")
    print(f"- Tổng số Điểm: {stats.get('points', 0)}")
    print(f"- Tổng số Chunks (PostgreSQL + Qdrant): {stats.get('chunks', 0)}")
    print("=" * 70)


if __name__ == "__main__":
    force = "--reset" in sys.argv or "--force" in sys.argv
    run_ingestion(force_reindex=force)
