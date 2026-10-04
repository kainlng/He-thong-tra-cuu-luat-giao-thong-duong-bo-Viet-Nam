import sys
import argparse
from pathlib import Path
from typing import Optional, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.config import PROCESSED_DIR, RAW_DIR
from src.core.utils import calculate_file_sha256
from src.ingestion.data_loader import process_all_pdfs
from src.ingestion.legal_parser import HierarchicalLegalParser
from src.ingestion.hierarchical_chunker import HierarchicalLegalChunker
from src.storage.postgres_manager import PostgresManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.indexer import LlamaIndexEmbedder, process_chunks_token_control


def run_ingestion(
    reprocess: bool = False,
    force_ocr: bool = False,
    reset_db: bool = False,
):
    """Pipeline nạp dữ liệu hoàn chỉnh:
    PDF -> Xberg Extraction (Tesseract vie) -> Parser Cấu trúc Cây -> Hierarchical Chunker -> Embedder -> PostgreSQL -> Qdrant.
    """
    print("=" * 70)
    print("KHỞI ĐỘNG PIPELINE INGESTION PHÁP LUẬT PHÂN CẤP (XBERG + POSTGRESQL + QDRANT)")
    print("=" * 70)

    # 0. Chạy DataLoader bóc tách PDF sang Markdown bằng Xberg
    print("\n[Bước 1/6] Kiểm tra & Bóc tách PDF từ data/raw bằng Xberg (Tesseract vie)...")
    process_all_pdfs(overwrite=reprocess, force_ocr=force_ocr)

    # 1. Khởi tạo Managers
    pg_mgr = PostgresManager()
    qdrant_mgr = QdrantManager()

    if reset_db:
        print("[Database Reset] Đang xóa và khởi tạo lại toàn bộ 7 bảng trong PostgreSQL...")
        pg_mgr.reset_db()
        print(f"[Database Reset] Đang làm mới collection '{qdrant_mgr.collection_name}' trong Qdrant...")
        qdrant_mgr.reset_collection()
    else:
        pg_mgr.init_db()

    parser = HierarchicalLegalParser()
    chunker = HierarchicalLegalChunker()
    embedder = LlamaIndexEmbedder()

    all_md_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not all_md_files:
        print(f"[Ingest Warning] Không tìm thấy file markdown nào trong {PROCESSED_DIR}.")
        return

    # Lọc danh sách file Markdown: chọn phiên bản đầy đủ nhất (nhiều ký tự nhất) giữa các engine (xberg/llamacloud)
    stem_map = {}
    for f in all_md_files:
        if f.name.startswith("."):
            continue
        base_stem = f.stem.replace("_xberg", "").replace("_llamacloud", "")
        if base_stem not in stem_map or f.stat().st_size > stem_map[base_stem].stat().st_size:
            stem_map[base_stem] = f
    md_files = sorted(stem_map.values(), key=lambda p: p.name)

    total_chunks_indexed = 0

    for md_file in md_files:
        print(f"\nĐang xử lý tài liệu: {md_file.name}...")
        file_hash = calculate_file_sha256(md_file)

        if not reset_db and pg_mgr.check_document_exists_by_hash(file_hash):
            print(f"Tài liệu {md_file.name} (hash: {file_hash[:12]}...) đã tồn tại trong PostgreSQL. Bỏ qua.")
            continue

        raw_content = md_file.read_text(encoding="utf-8")

        # 2. Phân tích cấu trúc phân cấp (có tích hợp Page Tracking và Provenance)
        print("  -> Bóc tách cây phân cấp (Văn bản -> Chương -> Điều -> Khoản -> Điểm)...")
        parsed_tree = parser.parse_markdown(raw_content, file_name=md_file.name, file_hash=file_hash)

        # 3. Tạo Chunks phân cấp
        print("  -> Tạo các Semantic Chunks và liên kết parent_chunk_id...")
        raw_chunks = chunker.create_chunks(parsed_tree)

        # 3b. Kiểm soát token bằng tokenizer thật & Subchunking nếu vượt ngưỡng 450 tokens
        chunks = process_chunks_token_control(raw_chunks, tokenizer=embedder.tokenizer)
        embeddable_chunks = [c for c in chunks if c.get("is_embeddable", True)]
        print(f"  -> Tổng số {len(chunks)} chunks ({len(embeddable_chunks)} retrieval chunks được embed) từ {len(parsed_tree['articles'])} Điều luật.")

        # 4. Sinh Embeddings CHỈ cho các chunks được đánh dấu is_embeddable=True
        print(f"  -> Đang sinh Vector Embeddings cho {len(embeddable_chunks)} chunks ({embedder.model.__class__.__name__})...")
        chunk_texts = [c["content"] for c in embeddable_chunks]
        embeddings = embedder.get_document_embeddings(chunk_texts, batch_size=16)

        # 5. Lưu toàn bộ cây phân cấp và toàn bộ chunks (kể cả container chunks) vào PostgreSQL (Source of Truth)
        print("  -> Đang lưu trữ toàn bộ cấu trúc vào 7 bảng PostgreSQL...")
        pg_mgr.insert_legal_tree(parsed_tree, chunks)

        # 6. Upsert Vector và Payload tối giản vào Qdrant (chỉ nạp embeddable chunks)
        print("  -> Đang đồng bộ Vector và Metadata Filter vào Qdrant...")
        qdrant_mgr.delete_points_by_document_id(parsed_tree["document"]["id"])
        qdrant_mgr.upsert_chunks(embeddable_chunks, embeddings)

        total_chunks_indexed += len(embeddable_chunks)

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
    cli_parser = argparse.ArgumentParser(description="Legal RAG Ingestion Pipeline with XBerg & Tesseract OCR")
    cli_parser.add_argument(
        "--reprocess", "--force-extract",
        action="store_true",
        help="Re-run PDF extraction with Xberg even if files were processed before"
    )
    cli_parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force Tesseract OCR on all PDF pages instead of auto-detection"
    )
    cli_parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset and recreate PostgreSQL tables and clear Qdrant collection"
    )
    cli_parser.add_argument(
        "--force",
        action="store_true",
        help="Full re-process and database reset (legacy alias)"
    )

    args = cli_parser.parse_args()

    reprocess_flag = args.reprocess or args.force
    reset_db_flag = args.reset_db or args.force
    force_ocr_flag = args.force_ocr

    run_ingestion(
        reprocess=reprocess_flag,
        force_ocr=force_ocr_flag,
        reset_db=reset_db_flag,
    )
