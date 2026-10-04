import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.legal_parser import HierarchicalLegalParser
from src.ingestion.hierarchical_chunker import HierarchicalLegalChunker
from src.storage.postgres_manager import PostgresManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.indexer import LlamaIndexEmbedder, process_chunks_token_control
from src.retrieval.query_engine import LegalQueryEngine

FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"


def run_benchmark():
    print("=" * 70)
    print("BẮT ĐẦU CHẠY RETRIEVAL BENCHMARK (5 CASES THEO ĐỀ XUẤT)")
    print("=" * 70)

    # Khởi tạo pipeline
    parser = HierarchicalLegalParser()
    chunker = HierarchicalLegalChunker()
    embedder = LlamaIndexEmbedder()
    pg_mgr = PostgresManager()
    pg_mgr.reset_db()
    qdrant_mgr = QdrantManager(collection_name="benchmark_retrieval_v2")

    # Ingest 2 fixtures phong phú: clause_with_points.md và long_point.md
    for f_name in ["clause_with_points.md", "long_point.md"]:
        f_path = FIXTURES_DIR / f_name
        content = f_path.read_text(encoding="utf-8")
        tree = parser.parse_markdown(content, f_name, file_hash=f"sha256:{f_name}")
        raw_chunks = chunker.create_chunks(tree)
        
        # Token control với ngưỡng thử nghiệm để tạo subchunk cho long_point
        chunks = process_chunks_token_control(
            raw_chunks,
            tokenizer=embedder.tokenizer,
            hard_max_tokens=200,
            target_tokens=150,
        )

        # Lưu trọn vẹn vào Postgres
        pg_mgr.insert_legal_tree(tree, chunks)

        # Chỉ embed các chunk có is_embeddable=True vào Qdrant
        embeddables = [c for c in chunks if c.get("is_embeddable", True)]
        texts = [c["content"] for c in embeddables]
        embs = embedder.get_document_embeddings(texts, batch_size=8)
        qdrant_mgr.upsert_chunks(embeddables, embs)

    engine = LegalQueryEngine(
        embedder=embedder,
        qdrant_mgr=qdrant_mgr,
        postgres_mgr=pg_mgr,
    )

    # Case 1: Query trả về Điểm cụ thể (ví dụ: dùng điện thoại khi lái xe)
    print("\n--- CASE 1: Query tìm Điểm cụ thể ('dùng điện thoại khi lái xe') ---")
    res1 = engine.retrieve("Dùng tay cầm và sử dụng điện thoại khi đang lái xe", top_k=5, top_n=2)
    assert len(res1) > 0, "Case 1 failed"
    top_art = res1[0]
    print(f"  Hit Điều: {top_art['article_number']} - {top_art['article_title']}")
    assert "Điều 5" in f"Điều {top_art['article_number']}"
    assert "điện thoại" in top_art["passage_text"]
    print("  ✅ CASE 1 PASSED: Điểm tìm thấy và mở rộng đầy đủ ngữ cảnh Điều!")

    # Case 2: Query mang ý nghĩa Khoản (ví dụ: mức phạt chạy quá tốc độ từ 5 đến dưới 10 km/h)
    print("\n--- CASE 2: Query mang ý nghĩa Khoản ('mức phạt chạy quá tốc độ 5 đến 10 km/h') ---")
    res2 = engine.retrieve("Mức phạt chạy quá tốc độ từ 05 km/h đến dưới 10 km/h", top_k=5, top_n=2)
    assert len(res2) > 0, "Case 2 failed"
    assert "800.000 đồng" in res2[0]["passage_text"]
    print("  ✅ CASE 2 PASSED: Tìm đúng quy định mức phạt của Khoản!")

    # Case 3: Query mang ý nghĩa toàn Điều (ví dụ: xử phạt người điều khiển xe ô tô vi phạm)
    print("\n--- CASE 3: Query mang ý nghĩa bao quát toàn Điều ('xử phạt ô tô vi phạm giao thông') ---")
    res3 = engine.retrieve("Quy định xử phạt người điều khiển xe ô tô vi phạm giao thông", top_k=5, top_n=2)
    assert len(res3) > 0, "Case 3 failed"
    assert "Điều 5" in f"Điều {res3[0]['article_number']}"
    print("  ✅ CASE 3 PASSED: Parent expansion tái tạo trọn vẹn Điều luật!")

    # Case 4: Query tìm nội dung nằm ở nửa sau của Point dài (đã được chia thành subchunk)
    print("\n--- CASE 4: Query tìm nội dung Subchunk ('gây ách tắc giao thông nhiều giờ liền trên cao tốc') ---")
    res4 = engine.retrieve("Hàng siêu trường siêu trọng gây ách tắc giao thông kéo dài trên cao tốc", top_k=5, top_n=2)
    assert len(res4) > 0, "Case 4 failed"
    assert "Điều 15" in f"Điều {res4[0]['article_number']}"
    assert "hàng siêu trường" in res4[0]["passage_text"].lower()
    print("  ✅ CASE 4 PASSED: Subchunk retrieval thành công và map ngược về Điều 15!")

    # Case 5: Hierarchical expansion chứng minh chuỗi Point -> Clause -> Article
    print("\n--- CASE 5: Kiểm tra chuỗi Hierarchical Expansion toàn vẹn ---")
    passage = res1[0]["passage_text"]
    assert "nghị định" in passage.lower()
    assert "Điều 5" in passage
    assert "1." in passage
    assert "2." in passage
    assert "a)" in passage and "b)" in passage and "c)" in passage
    print("  ✅ CASE 5 PASSED: Toàn bộ cây Điều -> Khoản -> Điểm được tái tạo liền mạch!")

    print("\n" + "=" * 70)
    print("🎉 TẤT CẢ 5 BENCHMARK RETRIEVAL CASES ĐÃ VƯỢT QUA 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
