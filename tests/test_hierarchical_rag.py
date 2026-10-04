import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.legal_parser import HierarchicalLegalParser
from src.ingestion.hierarchical_chunker import HierarchicalLegalChunker
from src.storage.postgres_manager import PostgresManager
from src.storage.qdrant_manager import QdrantManager
from src.storage.indexer import LlamaIndexEmbedder
from src.retrieval.context_expander import HierarchicalContextExpander
from src.retrieval.reranker import LegalReranker
from src.retrieval.query_engine import LegalQueryEngine
from src.generation.citation_builder import CitationBuilder


SAMPLE_LEGAL_MARKDOWN = """
## LUẬT TRẬT TỰ, AN TOÀN GIAO THÔNG ĐƯỜNG BỘ
Luật số: 36/2024/QH15

### Chương I: NHỮNG QUY ĐỊNH CHUNG

**Điều 1. Phạm vi điều chỉnh**
Luật này quy định về quy tắc, phương tiện, người tham gia giao thông đường bộ.

**Điều 2. Giải thích từ ngữ**
1. Trật tự, an toàn giao thông đường bộ là trạng thái giao thông trên đường bộ có trật tự, bảo đảm an toàn, thông suốt.
2. Phương tiện giao thông đường bộ là các loại xe cơ giới và xe thô sơ.

### Chương II: QUY TẮC GIAO THÔNG ĐƯỜNG BỘ

### Điều 9. Các hành vi bị nghiêm cấm
1. Điều khiển xe cơ giới tham gia giao thông đường bộ không có giấy phép lái xe theo quy định.
2. Điều khiển phương tiện tham gia giao thông đường bộ mà trong máu hoặc hơi thở có nồng độ cồn.
3. Điều khiển phương tiện tham gia giao thông đường bộ mà trong cơ thể có chất ma túy hoặc chất kích thích khác.
4. Đua xe, tổ chức đua xe, xúi giục, giúp sức, cổ vũ đua xe trái phép.
5. Dùng tay cầm và sử dụng điện thoại hoặc thiết bị điện tử khác khi điều khiển phương tiện tham gia giao thông.
"""


def run_all_tests():
    print("=" * 70)
    print("BẮT ĐẦU KIỂM THỬ TOÀN DIỆN HỆ THỐNG POSTGRESQL + QDRANT HIERARCHICAL RAG")
    print("=" * 70)

    # 1. Test Parser
    print("\n[TEST 1] Kiểm tra Hierarchical Legal Parser...")
    parser = HierarchicalLegalParser()
    tree = parser.parse_markdown(SAMPLE_LEGAL_MARKDOWN, file_name="36-2024-qh15.md", file_hash="sha256:test123456")
    
    assert tree["document"]["document_number"] == "36/2024/QH15", "Document number extraction failed"
    assert len(tree["chapters"]) == 2, f"Expected 2 chapters, got {len(tree['chapters'])}"
    assert len(tree["articles"]) == 3, f"Expected 3 articles, got {len(tree['articles'])}"
    assert len(tree["clauses"]) >= 7, f"Expected >= 7 clauses, got {len(tree['clauses'])}"
    print(f"Parser OK: {len(tree['chapters'])} Chương, {len(tree['articles'])} Điều, {len(tree['clauses'])} Khoản.")

    # 2. Test Chunker
    print("\n[TEST 2] Kiểm tra Hierarchical Chunker...")
    chunker = HierarchicalLegalChunker()
    chunks = chunker.create_chunks(tree)
    assert len(chunks) > 0, "No chunks generated"
    
    # Check parent_chunk_id
    child_chunks = [c for c in chunks if c["parent_chunk_id"] is not None]
    assert len(child_chunks) > 0, "Child chunks should have parent_chunk_id"
    print(f"Chunker OK: Đã tạo {len(chunks)} chunks, {len(child_chunks)} chunks có liên kết parent_chunk_id.")

    # 3. Test PostgreSQL Storage
    print("\n[TEST 3] Kiểm tra PostgreSQL Database Manager & DDL...")
    pg_mgr = PostgresManager()
    pg_mgr.reset_db()
    inserted = pg_mgr.insert_legal_tree(tree, chunks)
    assert inserted is True, "Failed to insert legal tree into PostgreSQL"
    
    stats = pg_mgr.get_database_stats()
    assert stats["articles"] == 3, f"Expected 3 articles in DB, got {stats['articles']}"
    assert stats["chunks"] == len(chunks), f"Expected {len(chunks)} chunks in DB, got {stats['chunks']}"
    print(f"  ✅ PostgreSQL Storage OK: Đã lưu trọn vẹn 7 bảng quan hệ ({stats}).")

    # 4. Test Context Expansion via PostgreSQL
    print("\n[TEST 4] Kiểm tra PostgreSQL Context Expansion...")
    sample_chunk_ids = [chunks[0]["id"], chunks[-1]["id"]]
    rich_ctx = pg_mgr.get_hierarchical_context_by_chunk_ids(sample_chunk_ids)
    assert len(rich_ctx) == len(sample_chunk_ids), "Context expansion mismatch"
    assert "article" in rich_ctx[0] and "document" in rich_ctx[0]
    print(f"  ✅ PostgreSQL Context Expansion OK: Truy xuất thành công thông tin Điều {rich_ctx[0]['article']['article_number']}.")

    # 5. Test Embedder & Qdrant Upsert (chỉ embed is_embeddable=True)
    print("\n[TEST 5] Kiểm tra Token Control, Embedder & Qdrant Upsert...")
    embedder = LlamaIndexEmbedder()
    from src.storage.indexer import process_chunks_token_control
    controlled_chunks = process_chunks_token_control(chunks, tokenizer=embedder.tokenizer)
    embeddable_chunks = [c for c in controlled_chunks if c.get("is_embeddable", True)]

    sample_texts = [c["content"] for c in embeddable_chunks]
    embs = embedder.get_document_embeddings(sample_texts, batch_size=8)
    assert embs.shape[0] == len(embeddable_chunks), "Embeddings count mismatch"
    
    qdrant_mgr = QdrantManager()
    qdrant_mgr.upsert_chunks(embeddable_chunks, embs)
    print(f"  ✅ Qdrant Upsert OK: Đã nạp {len(embeddable_chunks)} vectors (dim={embs.shape[1]}) cho các chunk embeddable.")

    # 6. Test Qdrant Filtered Search
    print("\n[TEST 6] Kiểm tra Qdrant Filtered Search...")
    query = "Nồng độ cồn và chất kích thích khi lái xe"
    query_vec = embedder.get_query_embedding(query)
    hits = qdrant_mgr.search_candidates(query_vec, top_k=5, status_filter="active", is_current_only=True)
    assert len(hits) > 0, "No hits returned from Qdrant search"
    print(f"  ✅ Qdrant Filter Search OK: Tìm thấy {len(hits)} ứng viên hàng đầu.")

    # 7. Test Query Engine & Reranker End-to-End
    print("\n[TEST 7] Kiểm tra Query Engine & Reranker...")
    query_engine = LegalQueryEngine(
        embedder=embedder,
        qdrant_mgr=qdrant_mgr,
        postgres_mgr=pg_mgr,
    )
    results = query_engine.retrieve(query, top_k=10, top_n=2)
    assert len(results) > 0, "Retrieve returned empty"
    print(f"  ✅ Query Engine & Reranker OK: Lấy được {len(results)} contexts sau khi xếp hạng lại.")
    for idx, r in enumerate(results, 1):
        print(f"     Top {idx}: Điều {r.get('article_number')} - {r.get('article_title')} (Score: {r.get('rerank_score', r.get('initial_score', 0.0)):.4f})")

    # 8. Test Citation Builder
    print("\n[TEST 8] Kiểm tra Deterministic Citation Builder...")
    llm_context = CitationBuilder.build_llm_context(results)
    citations = CitationBuilder.generate_citations(results)
    assert len(citations) == len(results), "Citations count mismatch"
    assert "CĂN CỨ PHÁP LÝ" in llm_context, "Malformed LLM context"
    print(f"  ✅ Citation Builder OK: Đã tạo {len(citations)} trích dẫn xác thực.")
    for c in citations:
        print(f"     📌 {c['citation_text']}")

    print("\n" + "=" * 70)
    print("🎉 TẤT CẢ 8 BƯỚC KIỂM THỬ ĐÃ VƯỢT QUA THÀNH CÔNG (100% PASSED)!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
