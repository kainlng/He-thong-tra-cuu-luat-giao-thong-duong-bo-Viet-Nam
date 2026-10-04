# Changelog

---

## [Unreleased]

### Added
- **Bổ sung Kho Tài liệu Pháp luật & Bóc tách Hàng loạt (Batch Ingestion)**:
  - Thêm mới **44 tệp PDF** văn bản quy phạm pháp luật giao thông đường bộ vào thư mục `data/raw/`.
  - OCR của các trang scan không thực hiện thành công do giới hạn max_content_size; pipeline sau đó giữ native extraction result, dẫn tới trang scan không có hoặc có rất ít text.
  - **Biện pháp xử lý**:
    - Bổ sung tham số `XBERG_MAX_CONTENT_SIZE` vào [`src/core/config.py`](file:///e:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/config.py) mặc định `1 * 1024 * 1024 * 1024` (1 GB).
    - Cấu hình `SecurityLimits(max_content_size=self.max_content_size, max_pages=None)` đồng bộ vào cả `OcrConfig` và `ExtractionConfig` trong [`XbergProcessor`](file:///e:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/xberg_processor.py).
      - Cập nhật trường `max_content_size` vào `get_config_dict()`: `config_hash` tự động thay đổi từ `cfg:5709869b9249` sang `cfg:2b7b1de02420`, giúp `IngestionManifest` nhận diện cấu hình trích xuất đã được nâng cấp và tự động bóc tách lại đầy đủ các tài liệu mà không cần can thiệp thủ công.
- **Module Tiện ích Toàn cục (`src/core/utils.py`)**:
  - Hợp nhất các hàm tính mã băm SHA-256 trùng lặp (`calculate_pdf_hash` trong `data_loader.py` và `calculate_file_hash` trong `legal_parser.py`) thành hàm tiện ích duy nhất [`calculate_file_sha256`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/utils.py#L6-L23) hỗ trợ đọc bộ đệm nhị phân 64KB (`chunk_size=65536`).
  - Cung cấp bí danh tương thích ngược (Backward compatibility aliases) cho `calculate_file_hash` và `calculate_pdf_hash`.
  - Bổ sung bộ kiểm thử tự động `test_calculate_file_sha256` trong [`tests/test_legal_parser.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/tests/test_legal_parser.py).

### Kế hoạch phát triển
- Triển khai đánh giá tự động định lượng RAG Triad (Faithfulness, Answer Relevance, Context Recall) qua thư viện TruLens/Ragas tại thư mục `benchmark/RAGEval`.
- Hỗ trợ lưu vết lịch sử hội thoại nhiều phiên vào PostgreSQL thay vì chỉ quản lý cục bộ trong Streamlit `st.session_state`.

---

## [2.0.0] - 2026-10-04

### Added
- **Kiến trúc Phân cấp Legal RAG v2.0**:
  - **Hierarchical Legal Parser v2.0 (`src/ingestion/legal_parser.py`)**:
    - Chuyển từ regex phân mảnh sang compiled header patterns (`clause_header_pattern`, `point_header_pattern`) kết hợp `re.finditer` và boundary slicing.
    - Xử lý triệt để Anomaly Orphan Point (Điều chứa trực tiếp các Điểm mà không có Khoản đánh số) bằng synthetic container node (`clause_number: None`, `is_synthetic: True`), không tự bịa "Khoản 1".
    - Trích xuất ngày ban hành và ngày có hiệu lực thực tế từ công thức ban hành hoặc điều khoản hiệu lực, không fallback về ngày giả định (`None` khi không tìm thấy).
    - Cross-check cơ quan ban hành giữa số hiệu, tiêu ngữ và tên file, trả về kèm `source` và `confidence`.
    - Bổ sung hàm kiểm định toàn vẹn cây `validate_parsed_tree()`.
  - **Hierarchical Legal Chunker v2.0 (`src/ingestion/hierarchical_chunker.py`)**:
    - Duy trì cấu trúc cây phân cấp 3 tầng: $\text{ArticleChunk} \xrightarrow{\text{parent\_chunk\_id}} \text{ClauseChunk} \xrightarrow{\text{parent\_chunk\_id}} \text{PointChunk}$.
    - **Invariable 1**: Khi Khoản có Điểm con, `ClauseChunk.is_embeddable = False` (chỉ đóng vai trò context cha trên PostgreSQL). `PointChunk` kế thừa trọn vẹn ngữ cảnh đoạn mở đầu của Khoản (`clause_intro`) vào nội dung embedding.
    - Bổ sung thuộc tính `root_article_chunk_id` trỏ thẳng về Điều luật gốc cho mọi chunk cháu chắt.
    - Bổ sung hàm kiểm định đồ thị chunk `validate_chunk_tree()`.
  - **Kiểm soát Token bằng Tokenizer Thật & Subchunking (`src/storage/indexer.py`)**:
    - **Invariable 2**: Sử dụng Rust Fast Tokenizer từ `tokenizers.Tokenizer` nạp trực tiếp `intfloat/multilingual-e5-base` từ local cache.
    - Tính toán token count trên chuỗi embedding hoàn chỉnh thực tế (`passage: [Luật...] [Điều...] [Khoản...] [Điểm...] {content}`), áp dụng trần cứng `HARD_MAX_TOKENS = 450` và mục tiêu `TARGET_TOKENS = 350`.
    - Phân rã đa tầng (dấu câu lớn `[;.\n]` $\rightarrow$ dấu phẩy `,` $\rightarrow$ từ ngữ) xử lý các câu phạt dài không có dấu chấm.
    - **Invariable 3**: Subchunk lưu trữ đầy đủ `source_chunk_id = original_chunk_id`, `subchunk_index`, và `parent_chunk_id = original_chunk_id`, đảm bảo khả năng truy ngược từ $\text{Subchunk} \rightarrow \text{Point} \rightarrow \text{Clause} \rightarrow \text{Article}$.
  - **Cơ chế Lưu trữ Tô-pô & Đồng bộ Qdrant Payload Filter (`src/storage/`)**:
    - `PostgresManager.insert_legal_tree()` chèn dữ liệu theo 4 tầng tô-pô độc lập (Article $\rightarrow$ Clause $\rightarrow$ Point $\rightarrow$ Subchunk), triệt tiêu lỗi vi phạm khóa ngoại `parent_chunk_id`.
    - `QdrantManager` chỉ nạp các chunk `is_embeddable = True` và lưu trữ metadata payload tối giản cho pre-filtering. Bổ sung `delete_points_by_document_id()` loại bỏ orphan vectors khi nạp lại văn bản.
  - **Tái Cấu Trúc XbergProcessor sang Native `scanned_pages` (`src/ingestion/xberg_processor.py`)**:
    - Chuyển `XBERG_OCR_STRATEGY` từ `"auto"` sang `"scanned_pages"` với `XBERG_OCR_MIN_CONFIDENCE = 0.5`.
    - Bổ sung hàm `resolve_ocr_strategy()` ưu tiên trực tiếp native binding `_rust.OcrStrategy.scanned_pages(min_confidence=...)` của phiên bản Xberg đang cài đặt.
    - Bổ sung hàm `diagnose_scanned_pages()` sử dụng `mypdf` để phục vụ benchmark/chẩn đoán ngoài vòng chính.
    - Chuẩn hóa provenance metadata phân tách rành mạch:
      `{"source_type": "pdf", "extraction_backend": "xberg", "ocr_backend": "tesseract", "ocr_strategy": "scanned_pages"}`.
- **Nạp Toàn Bộ Corpus Pháp Luật Giao Thông Đường Bộ**:
  - Nạp trọn vẹn 11 văn bản pháp luật chính yếu vào PostgreSQL (5.352 chunks, 430 Điều, 2.138 Khoản, 2.586 Điểm) và Qdrant collection `legal_chunks` (5.349 vectors) với 0 lỗi.
- **Bộ Kiểm Thử & Fixture Mới**:
  - `tests/fixtures/`: 8 tệp golden fixture mô phỏng toàn bộ góc cạnh pháp lý (bình thường, khoản có điểm, chỉ có điều, điểm mồ côi, ngày thiếu, OCR nhiễu, điểm siêu dài).
  - `tests/test_parser_fixtures.py`: Kiểm thử parser trên toàn bộ fixtures.
  - `tests/test_chunker_hierarchy.py`: Kiểm thử quan hệ phả hệ chunk, tính thừa kế ngữ cảnh và cờ `is_embeddable`.
  - `tests/test_token_control.py`: Kiểm thử bộ đếm token và subchunking.
  - `tests/test_retrieval_benchmarks.py`: Kiểm thử truy vấn thực tế 5 kịch bản pháp lý.
  - `tests/test_xberg_processor.py`: Kiểm thử 7 trường hợp trích xuất Xberg hybrid/scan/text và provenance.
  - `tests/benchmark_detector.py`: So sánh chẩn đoán heuristic pypdf vs Xberg scanned_pages trên 12 tệp PDF.

### Fixed
- Khắc phục lỗi `zfill` crash trong [`HierarchicalContextExpander`](file:///e:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/retrieval/context_expander.py) khi Điều luật chứa Khoản không đánh số (`clause_number is None`).
- Khắc phục lỗi thiếu điểm phụ thuộc do so khớp `clause_number` lỏng lẻo; chuyển sang so khớp chính xác bằng khóa ngoại `clause_id == cl.id`.
- Khắc phục lỗi `ValueError: Invalid OcrStrategy: missing field min_confidence` khi truyền chuỗi `"scanned_pages"` vào Xberg binding.
- Dọn dẹp tệp nhị phân `vector_db/` khỏi Git tracking theo đúng quy tắc `.gitignore`.

---

## [1.0.0] - 2026-09-21

### Added
- **Tích hợp bộ bóc tách tài liệu XBerg kết hợp Tesseract OCR**:
  - Bổ sung module [`XbergProcessor`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/xberg_processor.py#L86-L179) hỗ trợ trích xuất văn bản PDF tự động với OCR tiếng Việt (`vie.traineddata`), cho phép xử lý chất lượng cao các văn bản scan công báo.
  - Bổ sung hàm [`normalize_xberg_markdown`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/xberg_processor.py#L24-L84) chuẩn hóa Unicode NFC, phân tách tiêu đề Chương/Điều bị dính dòng, gỡ bỏ tiền tố định dạng list Markdown (`- 1\.` $\rightarrow$ `1.`, `- a)` $\rightarrow$ `a)`).
  - Tích hợp cờ cấu hình `--force-ocr` và `--reprocess` vào [`data_loader.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/data_loader.py#L255-L262) và [`ingest_pipeline.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/ingest_pipeline.py#L111-L145).
- **Hệ thống lưu vết bộ nhớ đệm Ingestion Manifest**:
  - Lớp [`IngestionManifest`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/data_loader.py#L47-L122) lưu vết mã băm tệp PDF (SHA-256) và mã băm cấu hình trích xuất (Config Hash) vào `.data_loader_manifest.json`, hỗ trợ cơ chế phục hồi và bỏ qua tệp đã xử lý (Resume & Skip).
- **Đồng bộ hóa tham số cấu hình hệ thống**:
  - Mở rộng [`config.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/config.py#L27-L47) hỗ trợ tự động dò đường dẫn `tessdata` (qua biến `TESSDATA_PREFIX`, thư mục dự án `data/tessdata`, hoặc cài đặt Windows mặc định).

### Changed
- Cập nhật [`HierarchicalLegalParser`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/legal_parser.py#L20-L506) để bảo toàn và bóc tách chính xác vị trí đánh dấu trang `<!-- PAGE {n} -->`, cho phép truy vết chính xác số trang tài liệu gốc cho từng Điều, Khoản, Điểm.
- Điều chỉnh Dockerfile và `docker-compose.yml` để cài đặt Tesseract OCR và các gói ngôn ngữ tiếng Việt cần thiết cho container ứng dụng.

---

## [0.9.0] - 2026-09-14

### Added
- **Quy tắc phân loại văn bản cấu hình hóa**:
  - Module [`legal_rules.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/legal_rules.py#L5-L94) cấu hình hóa danh mục cơ quan ban hành (`LEGAL_AGENCIES_CONFIG`): Quốc hội, UBTVQH, Chính phủ, Thủ tướng Chính phủ, Bộ Giao thông vận tải, Bộ Công an, Bộ Tài chính, Bộ Y tế, Bộ Xây dựng.
  - Quy tắc phân loại định dạng regex nhận diện hậu tố số hiệu (như `QH\d*`, `NĐ-CP`, `TT-BCA`), mẫu tên tệp và từ khóa tiêu ngữ/chữ ký.
- **Bộ kiểm thử tự động cho Parser**:
  - Tạo tệp kiểm thử [`test_legal_parser.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/tests/test_legal_parser.py) xác thực khả năng nhận diện tự động cho văn bản Luật, Nghị định và Thông tư.

### Fixed
- Xóa bỏ hoàn toàn tình trạng mã hóa cứng (hardcoded) cơ quan ban hành và loại văn bản trong [`legal_parser.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/legal_parser.py#L50-L72).
- Khắc phục lỗi trích xuất ngày ban hành và ngày có hiệu lực khi văn bản sử dụng định dạng ngày cuối tài liệu.

---

## [0.8.0] - 2026-09-03

### Changed
- **Nâng cấp kết nối cơ sở dữ liệu và máy chủ**:
  - Cải tiến [`PostgresManager`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/storage/postgres_manager.py#L32-L86) và [`QdrantManager`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/storage/qdrant_manager.py#L30-L76) với cơ chế tự động phát hiện lỗi mạng và kết nối dự phòng (Connection pooling, fallback gracefully sang local SQLite và Qdrant local).
  - Tối ưu hóa giao diện Streamlit [`app.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/app.py): bảng điều khiển trạng thái kết nối trực tiếp (Live Status Indicator) báo hiệu trạng thái của PostgreSQL và Qdrant.

### Fixed
- Sửa lỗi ngắt kết nối socket khi PostgreSQL hoặc Qdrant khởi động chậm hơn container ứng dụng.

---

## [0.7.0] - 2026-08-17

### Added
- **Quản lý đa khóa API LlamaCloud**:
  - Bổ sung hàm [`get_llama_cloud_api_keys`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/config.py#L54-L95) trong file cấu hình, hỗ trợ đọc danh sách khóa API từ các biến môi trường được đánh số (`LLAMA_CLOUD_API_KEY_1`, `LLAMA_CLOUD_API_KEY_2`...) hoặc chuỗi phân cách dấu phẩy.
- Cập nhật quy tắc `.gitignore` loại trừ các thư mục cache bóc tách, dữ liệu nhị phân trung gian và môi trường ảo.

### Changed
- Chuyển đổi công cụ bóc tách PDF đám mây từ LlamaParse sang LlamaCloud client API với cơ chế quay vòng khóa (Key Rotation).

---

## [0.6.0] - 2026-08-16

### Added
- **Tái cấu trúc mã nguồn theo mô hình phân tầng mô-đun (Domain-Driven Modular Structure)**:
  - Phân tách toàn bộ mã nguồn `src/` thành 6 phân hệ chuyên trách:
    - `src/core/`: Quản lý cấu hình ([`config.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/config.py)), quy tắc pháp lý ([`legal_rules.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/legal_rules.py)), lược đồ dữ liệu ([`models.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/models.py)), tiện ích dùng chung ([`utils.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/utils.py)).
    - `src/ingestion/`: Pipeline bóc tách PDF, tiền xử lý, bóc tách cấu trúc cây và chunking phân cấp.
    - `src/storage/`: Quản lý lưu trữ quan hệ PostgreSQL và cơ sở dữ liệu vector Qdrant.
    - `src/retrieval/`: Bộ máy truy vấn ý định thời gian, mở rộng ngữ cảnh Điều luật và xếp hạng lại Reranking.
    - `src/generation/`: Agent RAG, xây dựng trích dẫn xác thực và module kiểm chứng Điều luật.
    - `src/visualization/`: Chiếu không gian vector 2D/3D tương tác.
- **Hiện đại hóa giao diện người dùng Streamlit (`app.py`)**:
  - Thiết kế hệ thống giao diện trắng tối giản cảm hứng ChatGPT với phông chữ Plus Jakarta Sans.
  - Bổ sung thanh Accordion hiển thị căn cứ pháp lý, hiệu lực văn bản, điểm số tương quan và trích dẫn số trang gốc.
  - Tích hợp bộ công cụ trực quan hóa Vector Embeddings 2D/3D tương tác sử dụng PCA, t-SNE và phân cụm KMeans với Plotly.

---

## [0.5.0] - 2026-08-16

### Added
- **Kiến trúc Hierarchical Legal RAG đột phá**:
  - Triển khai mô hình cơ sở dữ liệu quan hệ 7 bảng PostgreSQL: `legal_documents`, `legal_versions`, `legal_chapters`, `legal_articles`, `legal_clauses`, `legal_points`, `legal_chunks` thông qua SQLAlchemy ORM ([`models.py`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/core/models.py#L30-L195)).
  - Xây dựng [`HierarchicalLegalParser`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/legal_parser.py#L20-L506) bóc tách văn bản pháp luật Việt Nam thành cấu trúc cây phân cấp hoàn chỉnh.
  - Xây dựng [`HierarchicalLegalChunker`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/ingestion/hierarchical_chunker.py#L55-L269) gắn nhãn `parent_chunk_id` và bổ sung tiền tố ngữ cảnh phân cấp (Breadcrumbs) vào từng đoạn chunk.
  - Xây dựng [`HierarchicalContextExpander`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/retrieval/context_expander.py#L5-L80) tự động tái hợp các chunk con về Điều luật gốc từ PostgreSQL trước khi gửi cho LLM, giải quyết triệt để vấn đề mất mát ngữ cảnh.
- **Tích hợp cơ sở dữ liệu Vector Qdrant**:
  - Lưu trữ vector 768 chiều với khoảng cách Cosine và chỉ mục Payload Filter phục vụ lọc trước theo trạng thái hiệu lực (`status`), cờ văn bản hiện hành (`is_current`), chủ đề vi phạm.
- **Mô hình Reranking hai tầng**:
  - Tích hợp FlashRank (`ms-marco-TinyBERT-L-2-v2`) tối ưu hóa CPU và fallback sang CrossEncoder (`BAAI/bge-reranker-base`).
- **Module trích dẫn xác thực và tự kiểm chứng (Self-Verification Loop)**:
  - [`CitationBuilder`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/generation/citation_builder.py#L4-L68) tạo trích dẫn pháp lý tất định (Deterministic Citations).
  - [`verify_answer`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/generation/agent.py#L95-L125) đối soát số Điều luật trong câu trả lời sinh ra từ LLM với dữ liệu căn cứ thực tế từ PostgreSQL; kích hoạt vòng gọi lại tự sửa sai (Retry Loop) nếu phát hiện hiện tượng ảo giác.

### Removed
- Xóa bỏ thư viện chỉ mục FAISS tĩnh (`faiss_index.bin`) và bộ lưu trữ SQLite FTS5 đơn giản (`legal_fts.db`).

---

## [0.4.0] - 2026-08-15

### Added
- Chuyển đổi sang hệ sinh thái LlamaIndex:
  - Tích hợp [`LlamaIndexEmbedder`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/storage/indexer.py#L28-L87) sử dụng gói `llama-index-embeddings-huggingface` với tiền tố `query: ` và `passage: ` chuẩn cho mô hình `intfloat/multilingual-e5-base`.
- Bổ sung kịch bản kiểm tra vector `inspect_vector_db.py`.

### Changed
- Cập nhật quy trình bóc tách sang Markdown theo chuẩn LlamaIndex.

---

## [0.3.0] - 2026-08-12

### Added
- **Đóng gói container hóa đa nền tảng**:
  - Tạo `Dockerfile` tối ưu hóa môi trường Python 3.13 và PyTorch CPU/CUDA.
  - Cấu hình `docker-compose.yml` liên kết ứng dụng với mạng và phân vùng dữ liệu lưu trữ bền vững.
- **Trực quan hóa không gian Vector**:
  - Module [`VectorVisualizer`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/visualization/vector_visualizer.py#L19-L340) hỗ trợ giảm chiều dữ liệu bằng PCA (2D/3D), t-SNE và phân cụm KMeans để phân tích cụm ngữ nghĩa của các điều luật giao thông.

---

## [0.2.0] - 2026-08-06

### Added
- **Tích hợp mô hình ngôn ngữ lớn tiên tiến**:
  - Kết nối mô hình `Qwen/Qwen2.5-72B-Instruct` qua Hugging Face Inference Client API.
  - Xây dựng lớp tác tử cơ bản [`LegalRAGAgent`](file:///E:/Python/He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam/src/generation/agent.py#L52-L198) điều phối truy vấn người dùng, nạp ngữ cảnh và gọi mô hình trả lời.
- **Tìm kiếm kết hợp từ vựng SQLite FTS5**:
  - Bổ sung chỉ mục từ khóa toàn văn SQLite FTS5 nhằm kiểm tra chéo (Double-check) kết quả tìm kiếm ngữ nghĩa dense retrieval.

---

## [0.1.0] - 2026-07-28

### Added
- Khởi tạo kho lưu trữ dự án tra cứu luật giao thông đường bộ Việt Nam.
- Thử nghiệm và đánh giá so sánh hiệu năng các mô hình trích xuất tài liệu PDF pháp luật tiếng Việt giữa Docling và Marker.
- Tạo cấu trúc dữ liệu thô ban đầu `data/raw/` chứa văn bản Luật Trật tự, an toàn giao thông đường bộ 2024 (Luật số 36/2024/QH15).
