import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from src.core.models import LegalChunkDTO, DocumentMetadata, ValidityMetadata, SourceMetadata, QdrantChunkPayload

CHUNKER_VERSION = "2.0.0"


def remove_vietnamese_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để phục vụ normalize search & indexing."""
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = text.replace("đ", "d").replace("Đ", "D")
    return text.lower().strip()


def extract_legal_topics_and_violations(text: str) -> Tuple[List[str], List[str]]:
    """Tự động phân loại topic và loại hành vi vi phạm từ nội dung điều khoản."""
    text_lower = text.lower()
    topics = []
    violations = []

    topic_keywords = {
        "nong_do_con": ["nồng độ cồn", "rượu", "bia", "chất kích thích", "ma túy"],
        "toc_do": ["tốc độ", "chạy quá tốc độ", "giới hạn tốc độ"],
        "den_tin_hieu": ["đèn đỏ", "đèn vàng", "tín hiệu đèn", "chấp hành tín hiệu"],
        "giay_to_xe": ["giấy phép lái xe", "bằng lái", "đăng ký xe", "đăng kiểm", "bảo hiểm"],
        "mu_bao_hiem": ["mũ bảo hiểm", "cài quai"],
        "duong_cao_toc": ["đường cao tốc", "cao tốc", "làn dừng khẩn cấp"],
        "xe_mo_to": ["xe mô tô", "xe gắn máy", "xe máy", "hai bánh"],
        "xe_o_to": ["xe ô tô", "ô tô tải", "xe khách", "xe container"],
        "nguoi_di_bo": ["người đi bộ", "qua đường"],
        "quy_tac_chung": ["quy tắc chung", "nguyên tắc", "phạm vi điều chỉnh"],
    }

    for topic, kws in topic_keywords.items():
        if any(kw in text_lower for kw in kws):
            topics.append(topic)

    violation_keywords = {
        "vuot_den_do": ["vượt đèn đỏ", "không chấp hành hiệu lệnh của đèn tín hiệu"],
        "vi_pham_nong_do_con": ["trong máu hoặc hơi thở có nồng độ cồn", "nồng độ cồn"],
        "chay_qua_toc_do": ["chạy quá tốc độ", "vượt quá tốc độ"],
        "di_nguoc_chieu": ["đi ngược chiều", "lùi xe trên đường cao tốc"],
        "khong_doi_mu_bao_hiem": ["không đội mũ bảo hiểm", "không cài quai"],
        "su_dung_dien_thoai": ["sử dụng điện thoại", "dùng tay cầm điện thoại"],
        "khong_co_bang_lai": ["không có giấy phép lái xe", "không có bằng lái"],
    }

    for viol, kws in violation_keywords.items():
        if any(kw in text_lower for kw in kws):
            violations.append(viol)

    return topics, violations


def validate_chunk_tree(chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Xác thực tính toàn vẹn của đồ thị phân cấp Chunks:
    - [x] ArticleChunk: parent_chunk_id = None
    - [x] ClauseChunk: parent_chunk_id trỏ tới ArticleChunk hợp lệ
    - [x] PointChunk: parent_chunk_id trỏ tới ClauseChunk hợp lệ
    - [x] Mọi parent_chunk_id không None phải tồn tại trong danh sách
    - [x] is_embeddable được gán đúng
    """
    errors: List[str] = []
    chunk_map = {c["id"]: c for c in chunks}

    if len(chunk_map) != len(chunks):
        errors.append("Phát hiện duplicate chunk IDs trong danh sách chunks.")

    for c in chunks:
        cid = c["id"]
        ctype = c.get("chunk_type")
        pid = c.get("parent_chunk_id")
        rid = c.get("root_article_chunk_id")

        if ctype == "article":
            if pid is not None:
                errors.append(f"ArticleChunk '{cid}' phải có parent_chunk_id = None, nhận: '{pid}'")
            if c.get("is_embeddable") is True:
                errors.append(f"ArticleChunk '{cid}' mặc định không nên là embeddable.")

        elif ctype == "clause":
            if pid is None or pid not in chunk_map:
                errors.append(f"ClauseChunk '{cid}' có parent_chunk_id không hợp lệ: '{pid}'")
            elif chunk_map[pid].get("chunk_type") != "article":
                errors.append(f"ClauseChunk '{cid}' phải có parent_chunk_id là ArticleChunk, nhận: '{chunk_map[pid].get('chunk_type')}'")

        elif ctype == "point":
            if pid is None or pid not in chunk_map:
                errors.append(f"PointChunk '{cid}' có parent_chunk_id không hợp lệ: '{pid}'")
            elif chunk_map[pid].get("chunk_type") != "clause":
                errors.append(f"PointChunk '{cid}' phải có parent_chunk_id là ClauseChunk, nhận: '{chunk_map[pid].get('chunk_type')}'")

        elif ctype == "subchunk":
            if pid is None or pid not in chunk_map:
                errors.append(f"Subchunk '{cid}' có parent_chunk_id không hợp lệ: '{pid}'")

    is_valid = len(errors) == 0
    return is_valid, errors


class HierarchicalLegalChunker:
    """Tạo các Chunks phân cấp chuẩn theo mô hình cây:
    ArticleChunk (parent=None)
       ↓
    ClauseChunk (parent=ArticleChunk)
       ↓
    PointChunk (parent=ClauseChunk)
    
    Quy tắc nghiệp vụ:
    - Luôn tạo ClauseChunk cho tất cả các Khoản (kể cả khi có Điểm).
    - Khi Khoản có Điểm: ClauseChunk.is_embeddable = False (chỉ làm context cha).
    - PointChunk luôn kế thừa ngữ cảnh Clause (`clause_intro`) và trỏ parent_chunk_id về ClauseChunk.
    """

    def __init__(self):
        self.version = CHUNKER_VERSION

    def create_chunks(self, parsed_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = parsed_tree["document"]
        ver = parsed_tree["version"]
        articles = parsed_tree["articles"]
        clauses = parsed_tree["clauses"]
        points = parsed_tree["points"]

        doc_id = doc["id"]
        ver_id = ver["id"]
        doc_number = doc["document_number"]
        doc_title = doc["short_title"] or doc["title"]
        file_name = doc["source"].get("file_name", "")
        doc_source = doc.get("source", {})

        # Gom nhóm clauses theo article_id và points theo clause_id
        clauses_by_art = {}
        for c in clauses:
            clauses_by_art.setdefault(c["article_id"], []).append(c)

        points_by_clause = {}
        for p in points:
            points_by_clause.setdefault(p["clause_id"], []).append(p)

        all_chunks = []
        seen_chunk_ids = set()
        global_chunk_idx = 1

        def make_unique_id(base_id: str) -> str:
            candidate = base_id
            idx = 2
            while candidate in seen_chunk_ids:
                candidate = f"{base_id}_{idx}"
                idx += 1
            seen_chunk_ids.add(candidate)
            return candidate

        for art in articles:
            art_id = art["id"]
            art_num = art["article_number"]
            art_title = art["title"]
            chap_id = art.get("chapter_id")
            art_clauses = clauses_by_art.get(art_id, [])

            # 1. Tạo Root Chunk cho Điều luật (Article Parent Chunk)
            raw_art_chunk_id = f"chunk_{doc_id.replace('doc_', '')}_{art_num.zfill(2)}_root"
            art_chunk_id = make_unique_id(raw_art_chunk_id)
            art_topics, art_violations = extract_legal_topics_and_violations(art.get("raw_text", ""))

            art_chunk_content = f"[{doc_title}] [Điều {art_num}: {art_title}]\n{art.get('raw_text', '')}"

            art_chunk = {
                "id": art_chunk_id,
                "document_id": doc_id,
                "version_id": ver_id,
                "chapter_id": chap_id,
                "article_id": art_id,
                "clause_id": None,
                "point_id": None,
                "parent_chunk_id": None,  # Root của cây Điều
                "root_article_chunk_id": art_chunk_id,
                "source_chunk_id": None,
                "subchunk_index": None,
                "is_embeddable": False,  # ArticleChunk chỉ lưu PostgreSQL làm ngữ cảnh đầy đủ
                "chunk_type": "article",
                "chunk_index": global_chunk_idx,
                "content": art_chunk_content,
                "normalized_content": remove_vietnamese_accents(art_chunk_content),
                "token_count": len(art_chunk_content.split()),
                "metadata": {
                    "document_number": doc_number,
                    "article_number": art_num,
                    "clause_number": None,
                    "point_label": None,
                    "document_type": doc.get("document_type", "luat"),
                    "domain": "giao_thong_duong_bo",
                    "topics": art_topics,
                    "violation_types": art_violations,
                },
                "validity": {
                    "effective_from": ver.get("effective_from"),
                    "effective_to": ver.get("effective_to"),
                    "status": ver.get("status", "active"),
                    "is_current": ver.get("is_current", True),
                },
                "source": {
                    "file_name": file_name,
                    "page_start": art.get("page_start", 1),
                    "page_end": art.get("page_end", 1),
                    "source_type": doc_source.get("source_type", "pdf"),
                    "ocr_engine": doc_source.get("ocr_engine", "tesseract"),
                    "ocr_backend": doc_source.get("ocr_backend", "xberg"),
                },
                "search": {
                    "qdrant_collection": "legal_chunks_v2",
                    "qdrant_point_id": art_chunk_id,
                },
            }
            all_chunks.append(art_chunk)
            global_chunk_idx += 1

            # 2. Xử lý từng Khoản trong Điều
            for clause in art_clauses:
                cl_id = clause["id"]
                cl_num = clause.get("clause_number")
                cl_intro = clause.get("clause_intro")
                cl_full_text = clause.get("clause_full_text", clause.get("content", ""))
                cl_points = points_by_clause.get(cl_id, [])

                num_part = str(cl_num).zfill(2) if cl_num else "body"
                raw_cl_chunk_id = f"chunk_{doc_id.replace('doc_', '')}_{art_num.zfill(2)}_{num_part}"
                cl_chunk_id = make_unique_id(raw_cl_chunk_id)
                cl_topics, cl_violations = extract_legal_topics_and_violations(cl_full_text)

                has_points = len(cl_points) > 0

                # Chuẩn bị nội dung cho Khoản
                cl_label_part = f" [Khoản {cl_num}]" if cl_num else ""
                cl_text = f"[{doc_title}] [Điều {art_num}: {art_title}]{cl_label_part}\n{cl_full_text}"

                # Khi Clause có Point: is_embeddable = False (theo Quy tắc 1)
                # Khi Clause không có Point (là lá): is_embeddable = True
                cl_chunk = {
                    "id": cl_chunk_id,
                    "document_id": doc_id,
                    "version_id": ver_id,
                    "chapter_id": chap_id,
                    "article_id": art_id,
                    "clause_id": cl_id,
                    "point_id": None,
                    "parent_chunk_id": art_chunk_id,  # Direct parent = ArticleChunk
                    "root_article_chunk_id": art_chunk_id,
                    "source_chunk_id": None,
                    "subchunk_index": None,
                    "is_embeddable": not has_points,  # False nếu có Điểm con, True nếu là Leaf
                    "chunk_type": "clause",
                    "chunk_index": global_chunk_idx,
                    "content": cl_text,
                    "normalized_content": remove_vietnamese_accents(cl_text),
                    "token_count": len(cl_text.split()),
                    "metadata": {
                        "document_number": doc_number,
                        "article_number": art_num,
                        "clause_number": cl_num,
                        "point_label": None,
                        "document_type": doc.get("document_type", "luat"),
                        "domain": "giao_thong_duong_bo",
                        "topics": cl_topics or art_topics,
                        "violation_types": cl_violations or art_violations,
                    },
                    "validity": art_chunk["validity"],
                    "source": {
                        "file_name": file_name,
                        "page_start": clause.get("page_start", art.get("page_start", 1)),
                        "page_end": clause.get("page_end", art.get("page_end", 1)),
                        "source_type": doc_source.get("source_type", "pdf"),
                        "ocr_engine": doc_source.get("ocr_engine", "tesseract"),
                        "ocr_backend": doc_source.get("ocr_backend", "xberg"),
                    },
                    "search": {
                        "qdrant_collection": "legal_chunks_v2",
                        "qdrant_point_id": cl_chunk_id,
                    },
                }
                all_chunks.append(cl_chunk)
                global_chunk_idx += 1

                # 3. Xử lý từng Điểm trong Khoản
                for pt in cl_points:
                    pt_id = pt["id"]
                    pt_label = pt["point_label"]
                    pt_content = pt["content"]

                    raw_pt_chunk_id = f"chunk_{doc_id.replace('doc_', '')}_{art_num.zfill(2)}_{num_part}_{pt_label}"
                    pt_chunk_id = make_unique_id(raw_pt_chunk_id)

                    # Quy tắc 1: Point embedding PHẢI mang Clause context khi Clause có Point
                    # Gắn cl_intro (đoạn dẫn của Khoản) vào tiêu đề Point
                    if cl_num and cl_intro:
                        pt_header = f"[{doc_title}] [Điều {art_num}: {art_title}] [Khoản {cl_num}: {cl_intro}] [Điểm {pt_label}]"
                    elif cl_num:
                        pt_header = f"[{doc_title}] [Điều {art_num}: {art_title}] [Khoản {cl_num}] [Điểm {pt_label}]"
                    elif cl_intro:
                        pt_header = f"[{doc_title}] [Điều {art_num}: {art_title}] [{cl_intro}] [Điểm {pt_label}]"
                    else:
                        pt_header = f"[{doc_title}] [Điều {art_num}: {art_title}] [Điểm {pt_label}]"

                    combined_text = f"{pt_header}\n{pt_content}"
                    pt_topics, pt_violations = extract_legal_topics_and_violations(combined_text)

                    pt_chunk = {
                        "id": pt_chunk_id,
                        "document_id": doc_id,
                        "version_id": ver_id,
                        "chapter_id": chap_id,
                        "article_id": art_id,
                        "clause_id": cl_id,
                        "point_id": pt_id,
                        "parent_chunk_id": cl_chunk_id,  # Direct parent = ClauseChunk!
                        "root_article_chunk_id": art_chunk_id,
                        "source_chunk_id": None,
                        "subchunk_index": None,
                        "is_embeddable": True,  # PointChunk luôn embeddable
                        "chunk_type": "point",
                        "chunk_index": global_chunk_idx,
                        "content": combined_text,
                        "normalized_content": remove_vietnamese_accents(combined_text),
                        "token_count": len(combined_text.split()),
                        "metadata": {
                            "document_number": doc_number,
                            "article_number": art_num,
                            "clause_number": cl_num,
                            "point_label": pt_label,
                            "document_type": doc.get("document_type", "luat"),
                            "domain": "giao_thong_duong_bo",
                            "topics": pt_topics or cl_topics or art_topics,
                            "violation_types": pt_violations or cl_violations or art_violations,
                        },
                        "validity": art_chunk["validity"],
                        "source": {
                            "file_name": file_name,
                            "page_start": pt.get("page_start", clause.get("page_start", art.get("page_start", 1))),
                            "page_end": pt.get("page_end", clause.get("page_end", art.get("page_end", 1))),
                            "source_type": doc_source.get("source_type", "pdf"),
                            "ocr_engine": doc_source.get("ocr_engine", "tesseract"),
                            "ocr_backend": doc_source.get("ocr_backend", "xberg"),
                        },
                        "search": {
                            "qdrant_collection": "legal_chunks_v2",
                            "qdrant_point_id": pt_chunk_id,
                        },
                    }
                    all_chunks.append(pt_chunk)
                    global_chunk_idx += 1

        is_valid, validation_errors = validate_chunk_tree(all_chunks)
        if not is_valid:
            print(f"[Chunker Cảnh báo Validation]: Phát hiện {len(validation_errors)} lỗi: {validation_errors}")

        return all_chunks
