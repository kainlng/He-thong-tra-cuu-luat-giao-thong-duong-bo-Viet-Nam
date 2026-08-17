import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from src.core.models import LegalChunkDTO, DocumentMetadata, ValidityMetadata, SourceMetadata, QdrantChunkPayload


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


class HierarchicalLegalChunker:
    """Tạo các Chunks phân cấp từ cây cấu trúc văn bản pháp luật:
    - Điểm (Point Chunk) -> Khoản (Clause Chunk) -> Điều (Article Chunk).
    - Mỗi Chunk đều gắn liền với Breadcrumb ngữ cảnh và parent_chunk_id.
    """

    def __init__(self):
        pass

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

        # Index clauses by article_id and points by clause_id
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
            token_count = len(art_chunk_content.split())

            art_chunk = {
                "id": art_chunk_id,
                "document_id": doc_id,
                "version_id": ver_id,
                "chapter_id": chap_id,
                "article_id": art_id,
                "clause_id": None,
                "point_id": None,
                "parent_chunk_id": None,  # Root of this article tree
                "chunk_type": "article",
                "chunk_index": global_chunk_idx,
                "content": art_chunk_content,
                "normalized_content": remove_vietnamese_accents(art_chunk_content),
                "token_count": token_count,
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
                    "effective_from": ver.get("effective_from", "2025-01-01"),
                    "effective_to": ver.get("effective_to"),
                    "status": ver.get("status", "active"),
                    "is_current": ver.get("is_current", True),
                },
                "source": {
                    "file_name": file_name,
                    "page_start": 1,
                    "page_end": 1,
                    "source_type": "markdown",
                },
                "search": {
                    "qdrant_collection": "legal_chunks",
                    "qdrant_point_id": art_chunk_id,
                },
            }
            all_chunks.append(art_chunk)
            global_chunk_idx += 1

            # 2. Xử lý từng Khoản trong Điều
            for clause in art_clauses:
                cl_id = clause["id"]
                cl_num = clause["clause_number"]
                cl_content = clause["content"]
                cl_points = points_by_clause.get(cl_id, [])

                raw_cl_chunk_id = f"chunk_{doc_id.replace('doc_', '')}_{art_num.zfill(2)}_{cl_num.zfill(2)}"
                cl_chunk_id = make_unique_id(raw_cl_chunk_id)
                cl_topics, cl_violations = extract_legal_topics_and_violations(cl_content)

                # Nếu Khoản không có Điểm con -> Đây là lá (Leaf Chunk)
                if not cl_points:
                    cl_text = f"[{doc_title}] [Điều {art_num}: {art_title}] [Khoản {cl_num}]\n{cl_content}"
                    cl_chunk = {
                        "id": cl_chunk_id,
                        "document_id": doc_id,
                        "version_id": ver_id,
                        "chapter_id": chap_id,
                        "article_id": art_id,
                        "clause_id": cl_id,
                        "point_id": None,
                        "parent_chunk_id": art_chunk_id,
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
                        "source": art_chunk["source"],
                        "search": {
                            "qdrant_collection": "legal_chunks",
                            "qdrant_point_id": cl_chunk_id,
                        },
                    }
                    all_chunks.append(cl_chunk)
                    global_chunk_idx += 1
                else:
                    # Nếu Khoản có các Điểm con -> Tạo Chunk cho từng Điểm
                    for pt in cl_points:
                        pt_id = pt["id"]
                        pt_label = pt["point_label"]
                        pt_content = pt["content"]

                        raw_pt_chunk_id = f"chunk_{doc_id.replace('doc_', '')}_{art_num.zfill(2)}_{cl_num.zfill(2)}_{pt_label}"
                        pt_chunk_id = make_unique_id(raw_pt_chunk_id)
                        combined_text = f"[{doc_title}] [Điều {art_num}: {art_title}] [Khoản {cl_num}] [Điểm {pt_label}]\n{pt_content}"
                        
                        pt_topics, pt_violations = extract_legal_topics_and_violations(combined_text)

                        pt_chunk = {
                            "id": pt_chunk_id,
                            "document_id": doc_id,
                            "version_id": ver_id,
                            "chapter_id": chap_id,
                            "article_id": art_id,
                            "clause_id": cl_id,
                            "point_id": pt_id,
                            "parent_chunk_id": art_chunk_id,
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
                            "source": art_chunk["source"],
                            "search": {
                                "qdrant_collection": "legal_chunks",
                                "qdrant_point_id": pt_chunk_id,
                            },
                        }
                        all_chunks.append(pt_chunk)
                        global_chunk_idx += 1

        return all_chunks
