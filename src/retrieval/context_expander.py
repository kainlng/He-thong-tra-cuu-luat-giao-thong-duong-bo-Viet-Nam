from typing import List, Dict, Any
from src.storage.postgres_manager import PostgresManager


class HierarchicalContextExpander:
    """Mở rộng và hợp nhất ngữ cảnh phân cấp từ PostgreSQL:
    Từ các chunk con (Điểm / Khoản) được tìm thấy bởi Qdrant -> Gom nhóm theo Điều luật cha -> Tái tạo Context hoàn chỉnh.
    """

    def __init__(self, postgres_manager: PostgresManager):
        self.pg = postgres_manager

    def expand_candidates(self, qdrant_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mở rộng và nhóm các ứng viên theo Điều luật gốc để tránh phân mảnh thông tin."""
        if not qdrant_candidates:
            return []

        chunk_ids = [c["chunk_id"] for c in qdrant_candidates if c.get("chunk_id")]
        rich_contexts = self.pg.get_hierarchical_context_by_chunk_ids(chunk_ids)

        # Map candidate score to rich context
        score_map = {c["chunk_id"]: c.get("score", 0.0) for c in qdrant_candidates}

        expanded_passages = []

        for ctx in rich_contexts:
            chunk_id = ctx["chunk_id"]
            art_id = ctx["article"].get("id", "")
            art_num = ctx["article"].get("article_number", "")
            art_title = ctx["article"].get("title", "")
            doc_info = ctx["document"]
            chap_info = ctx["chapter"]
            validity = ctx.get("validity", {})

            # Build full comprehensive text representation for Reranker and LLM
            doc_header = f"[{doc_info.get('title', 'Văn bản Luật')}] ({doc_info.get('document_number', '')})"
            chap_header = f"[Chương {chap_info.get('chapter_number', '')}: {chap_info.get('title', '')}]" if chap_info.get("chapter_number") else ""
            art_header = f"[Điều {art_num}: {art_title}]"

            # Render structured text of the article
            clauses = ctx["article"].get("clauses", [])
            points = ctx["article"].get("points", [])

            body_lines = []
            if clauses:
                for cl in clauses:
                    cl_num = cl.get("number")
                    cl_id = cl.get("id")
                    cl_content = cl.get("content", "")
                    
                    if cl_num:
                        body_lines.append(f"{cl_num}. {cl_content}")
                    else:
                        body_lines.append(cl_content)

                    # Attach points under this clause matching directly by clause_id
                    matching_points = [p for p in points if p.get("clause_id") == cl_id]
                    for pt in matching_points:
                        body_lines.append(f"   {pt['label']}) {pt['content']}")
            else:
                body_lines.append(ctx.get("content", ""))

            full_article_text = "\n".join(body_lines)

            passage_text = f"{doc_header}\n{chap_header}\n{art_header}\n{full_article_text}".strip()

            expanded_passages.append({
                "chunk_id": chunk_id,
                "article_id": art_id,
                "article_number": art_num,
                "article_title": art_title,
                "document_number": doc_info.get("document_number", ""),
                "document_title": doc_info.get("title", ""),
                "document_type": doc_info.get("document_type", "luat"),
                "chapter_number": chap_info.get("chapter_number", ""),
                "chapter_title": chap_info.get("title", ""),
                "passage_text": passage_text,
                "raw_chunk_content": ctx.get("content", ""),
                "initial_score": score_map.get(chunk_id, 0.0),
                "validity": validity,
                "source": ctx.get("source", {}),
                "metadata": ctx.get("metadata", {}),
            })

        return expanded_passages
