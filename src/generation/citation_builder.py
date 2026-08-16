from typing import List, Dict, Any


class CitationBuilder:
    """Tự động tạo trích dẫn pháp lý chính xác (Deterministic Legal Citation) và cấu trúc Context cho LLM."""

    @staticmethod
    def build_llm_context(retrieved_contexts: List[Dict[str, Any]]) -> str:
        """Định dạng context rõ ràng gửi cho LLM để tránh ảo giác trích dẫn."""
        context_blocks = []
        for i, ctx in enumerate(retrieved_contexts, 1):
            doc_title = ctx.get("document_title", "Văn bản Luật")
            doc_num = ctx.get("document_number", "")
            art_num = ctx.get("article_number", "")
            art_title = ctx.get("article_title", "")
            chap_num = ctx.get("chapter_number", "")
            chap_title = ctx.get("chapter_title", "")
            validity = ctx.get("validity", {})
            eff_from = validity.get("effective_from", "01/01/2025")
            status = validity.get("status", "active")
            passage = ctx.get("passage_text", "")

            block = (
                f"--- [CĂN CỨ PHÁP LÝ #{i}] ---\n"
                f"MÃ NGUỒN: {ctx.get('chunk_id', f'SRC_{i}')}\n"
                f"VĂN BẢN: {doc_title} (Số hiệu: {doc_num})\n"
                f"CHƯƠNG: {f'Chương {chap_num} - {chap_title}' if chap_num else 'N/A'}\n"
                f"ĐIỀU KHOẢN: Điều {art_num}. {art_title}\n"
                f"HIỆU LỰC: {status.upper()} (Áp dụng từ: {eff_from})\n\n"
                f"TOÀN VĂN QUY ĐỊNH:\n{passage}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    @staticmethod
    def generate_citations(retrieved_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tạo danh sách trích dẫn chuẩn hóa hiển thị trên giao diện người dùng."""
        citations = []
        for i, ctx in enumerate(retrieved_contexts, 1):
            doc_title = ctx.get("document_title", "Văn bản Luật")
            doc_num = ctx.get("document_number", "")
            art_num = ctx.get("article_number", "")
            art_title = ctx.get("article_title", "")
            chap_num = ctx.get("chapter_number", "")
            validity = ctx.get("validity", {})
            score = ctx.get("rerank_score", ctx.get("initial_score", 0.0))

            citation_text = f"[{i}] {doc_title} ({doc_num}), Điều {art_num} ({art_title})"
            if chap_num:
                citation_text += f", Chương {chap_num}"

            citations.append({
                "index": i,
                "citation_text": citation_text,
                "document_number": doc_num,
                "document_title": doc_title,
                "article_number": art_num,
                "article_title": art_title,
                "status": validity.get("status", "active"),
                "effective_from": validity.get("effective_from", "2025-01-01"),
                "score": float(score),
                "source_file": ctx.get("source", {}).get("file_name", ""),
                "full_text": ctx.get("passage_text", ""),
            })

        return citations
