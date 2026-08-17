import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.core.config import HF_TOKEN, LLM_MODEL_NAME
from src.storage.indexer import LlamaIndexEmbedder
from src.storage.postgres_manager import PostgresManager
from src.storage.qdrant_manager import QdrantManager
from src.retrieval.reranker import LegalReranker
from src.retrieval.query_engine import LegalQueryEngine
from src.generation.citation_builder import CitationBuilder

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv(BASE_DIR / ".env")

SYSTEM_PROMPT = """Bạn là Trợ lý AI chuyên nghiệp về Pháp luật Giao thông đường bộ Việt Nam (Luật Trật tự, an toàn giao thông đường bộ 2024 và các Nghị định liên quan).
Nhiệm vụ của bạn là giải đáp câu hỏi của người dùng một cách chính xác, rõ ràng và minh bạch dựa TRỰC TIẾP trên các CĂN CỨ PHÁP LÝ được cung cấp.

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin trong các CĂN CỨ PHÁP LÝ bên dưới để trả lời.
2. Trích dẫn chuẩn xác tên Văn bản, số Điều, Khoản, Điểm (ví dụ: "Theo Điều 9, Khoản 2 Luật Trật tự, an toàn giao thông đường bộ 36/2024/QH15...").
3. Nêu rõ hiệu lực của văn bản (đang áp dụng hay quy định mới).
4. Nếu căn cứ không chứa thông tin để trả lời câu hỏi, hãy nêu rõ chưa tìm thấy quy định trong tài liệu hiện tại, không tự bịa đặt số Điều luật.
5. Trình bày khoa học bằng Markdown (tiêu đề, gạch đầu dòng, chữ in đậm).
"""

FALLBACK_SYSTEM_PROMPT = """Bạn là Trợ lý AI kiểm tra tính chính xác pháp lý.
LẦN TRẢ LỜI TRƯỚC BỊ PHÁT HIỆN TRÍCH DẪN SAI SỐ ĐIỀU HOẶC KHÔNG CÓ TRONG CĂN CỨ PHÁP LÝ.

YÊU CẦU NGHIÊM NGẶT:
- CHỈ ĐƯỢC trích dẫn chính xác các số Điều luật CÓ MẶT trong căn cứ bên dưới.
- Tuyệt đối không suy đoán hoặc tạo thêm số Điều luật mới.
- Nếu không có câu trả lời, hãy thông báo rõ ràng.
"""


class LegalRAGAgent:
    """Legal RAG Agent tích hợp PostgreSQL (Source of Truth) + Qdrant (Vector Index) + Hierarchical Retrieval + Reranker + Deterministic Citation."""

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        print(f"[Legal Agent] Khởi tạo Trợ lý AI Pháp luật ({model_name})...")
        self.hf_token = HF_TOKEN or os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("[Cảnh báo Agent] Chưa tìm thấy HF_TOKEN trong .env, các lệnh gọi LLM có thể bị hạn chế.")

        self.model_name = model_name
        self.client = InferenceClient(api_key=self.hf_token) if self.hf_token else None

        # Khởi tạo các components
        print("[Legal Agent] Đang nạp Embedder, PostgreSQL, Qdrant và Reranker...")
        self.embedder = LlamaIndexEmbedder()
        self.postgres = PostgresManager()
        self.qdrant = QdrantManager()
        self.reranker = LegalReranker()

        self.query_engine = LegalQueryEngine(
            embedder=self.embedder,
            qdrant_mgr=self.qdrant,
            postgres_mgr=self.postgres,
            reranker=self.reranker,
        )
        print("[Legal Agent] Hệ thống AI Agent đã sẵn sàng hoạt động.")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            return "Không thể gọi mô hình LLM do chưa cấu hình HF_TOKEN hợp lệ trong file .env."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1200,
            temperature=0.15,
        )
        return response.choices[0].message.content

    def verify_answer(self, answer_text: str, retrieved_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Kiểm chứng trích dẫn số Điều luật trong câu trả lời với Căn cứ thực tế từ PostgreSQL."""
        cited_articles = set(re.findall(r"Điều\s+(\d+)", answer_text, flags=re.IGNORECASE))
        context_articles = set()

        for ctx in retrieved_contexts:
            art_num = str(ctx.get("article_number", "")).strip()
            if art_num:
                context_articles.add(art_num)

        mismatched_articles = []
        verified_articles = []

        for art in cited_articles:
            if art in context_articles:
                verified_articles.append(art)
            else:
                mismatched_articles.append(art)

        is_valid = len(mismatched_articles) == 0
        total_citations = len(cited_articles)
        score = 1.0 if total_citations == 0 else len(verified_articles) / total_citations

        return {
            "is_valid": is_valid,
            "score": score,
            "cited_articles": list(cited_articles),
            "verified_articles": verified_articles,
            "mismatched_articles": mismatched_articles,
        }

    def answer(
        self,
        query: str,
        top_k: int = 20,
        top_n: int = 4,
        force_current: Optional[bool] = None,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """Thực hiện chu trình RAG hoàn chỉnh:
        Query -> Time Intent -> Qdrant Filter + Dense Search -> PostgreSQL Context Expansion -> Reranking -> LLM Generation -> Verification.
        """
        time_intent = self.query_engine.parse_time_intent(query)
        
        # 1. Retrieval
        retrieved_contexts = self.query_engine.retrieve(
            query=query,
            top_k=top_k,
            top_n=top_n,
            force_current=force_current,
        )

        # 2. Xây dựng Context và Deterministic Citations
        context_str = CitationBuilder.build_llm_context(retrieved_contexts)
        citations = CitationBuilder.generate_citations(retrieved_contexts)

        user_prompt = (
            f"CĂN CỨ PHÁP LÝ DƯỚI ĐÂY:\n{context_str}\n\n"
            f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{query}\n\n"
            f"HÃY GIẢI ĐÁP CÂU HỎI TRÊN VÀ TRÍCH DẪN ĐẦY ĐỦ CĂN CỨ:"
        )

        # 3. LLM Call
        try:
            answer_text = self._call_llm(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            return {
                "query": query,
                "answer": f"Đã xảy ra lỗi khi gọi mô hình ngôn ngữ: {exc}",
                "citations": citations,
                "retrieved_contexts": retrieved_contexts,
                "verification": {"is_valid": False, "score": 0.0, "mismatched_articles": []},
                "time_intent": time_intent,
            }

        # 4. Kiểm chứng trích dẫn
        audit = self.verify_answer(answer_text, retrieved_contexts)
        retried = False

        if not audit["is_valid"] and max_retries > 0:
            retried = True
            print(f"[Verification Alert] Phát hiện trích dẫn Điều chưa khớp: {audit['mismatched_articles']}. Đang thực hiện fallback retry...")
            fallback_prompt = (
                f"{user_prompt}\n\n"
                f"[CẢNH BÁO KIỂM CHỨNG]: Lần trả lời trước bị phát hiện số Điều ({', '.join(audit['mismatched_articles'])}) "
                f"không có trong căn cứ. Hãy chỉ trích dẫn đúng các Điều luật có trong căn cứ trên."
            )
            try:
                answer_text = self._call_llm(FALLBACK_SYSTEM_PROMPT, fallback_prompt)
                audit = self.verify_answer(answer_text, retrieved_contexts)
            except Exception as exc:
                print(f"[Fallback Warning] Lỗi khi retry: {exc}")

        audit["retried"] = retried

        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "retrieved_contexts": retrieved_contexts,
            "verification": audit,
            "time_intent": time_intent,
        }


def main():
    agent = LegalRAGAgent()
    user_query = input("Nhập câu hỏi của bạn về Luật Giao thông đường bộ: ")
    res = agent.answer(user_query)
    print("\n--- TRẢ LỜI CỦA TRỢ LÝ LUẬT ---")
    print(res["answer"])
    print("\n--- TRÍCH DẪN NGUỒN XÁC THỰC (DETERMINISTIC CITATIONS) ---")
    for c in res["citations"]:
        print(f"{c['citation_text']} (Hiệu lực: {c['status']}) - Score: {c['score']:.4f}")
    print("\n--- KẾT QUẢ KIỂM CHỨNG ---")
    print(res["verification"])


if __name__ == "__main__":
    main()
