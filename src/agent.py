import re
from builtins import input, print
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from src.indexer import LlamaIndexEmbedder, FAISSVectorStore, INDEX_PATH, METADATA_PATH
from src.fts_store import SQLiteFTSStore, FTS_DB_PATH

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

SYSTEM_PROMPT = """Bạn là Trợ lý AI chuyên nghiệp về Luật Trật tự, an toàn giao thông đường bộ Việt Nam.
Nhiệm vụ của bạn là giải đáp câu hỏi của người dùng một cách chính xác, minh bạch dựa TRỰC TIẾP trên các điều luật được cung cấp bên dưới.

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin trong phần "CĂN CỨ PHÁP LÝ DƯỚI ĐÂY" để trả lời.
2. Trích dẫn rõ ràng tên Điều, Khoản (ví dụ: "Theo Điều 25, Khoản 1..."). Không được tự tạo hoặc trích dẫn sai số Điều luật.
3. Nếu căn cứ được cung cấp không chứa thông tin để trả lời câu hỏi, hãy lịch sự thông báo rằng chưa tìm thấy quy định cụ thể trong văn bản luật hiện tại.
4. Trả lời rõ ràng, dễ hiểu, trình bày bằng Markdown có cấu trúc (gạch đầu dòng, in đậm).
"""

FALLBACK_SYSTEM_PROMPT = """Bạn là Trợ lý AI kiểm tra tính chính xác pháp lý.
LẦN TRẢ LỜI TRƯỚC BỊ PHÁT HIỆN TRÍCH DẪN SAI SỐ ĐIỀU LUẬT HOẶC KHÔNG CÓ TRONG CĂN CỨ.

YÊU CẦU NGHIÊM NGẶT:
- CHỈ ĐƯỢC trích dẫn chính xác các số Điều luật CÓ MẶT TRONG DƯỚI ĐÂY.
- Tuyệt đối không suy đoán hoặc tạo thêm số Điều luật mới.
- Nếu không có câu trả lời trong văn bản bên dưới, hãy trả lời: "Chưa tìm thấy quy định cụ thể trong văn bản luật được cung cấp."
"""


class LegalRAGAgent:
    """Hybrid RAG Agent combining LlamaIndex Embeddings + FAISS Dense Search + SQLite FTS5 BM25."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
        print("Initializing Legal RAG Agent with LlamaIndex...")
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment variables (.env)")

        print(f"Connecting to Hugging Face LLM API ({model_name})...")
        self.client = InferenceClient(api_key=self.hf_token)
        self.model_name = model_name

        print("Loading LlamaIndex HuggingFaceEmbedding Model...")
        self.embedder = LlamaIndexEmbedder(model_name="intfloat/multilingual-e5-base")

        print("Loading FAISS Vector Database...")
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index files not found in {INDEX_PATH.parent}. Please run 'python src/indexer.py' first."
            )
        self.vector_store = FAISSVectorStore.load(INDEX_PATH, METADATA_PATH)

        print("Loading SQLite FTS5 BM25 Database...")
        if not FTS_DB_PATH.exists():
            raise FileNotFoundError(
                f"SQLite FTS5 database not found in {FTS_DB_PATH.parent}. Please run 'python src/indexer.py' first."
            )
        self.fts_store = SQLiteFTSStore(FTS_DB_PATH)
        print("Hybrid Indexing Engine with LlamaIndex initialized successfully.")

    def hybrid_retrieve(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Retrieve documents using Reciprocal Rank Fusion (RRF) combining FAISS Dense & SQLite FTS5 BM25."""
        # 1. FAISS Dense Vector Search
        query_emb = self.embedder.get_query_embedding(query)
        dense_results = self.vector_store.search(query_emb, top_k=top_k * 2)

        # 2. SQLite FTS5 BM25 Keyword Search
        bm25_results = self.fts_store.search_bm25(query, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_map = {}

        # Process Dense Ranks
        for rank, item in enumerate(dense_results, 1):
            doc_id = item["metadata"].get("location") or item["metadata"].get("header")
            doc_map[doc_id] = item["metadata"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # Process BM25 Ranks
        for rank, item in enumerate(bm25_results, 1):
            doc_id = item["metadata"].get("location") or item["metadata"].get("header")
            doc_map[doc_id] = item["metadata"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # Sort combined documents by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        hybrid_results = []
        for doc_id, score in sorted_docs:
            hybrid_results.append({
                "score": score,
                "metadata": doc_map[doc_id],
            })

        return hybrid_results

    def verify_answer(self, answer_text: str, retrieved_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify LLM generated answer against SQLite FTS5 database and retrieved legal context."""
        cited_articles = set(re.findall(r"Điều\s+(\d+)", answer_text, flags=re.IGNORECASE))
        context_articles = set()

        for src in retrieved_sources:
            meta = src.get("metadata", {})
            header = meta.get("header", "")
            art_num = meta.get("article_num", "")
            if art_num:
                context_articles.add(art_num)
            found_in_header = re.findall(r"Điều\s+(\d+)", header, flags=re.IGNORECASE)
            context_articles.update(found_in_header)

        mismatched_articles = []
        verified_articles = []

        for art in cited_articles:
            if art in context_articles:
                verified_articles.append(art)
            else:
                if self.fts_store.verify_citation(art):
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

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def answer(self, query: str, top_k: int = 3, max_retries: int = 1) -> dict:
        """Retrieve context using Hybrid RRF Search, generate response, and execute verification fallback."""
        sources = self.hybrid_retrieve(query, top_k=top_k)

        # Construct context string
        context_blocks = []
        for i, item in enumerate(sources, 1):
            meta = item["metadata"]
            header = meta.get("header", "Điều luật")
            text = meta.get("text", "")
            score = item.get("score", 0.0)
            context_blocks.append(f"--- CĂN CỨ PHÁP LÝ {i} (RRF Score: {score:.4f}) ---\n{header}\n{text}\n")

        context_str = "\n".join(context_blocks)
        user_prompt = f"CĂN CỨ PHÁP LÝ DƯỚI ĐÂY:\n{context_str}\n\nCÂU HỎI CỦA NGƯỜI DÙNG:\n{query}\n\nHÃY TRẢ LỜI CÂU HỎI TRÊN:"

        try:
            answer_text = self._call_llm(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            return {
                "query": query,
                "answer": f"Đã xảy ra lỗi khi gọi LLM API: {exc}",
                "sources": sources,
                "verification": {"is_valid": False, "score": 0.0, "mismatched_articles": [], "retried": False},
            }

        # Run verification check
        audit = self.verify_answer(answer_text, sources)
        retried = False

        # Fallback Retry Loop with strict stop condition (max_retries = 1)
        if not audit["is_valid"] and max_retries > 0:
            retried = True
            print(f"[Verification Alert] Detected citation mismatch: {audit['mismatched_articles']}. Executing fallback (retry 1/1)...")
            fallback_prompt = (
                f"{user_prompt}\n\n"
                f"[CẢNH BÁO KIỂM TRA]: Lần trả lời trước bị phát hiện số Điều ({', '.join(audit['mismatched_articles'])}) "
                f"không có trong căn cứ. Hãy chỉnh sửa lại chỉ trích dẫn đúng các Điều luật có trong căn cứ trên."
            )
            try:
                answer_text = self._call_llm(FALLBACK_SYSTEM_PROMPT, fallback_prompt)
                audit = self.verify_answer(answer_text, sources)
            except Exception as exc:
                print(f"[Fallback Warning] Retry call failed: {exc}")

        audit["retried"] = retried

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "verification": audit,
        }


def main():
    agent = LegalRAGAgent()
    user_query = input("Nhập câu hỏi của bạn về Luật Giao thông đường bộ Việt Nam: ")
    res = agent.answer(user_query, top_k=3)
    print("\n--- TRẢ LỜI CỦA TRỢ LÝ LUẬT ---")
    print(res["answer"])
    print("\n--- KẾT QUẢ KIỂM TRA FTS5 BM25 ---")
    print(res["verification"])


if __name__ == "__main__":
    main()