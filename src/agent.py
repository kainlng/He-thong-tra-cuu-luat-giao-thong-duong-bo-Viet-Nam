import os
import sys
from pathlib import Path

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from src.indexer import MultilingualE5Embedder, FAISSVectorStore, INDEX_PATH, METADATA_PATH

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = """Bạn là Trợ lý AI chuyên nghiệp về Luật Trật tự, an toàn giao thông đường bộ Việt Nam.
Nhiệm vụ của bạn là giải đáp câu hỏi của người dùng một cách chính xác, minh bạch dựa TRỰC TIẾP trên các điều luật được cung cấp bên dưới.

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin trong phần "CĂN CỨ PHÁP LÝ DƯỚI ĐÂY" để trả lời.
2. Trích dẫn rõ ràng tên Điều, Khoản (ví dụ: "Theo Điều 25, Khoản 1 Luật Trật tự, an toàn giao thông đường bộ 2024...").
3. Nếu căn cứ được cung cấp không chứa thông tin để trả lời câu hỏi, hãy lịch sự thông báo rằng chưa tìm thấy quy định cụ thể trong văn bản luật hiện tại.
4. Trả lời rõ ràng, dễ hiểu, trình bày bằng Markdown có cấu trúc (gạch đầu dòng, in đậm).
"""


class LegalRAGAgent:
    """RAG Agent integrating FAISS Vector Search with LLM Reasoning (Qwen 72B)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
        print("Initializing Legal RAG Agent...")
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment variables (.env)")

        print(f"Connecting to Hugging Face LLM API ({model_name})...")
        self.client = InferenceClient(api_key=self.hf_token)
        self.model_name = model_name

        print("Loading Multilingual E5 Embedding Model...")
        self.embedder = MultilingualE5Embedder(model_name="intfloat/multilingual-e5-base")

        print("Loading FAISS Vector Database...")
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index files not found in {INDEX_PATH.parent}. Please run 'python src/indexer.py' first."
            )
        self.vector_store = FAISSVectorStore.load(INDEX_PATH, METADATA_PATH)
        print(f"Loaded FAISS Vector Store with {self.vector_store.index.ntotal} legal article chunks.")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Embed search query and retrieve top-K legal chunks from FAISS."""
        query_emb = self.embedder.get_query_embedding(query)
        results = self.vector_store.search(query_emb, top_k=top_k)
        return results

    def answer(self, query: str, top_k: int = 3) -> dict:
        """Process user query, retrieve relevant law articles, and generate LLM response with citations."""
        sources = self.retrieve(query, top_k=top_k)

        # Construct context block from retrieved legal chunks
        context_blocks = []
        for i, item in enumerate(sources, 1):
            meta = item["metadata"]
            header = meta.get("header", "Điều luật")
            text = meta.get("text", "")
            score = item.get("score", 0.0)
            context_blocks.append(f"--- CĂN CỨ {i} (Độ khớp: {score:.4f}) ---\n{header}\n{text}\n")

        context_str = "\n".join(context_blocks)

        user_prompt = f"""CĂN CỨ PHÁP LÝ DƯỚI ĐÂY:
{context_str}

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

HÃY TRẢ LỜI CÂU HỎI TRÊN DỰA TRÊN CÁC CĂN CỨ PHÁP LÝ TRÊN:"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )
            answer_text = response.choices[0].message.content
        except Exception as exc:
            answer_text = f"Đã xảy ra lỗi khi gọi LLM API: {exc}"

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
        }


def main():
    agent = LegalRAGAgent()
    sample_query = "Đi xe máy trên đường cao tốc bị cấm hay không? Quy định tại điều nào?"
    print(f"\nQuery: {sample_query}\n")
    res = agent.answer(sample_query, top_k=3)
    print("--- TRẢ LỜI CỦA TRỢ LÝ LUẬT ---")
    print(res["answer"])


if __name__ == "__main__":
    main()