import os
import sys
from pathlib import Path

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from src.agent import LegalRAGAgent

# Set Streamlit page configuration
st.set_page_config(
    page_title="Hệ thống Tra cứu Luật Giao thông Đường bộ",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-title {
        color: #1E3A8A;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #2563EB;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">🚦 Tra cứu Luật Trật tự, An toàn Giao thông Đường bộ</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Cấu hình RAG System")
    top_k = st.slider("Số lượng Điều luật trích dẫn (Top-K):", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.markdown("### ℹ️ Thông tin Mô hình & Search")
    st.markdown("- **Embedding**: `intfloat/multilingual-e5-base`")
    st.markdown("- **Hybrid Retrieval**: `FAISS (Dense)` + `SQLite FTS5 (BM25)`")
    st.markdown("- **Verification**: `FTS5 Double-Check & Fallback (Max 1 Retry)`")
    st.markdown("- **LLM**: `Qwen/Qwen2.5-72B-Instruct`")

    if st.button("🧹 Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()

# Initialize Legal RAG Agent once in session_state
@st.cache_resource(show_spinner="Đang khởi tạo Trợ lý AI, FAISS và SQLite FTS5...")
def get_agent():
    return LegalRAGAgent()

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Không thể khởi động Trợ lý AI: {e}")
    st.stop()

# Helper renderer for verification audit block
def render_verification_audit(verification: dict):
    if not verification:
        return
    is_valid = verification.get("is_valid", True)
    score = verification.get("score", 1.0)
    cited = verification.get("cited_articles", [])
    mismatched = verification.get("mismatched_articles", [])
    retried = verification.get("retried", False)

    if is_valid:
        st.success(f"✅ **Đã kiểm chứng trích dẫn qua FTS5 BM25** (Độ khớp: {score:.0%})")
    else:
        st.warning(f"⚠️ **Cảnh báo kiểm chứng FTS5**: Phát hiện số Điều chưa khớp ({', '.join(mismatched)})")

    with st.expander("🔍 Chi tiết Kiểm chứng Pháp lý (FTS5 BM25 Audit)"):
        st.markdown(f"- **Trạng thái trích dẫn**: {'Đạt 100%' if is_valid else 'Cần xem xét'}")
        st.markdown(f"- **Các Điều luật được trích dẫn**: `{', '.join(cited) if cited else 'Không trích dẫn cụ thể'}`")
        if mismatched:
            st.markdown(f"- ❌ **Điều luật chưa tìm thấy trong căn cứ**: `{', '.join(mismatched)}`")
        if retried:
            st.markdown("- 🔄 **Hệ thống đã tự động kích hoạt Fallback Retry (Tối đa 1 lần)**")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi là Trợ lý AI Tra cứu Luật Trật tự, an toàn giao thông đường bộ. Bạn có câu hỏi hoặc thắc mắc gì về quy định giao thông không?",
            "sources": [],
            "verification": None,
        }
    ]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("verification"):
            render_verification_audit(message["verification"])
        if message.get("sources"):
            with st.expander("📜 Căn cứ pháp lý & Vị trí trích dẫn"):
                for i, src in enumerate(message["sources"], 1):
                    meta = src["metadata"]
                    score = src.get("score", 0.0)
                    source_file = meta.get("source", "Tài liệu Luật")
                    header = meta.get("header", "Điều luật")
                    location = meta.get("location", f"{source_file} > {header}")

                    st.markdown(
                        f"**{i}. {header}** *(RRF Score: {score:.4f})*\n"
                        f"- 📌 **Vị trí trích dẫn**: `{location}`\n"
                        f"- 📄 **Tên văn bản gốc**: `{source_file}`\n\n"
                        f"```text\n{meta.get('text', '')}\n```"
                    )

# Chat Input & Handling
if prompt := st.chat_input("Nhập câu hỏi của bạn (vd: Đi xe máy trên đường cao tốc bị cấm không?)..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu Hybrid (FAISS + FTS5 BM25) và kiểm chứng câu trả lời..."):
            res = agent.answer(prompt, top_k=top_k)
            answer_text = res["answer"]
            sources = res["sources"]
            verification = res.get("verification")

            st.markdown(answer_text)
            if verification:
                render_verification_audit(verification)

            if sources:
                with st.expander("📜 Căn cứ pháp lý & Vị trí trích dẫn"):
                    for i, src in enumerate(sources, 1):
                        meta = src["metadata"]
                        score = src.get("score", 0.0)
                        source_file = meta.get("source", "Tài liệu Luật")
                        header = meta.get("header", "Điều luật")
                        location = meta.get("location", f"{source_file} > {header}")

                        st.markdown(
                            f"**{i}. {header}** *(RRF Score: {score:.4f})*\n"
                            f"- 📌 **Vị trí trích dẫn**: `{location}`\n"
                            f"- 📄 **Tên văn bản gốc**: `{source_file}`\n\n"
                            f"```text\n{meta.get('text', '')}\n```"
                        )

    # Save to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer_text, "sources": sources, "verification": verification})
