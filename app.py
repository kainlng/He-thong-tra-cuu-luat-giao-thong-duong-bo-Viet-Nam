import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import streamlit as st

from src.config import EMBEDDING_DIMENSION
from src.agent import LegalRAGAgent
from src.postgres_manager import PostgresManager
from src.qdrant_manager import QdrantManager
from src.vector_visualizer import VectorVisualizer

# Set Streamlit page configuration
st.set_page_config(
    page_title="Hệ thống Tra cứu Pháp luật Phân cấp - PostgreSQL & Qdrant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .main-title {
        color: #1E3A8A;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .badge-active {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-expired {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-breadcrumb {
        background-color: #F1F5F9;
        color: #334155;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .citation-card {
        border-left: 4px solid #2563EB;
        background-color: #F8FAFC;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        border-radius: 0 6px 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">⚖️ Hệ thống Tra cứu Pháp luật Phân cấp</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Kiến trúc: <b>PostgreSQL (Source of Truth)</b> • <b>Qdrant (Vector Search)</b> • '
    '<b>Hierarchical Chunking (Chương → Điều → Khoản → Điểm)</b> • <b>FlashRank Reranker</b> • <b>Qwen 2.5 72B</b></div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang khởi tạo Trợ lý AI, PostgreSQL, Qdrant và Embeddings...")
def get_system_resources():
    agent = LegalRAGAgent()
    pg_mgr = agent.postgres
    qdrant_mgr = agent.qdrant
    return agent, pg_mgr, qdrant_mgr


try:
    agent, pg_mgr, qdrant_mgr = get_system_resources()
except Exception as e:
    st.error(f"Không thể khởi động hệ thống: {e}")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")

    # Time / Validity Filter Mode
    search_mode = st.radio(
        "Chế độ hiệu lực văn bản:",
        ["🟢 Quy định hiện hành (Active)", "🕒 Tra cứu theo mốc thời gian tùy chọn", "🌐 Tất cả phiên bản"],
    )

    force_current = True
    if "tùy chọn" in search_mode:
        custom_date = st.date_input("Chọn ngày áp dụng hiệu lực:", value=datetime.today())
        force_current = None
    elif "Tất cả" in search_mode:
        force_current = False

    top_n_rerank = st.slider("Số lượng Điều luật trích dẫn (Top-N Reranked):", min_value=1, max_value=8, value=3)

    st.markdown("---")
    st.markdown("### 📊 Thống kê Cơ sở Dữ liệu")
    stats = pg_mgr.get_database_stats()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Văn bản", stats.get("documents", 0))
        st.metric("Chương", stats.get("chapters", 0))
        st.metric("Khoản", stats.get("clauses", 0))
    with col_s2:
        st.metric("Phiên bản", stats.get("versions", 0))
        st.metric("Điều luật", stats.get("articles", 0))
        st.metric("Chunks", stats.get("chunks", 0))

    st.markdown("---")
    st.markdown("### ℹ️ Kiến trúc Hệ thống")
    st.markdown("- **Source of Truth**: `PostgreSQL 16 (7 Bảng Quan hệ)`")
    st.markdown("- **Vector Database**: `Qdrant v1.9 (Payload Pre-filtering)`")
    st.markdown("- **Embedding**: `intfloat/multilingual-e5-base (768d)`")
    st.markdown("- **Reranker**: `FlashRank / Cross-Encoder`")
    st.markdown("- **LLM**: `Qwen/Qwen2.5-72B-Instruct`")

    if st.button("🧹 Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()


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
        st.success(f"✅ **Đã kiểm chứng trích dẫn qua PostgreSQL** (Độ khớp: {score:.0%})")
    else:
        st.warning(f"⚠️ **Cảnh báo kiểm chứng**: Phát hiện số Điều chưa khớp ({', '.join(mismatched)})")

    with st.expander("🔍 Chi tiết Kiểm chứng Căn cứ Pháp lý (PostgreSQL Source Audit)"):
        st.markdown(f"- **Trạng thái trích dẫn**: {'Đạt 100%' if is_valid else 'Cần xem xét'}")
        st.markdown(f"- **Các Điều luật trích dẫn**: `{', '.join(cited) if cited else 'Không trích dẫn cụ thể'}`")
        if mismatched:
            st.markdown(f"- ❌ **Điều luật chưa tìm thấy trong căn cứ**: `{', '.join(mismatched)}`")
        if retried:
            st.markdown("- 🔄 **Hệ thống đã tự động kích hoạt Fallback Retry**")


# Navigation Tabs
tab_chat, tab_vector_map, tab_database_explorer = st.tabs([
    "💬 Trò chuyện & Tra cứu RAG Phân cấp",
    "🌌 Không gian Vector & Qdrant Explorer",
    "🏛️ Khám phá Cơ sở Dữ liệu Pháp luật (PostgreSQL)",
])


# ==============================================================================
# TAB 1: RAG CHAT ASSISTANT (HIERARCHICAL & CITATIONS)
# ==============================================================================
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào! Tôi là Trợ lý AI Tra cứu Luật Giao thông đường bộ Việt Nam. "
                    "Tôi sử dụng kiến trúc **PostgreSQL + Qdrant phân cấp** để trích dẫn chuẩn xác theo từng Chương, Điều, Khoản, Điểm. "
                    "Bạn cần tra cứu quy định hoặc mức xử phạt nào?"
                ),
                "citations": [],
                "verification": None,
            }
        ]

    # Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("verification"):
                render_verification_audit(message["verification"])
            if message.get("citations"):
                with st.expander(f"📜 Căn cứ pháp lý & Trích dẫn chính xác ({len(message['citations'])} Điều luật)"):
                    for c in message["citations"]:
                        status_badge = (
                            '<span class="badge-active">🟢 ĐANG ÁP DỤNG</span>'
                            if c.get("status") == "active"
                            else '<span class="badge-expired">🔴 HẾT HIỆU LỰC</span>'
                        )
                        st.markdown(
                            f'<div class="citation-card">'
                            f'<b>{c["citation_text"]}</b> {status_badge}<br>'
                            f'<small>Score: <code>{c.get("score", 0.0):.4f}</code> | File: <code>{c.get("source_file")}</code> | Ngày áp dụng: <code>{c.get("effective_from")}</code></small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"📄 Xem toàn văn Điều {c.get('article_number')}: {c.get('article_title')}"):
                            st.markdown(c.get("full_text", ""))

    # User Input Handling
    if prompt := st.chat_input("Nhập câu hỏi pháp luật (vd: Vượt đèn đỏ xe máy bị xử phạt thế nào?)..."):
        st.session_state.last_query = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang truy vấn Qdrant Vector Filter, mở rộng cây phân cấp PostgreSQL và Rerank..."):
                res = agent.answer(
                    query=prompt,
                    top_n=top_n_rerank,
                    force_current=force_current,
                )
                answer_text = res["answer"]
                citations = res.get("citations", [])
                verification = res.get("verification")

                st.markdown(answer_text)

                if verification:
                    render_verification_audit(verification)

                if citations:
                    with st.expander(f"📜 Căn cứ pháp lý & Trích dẫn chính xác ({len(citations)} Điều luật)"):
                        for c in citations:
                            status_badge = (
                                '<span class="badge-active">🟢 ĐANG ÁP DỤNG</span>'
                                if c.get("status") == "active"
                                else '<span class="badge-expired">🔴 HẾT HIỆU LỰC</span>'
                            )
                            st.markdown(
                                f'<div class="citation-card">'
                                f'<b>{c["citation_text"]}</b> {status_badge}<br>'
                                f'<small>Score: <code>{c.get("score", 0.0):.4f}</code> | File: <code>{c.get("source_file")}</code> | Ngày áp dụng: <code>{c.get("effective_from")}</code></small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            with st.expander(f"📄 Xem toàn văn Điều {c.get('article_number')}: {c.get('article_title')}"):
                                st.markdown(c.get("full_text", ""))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "citations": citations,
            "verification": verification,
        })


# ==============================================================================
# TAB 2: VECTOR SPACE & QDRANT EXPLORER
# ==============================================================================
with tab_vector_map:
    st.markdown("### 🌌 Bản đồ Không gian Vector Qdrant & Phân cụm Ngữ nghĩa")
    st.markdown(
        "Mỗi điểm trên không gian 3D/2D đại diện cho một **Semantic Legal Chunk** được vector hóa từ văn bản luật "
        "bằng mô hình `intfloat/multilingual-e5-base`."
    )

    all_chunks = pg_mgr.get_all_chunks()
    if all_chunks:
        sample_size = min(len(all_chunks), 150)
        st.info(f"Đang hiển thị {sample_size} chunks từ cơ sở dữ liệu PostgreSQL + Qdrant.")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            dim_mode = st.radio("Chế độ hiển thị:", ["3D (Xoay 360°)", "2D (Phẳng)"], horizontal=True)
        with col2:
            proj_method = st.selectbox("Thuật toán giảm chiều:", ["PCA", "t-SNE"])
        with col3:
            project_query = st.checkbox("🎯 Chiếu câu hỏi lên không gian Vector", value=True)

        # Generate vectors on the fly for visualization
        sample_texts = [c["content"] for c in all_chunks[:sample_size]]
        sample_vectors = agent.embedder.get_document_embeddings(sample_texts, batch_size=32)

        vis_metadata = [
            {
                "header": f"Điều {c.get('metadata', {}).get('article_number', '')} ({c.get('chunk_type', '')})",
                "source": c.get("source", {}).get("file_name", ""),
                "location": f"ID: {c['id']}",
                "text": c["content"],
            }
            for c in all_chunks[:sample_size]
        ]

        visualizer = VectorVisualizer(sample_vectors, vis_metadata, n_clusters=min(5, len(all_chunks)))

        query_vec = None
        query_text = ""
        nearest_idx = []

        if project_query:
            default_q = st.session_state.get("last_query", "Quy định về nồng độ cồn và vượt đèn đỏ")
            query_input = st.text_input("Nhập câu hỏi để chiếu lên không gian vector:", value=default_q)
            if query_input.strip():
                query_text = query_input.strip()
                query_vec = agent.embedder.get_query_embedding(query_text)[0]
                # Calculate nearest by cosine similarity
                sims = np.dot(sample_vectors, query_vec)
                nearest_idx = np.argsort(sims)[::-1][:3].tolist()

        if "3D" in dim_mode:
            fig_space = visualizer.plot_3d_space(
                method=proj_method,
                query_vector=query_vec,
                query_text=query_text,
                nearest_indices=nearest_idx,
            )
            st.plotly_chart(fig_space, use_container_width=True)
        else:
            fig_space = visualizer.plot_2d_space(
                method=proj_method,
                query_vector=query_vec,
                query_text=query_text,
                nearest_indices=nearest_idx,
            )
            st.plotly_chart(fig_space, use_container_width=True)
    else:
        st.warning("Chưa có dữ liệu chunks trong Database. Vui lòng chạy Ingestion Pipeline trước: `python src/ingest_pipeline.py`.")


# ==============================================================================
# TAB 3: POSTGRESQL DATABASE EXPLORER
# ==============================================================================
with tab_database_explorer:
    st.markdown("### 🏛️ Trình Khám phá Cây Cấu trúc Dữ liệu Pháp luật (PostgreSQL)")
    st.markdown(
        "Dữ liệu pháp luật được chuẩn hóa trong **7 bảng quan hệ** PostgreSQL. "
        "Tại đây bạn có thể kiểm tra danh mục văn bản, các phiên bản hiệu lực, và duyệt chi tiết cây Chương → Điều → Khoản → Điểm."
    )

    search_kw = st.text_input("🔍 Tìm kiếm nhanh trong toàn bộ Chunks:", "")
    all_chunks = pg_mgr.get_all_chunks()

    if all_chunks:
        filtered_chunks = []
        for c in all_chunks:
            meta = c.get("metadata") or {}
            content = c.get("content", "")
            art_num = meta.get("article_number", "")
            if not search_kw or search_kw.lower() in content.lower() or search_kw in art_num:
                filtered_chunks.append({
                    "Chunk ID": c["id"],
                    "Loại Chunk": c["chunk_type"],
                    "Số Điều": art_num,
                    "Topics": ", ".join(meta.get("topics", [])),
                    "Hành vi vi phạm": ", ".join(meta.get("violation_types", [])),
                    "Hiệu lực": c.get("validity", {}).get("status", "active"),
                    "Trích đoạn nội dung": content[:120] + "...",
                })

        st.markdown(f"**Hiển thị {len(filtered_chunks)} / {len(all_chunks)} chunks:**")
        st.dataframe(filtered_chunks, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu trong PostgreSQL. Chạy lệnh: `python src/ingest_pipeline.py` để nạp dữ liệu.")
