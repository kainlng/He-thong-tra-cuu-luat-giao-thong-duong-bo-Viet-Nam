import os
import sys
import pickle
from pathlib import Path

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import faiss
import streamlit as st
from src.agent import LegalRAGAgent
from src.indexer import INDEX_PATH, METADATA_PATH
from src.vector_visualizer import VectorVisualizer

# Set Streamlit page configuration
st.set_page_config(
    page_title="Hệ thống Tra cứu & Khám phá Không gian Vector Luật Giao thông",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1350px;
        margin: 0 auto;
    }
    .main-title {
        color: #1E3A8A;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #4B5563;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    .vector-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">🚦 Hệ thống Tra cứu & Khám phá Vector Database Luật Giao thông</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Trực quan hóa Không gian Vector 768 Chiều • FAISS Dense Search • SQLite FTS5 BM25 • Qwen 2.5 72B</div>', unsafe_allow_html=True)

# Initialize Legal RAG Agent once in session_state
@st.cache_resource(show_spinner="Đang khởi tạo Trợ lý AI, FAISS và SQLite FTS5...")
def get_agent():
    return LegalRAGAgent()

@st.cache_resource(show_spinner="Đang nạp toàn bộ Vector và khởi tạo Visualizer 2D/3D...")
def get_visualizer():
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    
    # Reconstruct all vectors into numpy array
    vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for i in range(index.ntotal):
        vectors[i] = index.reconstruct(i)
    
    vis = VectorVisualizer(vectors, metadata, n_clusters=5)
    return vis, vectors, metadata

try:
    agent = get_agent()
    visualizer, all_vectors, all_metadata = get_visualizer()
except Exception as e:
    st.error(f"Không thể khởi động hệ thống: {e}")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    top_k = st.slider("Số lượng Điều luật trích dẫn (Top-K):", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.markdown("### 📊 Thống kê Vector Database")
    st.metric("Tổng số Vectors", len(all_vectors))
    st.metric("Số chiều Embeddings", all_vectors.shape[1])
    st.metric("Cấu trúc Index", "FAISS FlatIP + SQLite FTS5")

    st.markdown("---")
    st.markdown("### ℹ️ Thông tin Mô hình")
    st.markdown("- **Embedding**: `intfloat/multilingual-e5-base`")
    st.markdown("- **Hybrid Search**: `FAISS (Dense)` + `FTS5 (BM25)`")
    st.markdown("- **Verification**: `FTS5 Citation Audit`")
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

# Navigation Tabs
tab_chat, tab_vector_map, tab_matrix = st.tabs([
    "💬 Trò chuyện & Tra cứu RAG",
    "🌌 Bản đồ Không gian Vector 2D / 3D",
    "🔬 Phân tích Ma trận & Phổ nhiệt (Heatmap)",
])

# ==============================================================================
# TAB 1: RAG CHAT ASSISTANT
# ==============================================================================
with tab_chat:
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
        # Save prompt to session state for live vector projection in Tab 2
        st.session_state.last_query = prompt
        
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
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "verification": verification
        })

# ==============================================================================
# TAB 2: VECTOR SPACE EXPLORER (2D / 3D)
# ==============================================================================
with tab_vector_map:
    st.markdown("### 🌌 Bản đồ Không gian Ngữ nghĩa Vector 2D / 3D")
    st.markdown(
        "Mỗi điểm trên đồ thị biểu diễn một **Điều luật (đã được số hóa thành vector 768 chiều)**. "
        "Các Điều luật có ngữ nghĩa gần nhau sẽ tự động hội tụ thành các cụm màu tương ứng."
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        dim_mode = st.radio("Chế độ hiển thị không gian:", ["3D (Xoay 360°)", "2D (Phẳng)"], horizontal=True)
    with col2:
        proj_method = st.selectbox("Thuật toán giảm chiều:", ["PCA", "t-SNE"])
    with col3:
        project_query = st.checkbox("🎯 Chiếu vị trí Câu hỏi (Query Projection)", value=True)

    query_vec = None
    query_text = ""
    nearest_idx = []

    if project_query:
        default_q = st.session_state.get("last_query", "Quy định về nồng độ cồn và tốc độ xe máy")
        query_input = st.text_input("Nhập câu hỏi để chiếu lên không gian vector:", value=default_q)
        if query_input.strip():
            query_text = query_input.strip()
            # Embed query using agent embedder
            query_vec = agent.embedder.get_query_embedding(query_text)[0]
            # Search top nearest vectors in FAISS
            scores, indices = agent.vector_store.index.search(query_vec.reshape(1, -1), top_k)
            nearest_idx = [int(idx) for idx in indices[0] if idx != -1]

    # Render Plotly Chart
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

    if nearest_idx:
        st.markdown(f"#### 🎯 Top {len(nearest_idx)} Điều luật gần nhất với câu hỏi trên không gian vector:")
        cols = st.columns(len(nearest_idx))
        for i, idx in enumerate(nearest_idx):
            with cols[i]:
                meta = all_metadata[idx]
                st.info(f"**Top {i+1} Match (ID #{idx})**\n\n📌 **{meta.get('header', '')}**\n\n`{meta.get('location', '')}`")

# ==============================================================================
# TAB 3: HEATMAPS & RAW VECTOR INSPECTOR
# ==============================================================================
with tab_matrix:
    st.markdown("### 🔬 Phân tích Chi tiết Vector & Ma trận Tương quan")

    st.markdown("#### 🔥 1. Ma trận Tương đồng Cosine giữa các Điều luật (Cosine Similarity)")
    st.markdown("Màu càng sáng (vàng) thể hiện 2 Điều luật có nội dung càng tương đồng với nhau.")
    max_heatmap_items = st.slider("Số lượng Điều luật hiển thị trên ma trận:", min_value=10, max_value=50, value=25)
    fig_sim = visualizer.plot_similarity_heatmap(max_items=max_heatmap_items)
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🧬 2. Phổ nhiệt Trọng số 768 Chiều của từng Vector (Vector Weight Spectrum)")
    st.markdown("Mỗi vector 768 chiều được trải thành ma trận $24 \\times 32$ để quan sát phổ phân bố số thực float32.")
    
    selected_idx = st.selectbox(
        "Chọn Điều luật cần soi phổ vector 768 chiều:",
        options=list(range(len(all_metadata))),
        format_func=lambda i: f"Vector #{i}: {all_metadata[i].get('header', '')} ({all_metadata[i].get('source', '')})"
    )

    col_meta, col_heat = st.columns([1, 2])
    with col_meta:
        meta = all_metadata[selected_idx]
        vec = all_vectors[selected_idx]
        st.markdown(f"**Tên Điều**: `{meta.get('header')}`")
        st.markdown(f"**Văn bản nguồn**: `{meta.get('source')}`")
        st.markdown(f"**Vị trí**: `{meta.get('location')}`")
        st.markdown(f"**Chuẩn L2 Norm**: `{np.linalg.norm(vec):.4f}`")
        st.markdown(f"**Giá trị trung bình**: `{vec.mean():.6f}`")
        st.markdown(f"**Độ lệch chuẩn (Std)**: `{vec.std():.6f}`")
        with st.expander("📄 Xem toàn văn Điều luật"):
            st.text(meta.get("text", ""))

    with col_heat:
        fig_weight = visualizer.plot_vector_weights_heatmap(selected_idx)
        st.plotly_chart(fig_weight, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 3. Bảng Dữ liệu Toàn bộ Vector trong Database")
    
    search_keyword = st.text_input("🔍 Tìm kiếm Điều luật theo từ khóa trong bảng:")
    
    table_data = []
    for i, meta in enumerate(all_metadata):
        header = meta.get("header", "")
        text = meta.get("text", "")
        if search_keyword.lower() in header.lower() or search_keyword.lower() in text.lower():
            vec = all_vectors[i]
            table_data.append({
                "Vector ID": f"#{i}",
                "Tiêu đề Điều": header,
                "Văn bản nguồn": meta.get("source", ""),
                "Số ký tự": len(text),
                "L2 Norm": round(float(np.linalg.norm(vec)), 4),
                "Mẫu 5 số đầu float32": str(vec[:5].round(4).tolist()),
                "Trích đoạn": text[:100] + "...",
            })

    st.dataframe(table_data, use_container_width=True)
