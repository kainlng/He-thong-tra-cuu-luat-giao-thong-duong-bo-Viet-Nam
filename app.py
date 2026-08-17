import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import streamlit as st

from src.generation.agent import LegalRAGAgent
from src.storage.postgres_manager import PostgresManager
from src.storage.qdrant_manager import QdrantManager
from src.visualization.vector_visualizer import VectorVisualizer

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="LegalGPT - Trợ lý Luật Giao thông Việt Nam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. CHATGPT-STYLE REFINED CSS DESIGN SYSTEM
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main Container Centering & Width with Bottom Padding for Sticky Chat */
    .block-container {
        max-width: 960px !important;
        padding-top: 0.75rem !important;
        padding-bottom: 140px !important;
    }

    /* ==================== TOP COMPACT HEADER ==================== */
    .app-top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 0.85rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
    }
    .app-branding {
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .app-logo-badge {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2);
    }
    .app-title-text {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .app-subtitle-text {
        font-size: 0.70rem;
        color: #64748b;
        margin: 0;
        font-weight: 500;
    }
    .system-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 0.18rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 600;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* ==================== TABS (CRISP & LEGIBLE) ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        border-bottom: 2px solid #e2e8f0 !important;
        padding-bottom: 0px !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        opacity: 1 !important;
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-bottom: none !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1e3a8a !important;
        background: #e2e8f0 !important;
    }
    .stTabs [data-baseweb="tab"] p, 
    .stTabs [data-baseweb="tab"] span, 
    .stTabs [data-baseweb="tab"] div {
        color: #334155 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        filter: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1d4ed8 !important;
        background: #ffffff !important;
        border-top: 2.5px solid #2563eb !important;
        border-left: 1px solid #cbd5e1 !important;
        border-right: 1px solid #cbd5e1 !important;
        border-bottom: 2px solid #ffffff !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] p, 
    .stTabs [aria-selected="true"] span {
        color: #1d4ed8 !important;
        font-weight: 700 !important;
    }

    /* ==================== SIDEBAR STYLING (FIX CONTRAST) ==================== */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0f172a;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span {
        color: #1e293b !important;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-align: left !important;
        padding: 0.4rem 0.65rem !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #1e3a8a !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        margin-bottom: 0.6rem !important;
    }
    .sidebar-footer {
        margin-top: 1.5rem;
        padding-top: 0.75rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.72rem;
        color: #64748b !important;
        text-align: center;
    }
    .sidebar-footer code {
        background: #e2e8f0 !important;
        color: #0f172a !important;
        padding: 0.1rem 0.35rem !important;
        border-radius: 4px !important;
        font-size: 0.72rem !important;
    }

    /* ==================== HERO EMPTY STATE ==================== */
    .hero-container {
        text-align: center;
        padding: 1.5rem 1rem 1.25rem 1rem;
        max-width: 680px;
        margin: 0 auto;
    }
    .hero-icon {
        font-size: 1.8rem;
        margin-bottom: 0.6rem;
        display: inline-block;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        width: 52px;
        height: 52px;
        line-height: 52px;
        border-radius: 14px;
        border: 1px solid #bfdbfe;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    .hero-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }
    .hero-description {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.5;
        margin-bottom: 1.25rem;
    }

    /* ==================== CHAT MESSAGES ==================== */
    .stChatMessage {
        border-radius: 14px !important;
        padding: 0.95rem 1.15rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid rgba(226, 232, 240, 0.9) !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
        line-height: 1.6 !important;
        font-size: 0.92rem !important;
    }
    
    /* Citation Cards */
    .citation-card-box {
        border-left: 3.5px solid #2563eb;
        background: #f8fafc;
        border-radius: 0 8px 8px 0;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }
    .citation-header-title {
        font-weight: 600;
        color: #0f172a;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 4px;
    }
    .citation-meta-row {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 0.25rem;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .citation-tag {
        background: #e2e8f0;
        color: #334155;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.72rem;
    }

    /* Badges */
    .badge-active {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.12rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 600;
        border: 1px solid #bbf7d0;
        display: inline-flex;
        align-items: center;
        gap: 3px;
    }
    .badge-expired {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.12rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 600;
        border: 1px solid #fecaca;
        display: inline-flex;
        align-items: center;
        gap: 3px;
    }

    /* Verification Box */
    .verification-box-success {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.6rem 0;
        font-size: 0.82rem;
        color: #166534;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .verification-box-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.6rem 0;
        font-size: 0.82rem;
        color: #92400e;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ==================== STICKY FIXED CHAT INPUT (BOTTOM CENTER) ==================== */
    [data-testid="stBottom"] {
        background: transparent !important;
        z-index: 9999 !important;
        padding-bottom: 0.75rem !important;
    }
    [data-testid="stBottom"] > div {
        max-width: 900px !important;
        margin: 0 auto !important;
        padding: 0 1rem !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.12) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 8px 28px -2px rgba(37, 99, 235, 0.22) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #0f172a !important;
        font-size: 0.92rem !important;
    }
    
    .chat-disclaimer {
        text-align: center;
        font-size: 0.72rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. INITIALIZATION & SESSION STATE MANAGEMENT
# ==============================================================================
@st.cache_resource(show_spinner="⚡ Đang khởi tạo Trợ lý AI, PostgreSQL, Qdrant và Embeddings...")
def get_system_resources():
    agent = LegalRAGAgent()
    pg_mgr = agent.postgres
    qdrant_mgr = agent.qdrant
    return agent, pg_mgr, qdrant_mgr


try:
    agent, pg_mgr, qdrant_mgr = get_system_resources()
except Exception as e:
    st.error(f"❌ Không thể khởi động hệ thống Legal RAG: {e}")
    st.info("Gợi ý: Kiểm tra file .env, Docker containers PostgreSQL và Qdrant đang hoạt động.")
    st.stop()

# Initialize multi-session management
if "sessions" not in st.session_state:
    initial_session_id = str(uuid.uuid4())[:8]
    st.session_state.sessions = {
        initial_session_id: {
            "title": "Cuộc trò chuyện mới",
            "created_at": datetime.now(),
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "Xin chào! Tôi là **LegalGPT** - Trợ lý AI chuyên sâu về Pháp luật Giao thông Đường bộ Việt Nam.\n\n"
                        "Tôi tra cứu và trích dẫn chuẩn xác theo kiến trúc **PostgreSQL (Source of Truth) + Qdrant (Vector Index Phân cấp)**. "
                        "Bạn có thể đặt câu hỏi về mức xử phạt, quy định hiệu lực mới nhất 2024-2025 hoặc chọn các câu hỏi gợi ý bên dưới."
                    ),
                    "citations": [],
                    "verification": None,
                }
            ],
        }
    }
    st.session_state.current_session_id = initial_session_id

# Ensure active session exists
if st.session_state.current_session_id not in st.session_state.sessions:
    st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]

current_session = st.session_state.sessions[st.session_state.current_session_id]

# Pending prompt handler for suggestions
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==============================================================================
# 4. SIDEBAR (CHATGPT NAVIGATION, SESSIONS & CONTROLS)
# ==============================================================================
with st.sidebar:
    # New Chat Button
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = {
            "title": "Cuộc trò chuyện mới",
            "created_at": datetime.now(),
            "messages": [
                {
                    "role": "assistant",
                    "content": "Tôi có thể giúp bạn giải đáp điều gì về Luật Giao thông hôm nay?",
                    "citations": [],
                    "verification": None,
                }
            ],
        }
        st.session_state.current_session_id = new_id
        st.rerun()

    st.markdown("### 💬 Lịch sử hội thoại")
    # List sessions with delete option
    session_ids = list(st.session_state.sessions.keys())
    for sid in reversed(session_ids):
        s_data = st.session_state.sessions[sid]
        col_btn, col_del = st.columns([5, 1])
        with col_btn:
            is_active = (sid == st.session_state.current_session_id)
            btn_label = f"📌 {s_data['title']}" if is_active else f"💬 {s_data['title']}"
            if st.button(btn_label, key=f"session_btn_{sid}", use_container_width=True, disabled=is_active):
                st.session_state.current_session_id = sid
                st.rerun()
        with col_del:
            if len(st.session_state.sessions) > 1:
                if st.button("🗑️", key=f"del_sess_{sid}", help="Xóa đoạn chat này"):
                    del st.session_state.sessions[sid]
                    if st.session_state.current_session_id == sid:
                        st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
                    st.rerun()

    st.markdown("---")

    # Legal Validity & RAG Search Filter
    with st.expander("⚙️ Bộ lọc & Cấu hình RAG", expanded=True):
        search_mode = st.radio(
            "Hiệu lực văn bản:",
            ["🟢 Quy định hiện hành (Active)", "🕒 Theo mốc thời gian tùy chọn", "🌐 Toàn bộ phiên bản"],
            index=0,
        )

        force_current = True
        if "tùy chọn" in search_mode:
            custom_date = st.date_input("Mốc thời gian áp dụng:", value=datetime.today())
            force_current = None
        elif "Toàn bộ" in search_mode:
            force_current = False

        top_n_rerank = st.slider("Số lượng Điều luật trích dẫn (Top-N):", min_value=1, max_value=8, value=3)

    # Database Statistics
    with st.expander("📊 Thống kê Cơ sở Dữ liệu", expanded=False):
        stats = pg_mgr.get_database_stats()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Văn bản", stats.get("documents", 0))
            st.metric("Chương", stats.get("chapters", 0))
            st.metric("Khoản", stats.get("clauses", 0))
        with c2:
            st.metric("Phiên bản", stats.get("versions", 0))
            st.metric("Điều luật", stats.get("articles", 0))
            st.metric("Chunks", stats.get("chunks", 0))

    # Sidebar Footer Information
    st.markdown(
        """
        <div class="sidebar-footer">
            <b>LegalGPT v2.0</b> • Powered by<br>
            <code>Qwen 2.5 72B</code> • <code>e5-base</code><br>
            PostgreSQL 16 & Qdrant Engine
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 5. HELPER RENDERING FUNCTIONS
# ==============================================================================
def render_verification_badge(verification: dict):
    if not verification:
        return
    is_valid = verification.get("is_valid", True)
    score = verification.get("score", 1.0)
    cited = verification.get("cited_articles", [])
    mismatched = verification.get("mismatched_articles", [])
    retried = verification.get("retried", False)

    if is_valid:
        st.markdown(
            f"""
            <div class="verification-box-success">
                <span>🛡️</span>
                <div>
                    <b>Xác thực PostgreSQL Source-of-Truth:</b> 100% khớp căn cứ pháp lý 
                    <code>({', '.join([f'Điều {a}' for a in cited]) if cited else 'Tổng hợp'})</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verification-box-warning">
                <span>⚠️</span>
                <div>
                    <b>Cảnh báo đối soát:</b> Phát hiện điều luật chưa đối khớp trong CSDL: 
                    <code>{', '.join(mismatched)}</code>
                    {'(Đã tự động kích hoạt fallback retry)' if retried else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_citations_block(citations: list):
    if not citations:
        return

    with st.expander(f"📜 Căn cứ Pháp lý & Trích dẫn Nguồn ({len(citations)} Điều luật liên quan)", expanded=False):
        for c in citations:
            status_badge = (
                '<span class="badge-active">🟢 ĐANG ÁP DỤNG</span>'
                if c.get("status") == "active"
                else '<span class="badge-expired">🔴 HẾT HIỆU LỰC</span>'
            )
            score_val = c.get("score", 0.0)
            st.markdown(
                f"""
                <div class="citation-card-box">
                    <div class="citation-header-title">
                        <span><b>{c.get('citation_text')}</b></span>
                        {status_badge}
                    </div>
                    <div class="citation-meta-row">
                        <span class="citation-tag">🎯 Relevance: {score_val:.4f}</span>
                        <span class="citation-tag">📁 Nguồn: {c.get('source_file')}</span>
                        <span class="citation-tag">📅 Áp dụng: {c.get('effective_from')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"📄 Toàn văn Điều {c.get('article_number')}: {c.get('article_title')}"):
                st.markdown(c.get("full_text", "_Không có nội dung chi tiết._"))


# ==============================================================================
# 6. MAIN APPLICATION HEADER & TABS
# ==============================================================================
st.markdown(
    """
    <div class="app-top-header">
        <div class="app-branding">
            <div class="app-logo-badge">⚖️</div>
            <div>
                <h1 class="app-title-text">LegalGPT • Trợ lý Pháp luật Giao thông Đường bộ</h1>
                <p class="app-subtitle-text">Kiến trúc RAG Phân cấp: PostgreSQL 16 (Source of Truth) & Qdrant v1.9 Vector Search</p>
            </div>
        </div>
        <div class="system-status-pill">
            <span class="status-dot"></span>
            <span>Hệ thống Hoạt động</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_vector_map, tab_database_explorer, tab_about = st.tabs([
    "💬 Trò chuyện AI (ChatGPT View)",
    "🌌 Không gian Vector 3D/2D (Qdrant)",
    "🏛️ Khám phá CSDL (PostgreSQL)",
    "ℹ️ Kiến trúc Hệ thống",
])


# ==============================================================================
# TAB 1: CHATGPT-STYLE CONVERSATION
# ==============================================================================
with tab_chat:
    messages = current_session["messages"]
    has_user_message = any(m["role"] == "user" for m in messages)

    # 1. HERO EMPTY STATE & SUGGESTION CARDS
    if not has_user_message:
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-icon">⚖️</div>
                <h2 class="hero-title">Tôi có thể giúp bạn tra cứu gì hôm nay?</h2>
                <p class="hero-description">
                    Hỏi bất kỳ quy định nào về mức xử phạt, quy tắc tham gia giao thông, bằng lái hoặc thủ tục đăng ký xe theo Luật Trật tự, An toàn giao thông đường bộ mới nhất.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Prompt Suggestions Grid
        col_p1, col_p2 = st.columns(2)
        suggestions = [
            ("🚦 Mức phạt vượt đèn đỏ", "Tra cứu mức phạt cho xe máy và ô tô theo quy định hiện hành", "Vượt đèn đỏ đối với xe máy và ô tô bị xử phạt bao nhiêu tiền và có bị tước bằng lái không?"),
            ("🍺 Quy định nồng độ cồn", "Khung xử phạt nồng độ cồn và thời gian tước GPLX mới nhất", "Mức xử phạt vi phạm nồng độ cồn khi lái xe máy và ô tô hiện nay quy định thế nào?"),
            ("🏍️ Độ tuổi & Giấy phép lái xe", "Quy định độ tuổi điều khiển xe máy 50cc và thi bằng lái A1", "Bao nhiêu tuổi được phép điều khiển xe máy 50cc và điều kiện thi bằng lái xe máy là gì?"),
            ("📋 Thủ tục sang tên & Biển số", "Quy định về biển số định danh và thủ tục chuyển nhượng xe", "Thủ tục sang tên đổi chủ xe và quy định về biển số xe định danh hiện nay như thế nào?"),
        ]

        with col_p1:
            if st.button(f"**{suggestions[0][0]}**\n\n{suggestions[0][1]}", key="sugg_0", use_container_width=True):
                st.session_state.pending_prompt = suggestions[0][2]
                st.rerun()
            if st.button(f"**{suggestions[1][0]}**\n\n{suggestions[1][1]}", key="sugg_1", use_container_width=True):
                st.session_state.pending_prompt = suggestions[1][2]
                st.rerun()

        with col_p2:
            if st.button(f"**{suggestions[2][0]}**\n\n{suggestions[2][1]}", key="sugg_2", use_container_width=True):
                st.session_state.pending_prompt = suggestions[2][2]
                st.rerun()
            if st.button(f"**{suggestions[3][0]}**\n\n{suggestions[3][1]}", key="sugg_3", use_container_width=True):
                st.session_state.pending_prompt = suggestions[3][2]
                st.rerun()

    # 2. RENDER CHAT HISTORY
    for msg in messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
            st.markdown(msg["content"])
            if msg.get("verification"):
                render_verification_badge(msg["verification"])
            if msg.get("citations"):
                render_citations_block(msg["citations"])

    # 3. CHAT INPUT & PROMPT HANDLING
    user_input = st.chat_input("Hỏi LegalGPT về Luật Giao thông (ví dụ: Mức phạt đi ngược chiều xe máy?)...")

    # Check if there is a pending prompt from suggestion card
    active_prompt = None
    if user_input:
        active_prompt = user_input
    elif st.session_state.pending_prompt:
        active_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if active_prompt:
        # Update session title if first message
        if not has_user_message:
            short_title = active_prompt[:26] + ("..." if len(active_prompt) > 26 else "")
            current_session["title"] = short_title

        # Append User Message
        current_session["messages"].append({"role": "user", "content": active_prompt})
        st.session_state.last_query = active_prompt

        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        # Assistant Thinking & RAG Process
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("🔍 Đang truy vấn Qdrant Vector Filter, mở rộng cây phân cấp PostgreSQL và Rerank..."):
                res = agent.answer(
                    query=active_prompt,
                    top_n=top_n_rerank,
                    force_current=force_current,
                )
                answer_text = res["answer"]
                citations = res.get("citations", [])
                verification = res.get("verification")

                st.markdown(answer_text)

                if verification:
                    render_verification_badge(verification)

                if citations:
                    render_citations_block(citations)

        # Save Assistant Response to Session
        current_session["messages"].append({
            "role": "assistant",
            "content": answer_text,
            "citations": citations,
            "verification": verification,
        })
        st.rerun()

    # Footer Disclaimer
    st.markdown(
        '<div class="chat-disclaimer">LegalGPT có thể mắc lỗi. Vui lòng đối chiếu với văn bản quy phạm pháp luật chính thức.</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 2: VECTOR SPACE & QDRANT EXPLORER
# ==============================================================================
with tab_vector_map:
    st.markdown("### 🌌 Bản đồ Không gian Vector Qdrant & Phân cụm Ngữ nghĩa")
    st.markdown(
        "Mỗi điểm trên không gian 3D/2D đại diện cho một **Semantic Legal Chunk** được vector hóa từ văn bản luật "
        "bằng mô hình `intfloat/multilingual-e5-base` (768 chiều)."
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
        st.warning("Chưa có dữ liệu chunks trong Database. Vui lòng chạy Ingestion Pipeline: `python src/ingest_pipeline.py`.")


# ==============================================================================
# TAB 3: POSTGRESQL DATABASE EXPLORER
# ==============================================================================
with tab_database_explorer:
    st.markdown("### 🏛️ Trình Khám phá Cây Cấu trúc Dữ liệu Pháp luật (PostgreSQL)")
    st.markdown(
        "Dữ liệu pháp luật được lưu trữ toàn vẹn trong **7 bảng quan hệ** PostgreSQL. "
        "Duyệt danh mục văn bản, phiên bản hiệu lực và chi tiết cấu trúc phân cấp Chương → Điều → Khoản → Điểm."
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


# ==============================================================================
# TAB 4: SYSTEM ARCHITECTURE & ABOUT
# ==============================================================================
with tab_about:
    st.markdown("### ℹ️ Kiến trúc Hệ thống Legal RAG Phân cấp")
    st.markdown(
        """
        Hệ thống được thiết kế theo tiêu chuẩn công nghiệp nhằm giải quyết triệt để vấn đề ảo giác (hallucination) 
        trong tra cứu văn bản pháp luật bằng các thành phần cốt lõi:

        - **Source of Truth (PostgreSQL 16)**: Lưu trữ 100% dữ liệu gốc chuẩn hóa theo 7 bảng quan hệ (`documents`, `versions`, `chapters`, `articles`, `clauses`, `points`, `chunks`).
        - **Vector Database (Qdrant v1.9)**: Đánh chỉ mục Dense Vector với bộ lọc Payload đa tầng (`status`, `effective_from`, `topics`).
        - **Embedding Model**: `intfloat/multilingual-e5-base` (768 chiều), tối ưu hóa biểu diễn ngữ nghĩa tiếng Việt.
        - **Hierarchical Context Expander**: Tự động liên kết ngữ cảnh từ Khoản/Điểm lên toàn văn Điều luật để LLM hiểu trọn vẹn quy định.
        - **Reranker (FlashRank / Cross-Encoder)**: Tái xếp hạng các đoạn văn bản quan trọng nhất trước khi đưa vào LLM.
        - **Large Language Model**: `Qwen/Qwen2.5-72B-Instruct` qua Hugging Face Inference API.
        - **Deterministic Verification Audit**: Đối soát tự động số Điều luật trong câu trả lời với CSDL thực tế để đảm bảo độ chính xác 100%.
        """
    )
