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

    /* Global Typography & White Page Theme */
    html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stHeader"], [data-testid="stMain"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #1d4ed8 !important;
    }

    /* Main Container Centering & Width */
    .block-container {
        max-width: 920px !important;
        padding-top: 0.75rem !important;
        padding-bottom: 140px !important;
        background-color: #ffffff !important;
    }

    /* TOP COMPACT HEADER */
    .app-top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0.9rem;
        background: #ffffff !important;
        border: 1.5px solid #dbeafe !important;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.06);
    }
    .app-branding {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .app-logo-badge {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2);
    }
    .app-title-text {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e3a8a !important;
        margin: 0;
        line-height: 1.2;
    }
    .app-subtitle-text {
        font-size: 0.72rem;
        color: #2563eb !important;
        margin: 0;
        font-weight: 500;
    }
    .system-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        border: 1px solid #bfdbfe !important;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #2563eb;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* SIDEBAR STYLING - WHITE BACKGROUND & BLUE FONT */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1.5px solid #dbeafe !important;
    }
    [data-testid="stSidebar"] * {
        color: #1d4ed8 !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span {
        color: #1d4ed8 !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #ffffff !important;
        color: #1d4ed8 !important;
        border: 1.5px solid #bfdbfe !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-align: left !important;
        padding: 0.4rem 0.65rem !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #eff6ff !important;
        border-color: #2563eb !important;
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
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #2563eb !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1.5px solid #dbeafe !important;
        border-radius: 10px !important;
        margin-bottom: 0.6rem !important;
    }
    .sidebar-footer {
        margin-top: 1.5rem;
        padding-top: 0.75rem;
        border-top: 1px solid #dbeafe;
        font-size: 0.72rem;
        color: #3b82f6 !important;
        text-align: center;
    }
    .sidebar-footer code {
        background: #eff6ff !important;
        color: #1e3a8a !important;
        padding: 0.1rem 0.35rem !important;
        border-radius: 4px !important;
        font-size: 0.72rem !important;
    }

    /* HERO EMPTY STATE */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        max-width: 680px;
        margin: 0 auto;
    }
    .hero-icon {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        display: inline-block;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        width: 60px;
        height: 60px;
        line-height: 60px;
        border-radius: 16px;
        border: 1px solid #bfdbfe;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.15);
    }
    .hero-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1e3a8a !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
        text-shadow: none !important;
    }
    .hero-description {
        font-size: 0.95rem;
        color: #2563eb !important;
        line-height: 1.65;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    /* CHAT MESSAGES - BLUE FONT & WHITE BACKDROP */
    .stChatMessage, [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 1.1rem 1.3rem !important;
        margin-bottom: 1.2rem !important;
        border: 1.5px solid #dbeafe !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.05) !important;
        line-height: 1.65 !important;
        font-size: 0.94rem !important;
        background-color: #ffffff !important;
    }

    /* User Message Bubble - Soft Blue Tinted Box */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #eff6ff !important;
        border: 2px solid #2563eb !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.1) !important;
    }

    /* Assistant Message Bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    [data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) {
        background-color: #ffffff !important;
        border: 1.5px solid #dbeafe !important;
    }

    /* ALL CHAT MESSAGE TEXT IS BLUE */
    .stChatMessage *,
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4 {
        color: #1d4ed8 !important;
        font-weight: 500;
    }

    /* Bold Text - Dark Blue */
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }

    /* Code blocks inside chat messages */
    [data-testid="stChatMessage"] code {
        background-color: #eff6ff !important;
        color: #1e3a8a !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 6px !important;
        padding: 0.15rem 0.4rem !important;
        font-family: monospace !important;
    }

    /* Citation Cards */
    .citation-card-box {
        border-left: 4px solid #2563eb;
        background: #ffffff;
        border-radius: 0 8px 8px 0;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        border-top: 1px solid #dbeafe;
        border-right: 1px solid #dbeafe;
        border-bottom: 1px solid #dbeafe;
    }
    .citation-header-title {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 4px;
    }
    .citation-meta-row {
        font-size: 0.74rem;
        color: #3b82f6;
        margin-top: 0.25rem;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .citation-tag {
        background: #eff6ff;
        color: #1d4ed8;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.72rem;
        border: 1px solid #dbeafe;
    }

    /* Badges */
    .badge-active {
        background-color: #eff6ff;
        color: #1d4ed8;
        padding: 0.12rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 600;
        border: 1px solid #bfdbfe;
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
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.6rem 0;
        font-size: 0.82rem;
        color: #1d4ed8;
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

    /* STICKY FIXED CHAT INPUT (BOTTOM CENTER) - WHITE BG & BLUE BORDER & BLUE TEXT */
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
        background-color: #ffffff !important;
        border: 2px solid #2563eb !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.12) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #1d4ed8 !important;
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.2) !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] p,
    [data-testid="stChatInput"] span {
        color: #1d4ed8 !important;
        caret-color: #2563eb !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #93c5fd !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button {
        color: #2563eb !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: #2563eb !important;
        stroke: #2563eb !important;
    }

    .chat-disclaimer {
        text-align: center;
        font-size: 0.72rem;
        color: #3b82f6;
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
                        "Bạn có thể đặt câu hỏi bất kỳ về quy định hoặc mức xử phạt giao thông bên dưới."
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
# 6. MAIN APPLICATION HEADER
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

# ==============================================================================
# 7. PURE CHAT INTERFACE
# ==============================================================================
messages = current_session["messages"]
has_user_message = any(m["role"] == "user" for m in messages)

# 1. HERO EMPTY STATE
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

# 2. RENDER CHAT HISTORY
for msg in messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
        st.markdown(msg["content"])
        if msg.get("verification"):
            render_verification_badge(msg["verification"])
        if msg.get("citations"):
            render_citations_block(msg["citations"])

# 3. CHAT INPUT & PROMPT HANDLING
active_prompt = st.chat_input("Hỏi LegalGPT về Luật Giao thông (ví dụ: Mức phạt đi ngược chiều xe máy?)...")

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
