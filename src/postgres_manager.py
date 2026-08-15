import sys
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError

from src.config import POSTGRES_URL
from src.models import (
    Base,
    LegalDocumentORM,
    LegalVersionORM,
    LegalChapterORM,
    LegalArticleORM,
    LegalClauseORM,
    LegalPointORM,
    LegalChunkORM,
)

logger = logging.getLogger(__name__)


class PostgresManager:
    """Quản trị cơ sở dữ liệu quan hệ PostgreSQL - Source of Truth cho toàn bộ hệ thống Legal RAG."""

    def __init__(self, db_url: str = POSTGRES_URL):
        self.db_url = db_url
        self._is_connected = False

        try:
            self.engine = create_engine(
                self.db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False,
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._is_connected = True
            print(f"[PostgreSQL] Kết nối thành công tới Database: {self.db_url.split('@')[-1]}")
        except Exception as exc:
            print(f"[PostgreSQL Cảnh báo] Không thể kết nối tới PostgreSQL ({exc}). Đang chuyển sang SQLite Local...")
            from src.config import VECTOR_DB_DIR
            sqlite_path = VECTOR_DB_DIR / "legal_relational_fallback.db"
            self.db_url = f"sqlite:///{sqlite_path}"
            self.engine = create_engine(self.db_url, echo=False)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self._is_connected = False

    def init_db(self):
        """Khởi tạo toàn bộ 7 bảng quan hệ và chỉ mục trong Database."""
        Base.metadata.create_all(bind=self.engine)
        print("[PostgreSQL] Đã khởi tạo thành công 7 bảng quan hệ và chỉ mục.")

    def reset_db(self):
        """Xóa và tạo lại cấu trúc bảng (dùng khi re-index)."""
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        print("[PostgreSQL] Đã làm mới (reset) toàn bộ cơ sở dữ liệu.")

    def get_session(self) -> Session:
        return self.SessionLocal()

    def check_document_exists_by_hash(self, file_hash: str) -> bool:
        """Kiểm tra xem file PDF / markdown này đã được ingest trước đó hay chưa."""
        if not file_hash:
            return False
        with self.get_session() as session:
            # Query JSON source column for file_hash
            docs = session.query(LegalDocumentORM).all()
            for doc in docs:
                if doc.source and doc.source.get("file_hash") == file_hash:
                    return True
        return False

    def insert_legal_tree(self, parsed_tree: Dict[str, Any], chunks: List[Dict[str, Any]]) -> bool:
        """Thực hiện lưu toàn bộ cây cấu trúc văn bản pháp luật vào 7 bảng trong 1 Transaction nguyên tử."""
        doc_data = parsed_tree["document"]
        ver_data = parsed_tree["version"]
        chap_list = parsed_tree["chapters"]
        art_list = parsed_tree["articles"]
        cl_list = parsed_tree["clauses"]
        pt_list = parsed_tree["points"]

        with self.get_session() as session:
            try:
                # 1. Document
                existing_doc = session.query(LegalDocumentORM).filter_by(id=doc_data["id"]).first()
                if existing_doc:
                    session.delete(existing_doc)
                    session.flush()

                # Convert string dates to date objects if needed
                issued_d = datetime.strptime(doc_data["issued_date"], "%Y-%m-%d").date() if doc_data.get("issued_date") else None
                first_eff = datetime.strptime(doc_data["first_effective_date"], "%Y-%m-%d").date() if doc_data.get("first_effective_date") else None

                # Cache valid IDs for strict Foreign Key validation
                valid_chapter_ids = {chap["id"] for chap in chap_list}
                valid_article_ids = {art["id"] for art in art_list}
                valid_clause_ids = {cl["id"] for cl in cl_list}
                valid_point_ids = {pt["id"] for pt in pt_list}

                # 1. Documents
                doc_orm = LegalDocumentORM(
                    id=doc_data["id"],
                    document_number=doc_data["document_number"],
                    title=doc_data["title"],
                    short_title=doc_data.get("short_title"),
                    document_type=doc_data.get("document_type", "luat"),
                    issuing_agency=doc_data.get("issuing_agency", {}),
                    domain=doc_data.get("domain", []),
                    issued_date=issued_d,
                    first_effective_date=first_eff,
                    current_status=doc_data.get("current_status", "active"),
                    language=doc_data.get("language", "vi"),
                    source=doc_data.get("source", {}),
                )
                session.add(doc_orm)
                session.flush()

                # 2. Version
                eff_from = datetime.strptime(ver_data["effective_from"], "%Y-%m-%d").date() if ver_data.get("effective_from") else date(2025, 1, 1)
                eff_to = datetime.strptime(ver_data["effective_to"], "%Y-%m-%d").date() if ver_data.get("effective_to") else None

                ver_orm = LegalVersionORM(
                    id=ver_data["id"],
                    document_id=ver_data["document_id"],
                    version_number=ver_data.get("version_number", 1),
                    effective_from=eff_from,
                    effective_to=eff_to,
                    status=ver_data.get("status", "active"),
                    is_current=ver_data.get("is_current", True),
                    amended_by=ver_data.get("amended_by", []),
                    amends=ver_data.get("amends", []),
                    replaces=ver_data.get("replaces"),
                    legal_basis=ver_data.get("legal_basis", []),
                )
                session.add(ver_orm)
                session.flush()

                # 3. Chapters
                for chap in chap_list:
                    chap_orm = LegalChapterORM(
                        id=chap["id"],
                        document_id=chap["document_id"],
                        version_id=chap["version_id"],
                        chapter_number=chap.get("chapter_number"),
                        title=chap["title"],
                        order_num=chap.get("order_num", 1),
                        parent_id=chap.get("parent_id"),
                    )
                    session.add(chap_orm)
                session.flush()

                # 4. Articles
                for art in art_list:
                    chap_ref = art.get("chapter_id") if art.get("chapter_id") in valid_chapter_ids else None
                    art_orm = LegalArticleORM(
                        id=art["id"],
                        document_id=art["document_id"],
                        version_id=art["version_id"],
                        chapter_id=chap_ref,
                        article_number=art["article_number"],
                        title=art["title"],
                        order_num=art.get("order_num", 1),
                        parent_id=art.get("parent_id"),
                    )
                    session.add(art_orm)
                session.flush()

                # 5. Clauses
                for cl in cl_list:
                    chap_ref = cl.get("chapter_id") if cl.get("chapter_id") in valid_chapter_ids else None
                    cl_orm = LegalClauseORM(
                        id=cl["id"],
                        document_id=cl["document_id"],
                        version_id=cl["version_id"],
                        chapter_id=chap_ref,
                        article_id=cl["article_id"],
                        clause_number=cl["clause_number"],
                        content=cl["content"],
                        order_num=cl.get("order_num", 1),
                        parent_id=cl.get("parent_id"),
                    )
                    session.add(cl_orm)
                session.flush()

                # 6. Points
                for pt in pt_list:
                    chap_ref = pt.get("chapter_id") if pt.get("chapter_id") in valid_chapter_ids else None
                    pt_orm = LegalPointORM(
                        id=pt["id"],
                        document_id=pt["document_id"],
                        version_id=pt["version_id"],
                        chapter_id=chap_ref,
                        article_id=pt["article_id"],
                        clause_id=pt["clause_id"],
                        point_label=pt["point_label"],
                        content=pt["content"],
                        order_num=pt.get("order_num", 1),
                        parent_id=pt.get("parent_id"),
                    )
                    session.add(pt_orm)
                session.flush()

                # 7. Chunks - Tách Root Chunks (parent=None) chèn trước, Leaf Chunks chèn sau
                root_chunks = [c for c in chunks if not c.get("parent_chunk_id")]
                leaf_chunks = [c for c in chunks if c.get("parent_chunk_id")]

                def create_chunk_orm(chk: Dict[str, Any], is_root: bool = False) -> LegalChunkORM:
                    chap_ref = chk.get("chapter_id") if chk.get("chapter_id") in valid_chapter_ids else None
                    art_ref = chk["article_id"] if chk["article_id"] in valid_article_ids else None
                    cl_ref = chk.get("clause_id") if chk.get("clause_id") in valid_clause_ids else None
                    pt_ref = chk.get("point_id") if chk.get("point_id") in valid_point_ids else None

                    return LegalChunkORM(
                        id=chk["id"],
                        document_id=chk["document_id"],
                        version_id=chk["version_id"],
                        chapter_id=chap_ref,
                        article_id=art_ref or art_list[0]["id"],
                        clause_id=cl_ref,
                        point_id=pt_ref,
                        parent_chunk_id=None if is_root else chk.get("parent_chunk_id"),
                        chunk_type=chk["chunk_type"],
                        chunk_index=chk.get("chunk_index", 1),
                        content=chk["content"],
                        normalized_content=chk.get("normalized_content"),
                        token_count=chk.get("token_count", 0),
                        metadata_json=chk.get("metadata", {}),
                        validity=chk.get("validity", {}),
                        source=chk.get("source", {}),
                        search=chk.get("search", {}),
                    )

                # Chèn Root Chunks
                for chk in root_chunks:
                    session.add(create_chunk_orm(chk, is_root=True))
                session.flush()

                # Chèn Leaf Chunks
                for chk in leaf_chunks:
                    session.add(create_chunk_orm(chk, is_root=False))
                session.flush()

                session.commit()
                print(f"[PostgreSQL] Đã lưu thành công cây văn bản {doc_data['document_number']}: "
                      f"{len(art_list)} Điều, {len(cl_list)} Khoản, {len(pt_list)} Điểm, {len(chunks)} Chunks.")
                return True
            except Exception as exc:
                session.rollback()
                print(f"[PostgreSQL Lỗi Transaction]: {exc}")
                raise exc

    def get_hierarchical_context_by_chunk_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Truy xuất toàn bộ thông tin cây phân cấp từ danh sách chunk_id trả về từ Qdrant.
        JOIN: legal_chunks -> legal_articles -> legal_chapters -> legal_documents.
        """
        if not chunk_ids:
            return []

        results = []
        with self.get_session() as session:
            for chunk_id in chunk_ids:
                chunk = session.query(LegalChunkORM).filter_by(id=chunk_id).first()
                if not chunk:
                    continue

                art = session.query(LegalArticleORM).filter_by(id=chunk.article_id).first()
                doc = session.query(LegalDocumentORM).filter_by(id=chunk.document_id).first()
                chap = session.query(LegalChapterORM).filter_by(id=chunk.chapter_id).first() if chunk.chapter_id else None

                # Lấy toàn bộ các Khoản của Điều này để làm rich context
                clauses = session.query(LegalClauseORM).filter_by(article_id=chunk.article_id).order_by(LegalClauseORM.order_num).all()
                points = session.query(LegalPointORM).filter_by(article_id=chunk.article_id).order_by(LegalPointORM.order_num).all()

                results.append({
                    "chunk_id": chunk.id,
                    "chunk_type": chunk.chunk_type,
                    "content": chunk.content,
                    "document": {
                        "id": doc.id if doc else "",
                        "document_number": doc.document_number if doc else "",
                        "title": doc.title if doc else "",
                        "short_title": doc.short_title if doc else "",
                        "document_type": doc.document_type if doc else "luat",
                    },
                    "chapter": {
                        "chapter_number": chap.chapter_number if chap else "",
                        "title": chap.title if chap else "",
                    },
                    "article": {
                        "id": art.id if art else "",
                        "article_number": art.article_number if art else "",
                        "title": art.title if art else "",
                        "clause_count": len(clauses),
                        "clauses": [{"number": c.clause_number, "content": c.content} for c in clauses],
                        "points": [{"clause_id": p.clause_id, "label": p.point_label, "content": p.content} for p in points],
                    },
                    "validity": chunk.validity or {},
                    "source": chunk.source or {},
                    "metadata": chunk.metadata_json or {},
                })

        return results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Lấy danh sách toàn bộ chunks để đồng bộ hoặc trực quan hóa."""
        with self.get_session() as session:
            chunks = session.query(LegalChunkORM).all()
            return [
                {
                    "id": c.id,
                    "document_id": c.document_id,
                    "article_id": c.article_id,
                    "chunk_type": c.chunk_type,
                    "content": c.content,
                    "metadata": c.metadata_json,
                    "validity": c.validity,
                    "source": c.source,
                }
                for c in chunks
            ]

    def get_database_stats(self) -> Dict[str, int]:
        """Thống kê tổng số lượng thực thể đang được lưu trong PostgreSQL."""
        with self.get_session() as session:
            return {
                "documents": session.query(func.count(LegalDocumentORM.id)).scalar() or 0,
                "versions": session.query(func.count(LegalVersionORM.id)).scalar() or 0,
                "chapters": session.query(func.count(LegalChapterORM.id)).scalar() or 0,
                "articles": session.query(func.count(LegalArticleORM.id)).scalar() or 0,
                "clauses": session.query(func.count(LegalClauseORM.id)).scalar() or 0,
                "points": session.query(func.count(LegalPointORM.id)).scalar() or 0,
                "chunks": session.query(func.count(LegalChunkORM.id)).scalar() or 0,
            }
