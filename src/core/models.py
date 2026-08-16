from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Cross-platform JSON & Array types (native PostgreSQL JSONB/ARRAY with seamless SQLite JSON fallback)
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")
ARRAY_STR_TYPE = JSON().with_variant(ARRAY(String), "postgresql")


# ==============================================================================
# 1. SQLALCHEMY ORM MODELS (PostgreSQL 7 Tables)
# ==============================================================================

class LegalDocumentORM(Base):
    """Văn bản pháp luật gốc (Luật, Nghị định, Thông tư...)."""
    __tablename__ = "legal_documents"

    id = Column(String(64), primary_key=True)
    document_number = Column(String(100), nullable=False, unique=True)
    title = Column(Text, nullable=False)
    short_title = Column(String(255), nullable=True)
    document_type = Column(String(50), nullable=True)  # luat, nghi_dinh, thong_tu...
    issuing_agency = Column(JSON_TYPE, default=dict)  # {"name": "Chính phủ", "code": "CHINH_PHU"}
    domain = Column(ARRAY_STR_TYPE, default=list)  # ["giao_thong_duong_bo", "xu_phat"]
    issued_date = Column(Date, nullable=True)
    first_effective_date = Column(Date, nullable=True)
    current_status = Column(String(50), default="active")  # active, expired, replaced
    language = Column(String(10), default="vi")
    source = Column(JSON_TYPE, default=dict)  # {"file_name": "...", "file_hash": "...", "page_count": 45}
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    versions = relationship("LegalVersionORM", back_populates="document", cascade="all, delete-orphan")
    chapters = relationship("LegalChapterORM", back_populates="document", cascade="all, delete-orphan")
    articles = relationship("LegalArticleORM", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("LegalChunkORM", back_populates="document", cascade="all, delete-orphan")


class LegalVersionORM(Base):
    """Phiên bản và vòng đời hiệu lực của văn bản."""
    __tablename__ = "legal_versions"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, default=1)
    effective_from = Column(Date, nullable=False)
    effective_to = Column(Date, nullable=True)
    status = Column(String(50), default="active")  # active, expired, replaced
    is_current = Column(Boolean, default=True)
    amended_by = Column(JSON_TYPE, default=list)
    amends = Column(JSON_TYPE, default=list)
    replaces = Column(String(64), nullable=True)
    legal_basis = Column(JSON_TYPE, default=list)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    document = relationship("LegalDocumentORM", back_populates="versions")
    chapters = relationship("LegalChapterORM", back_populates="version", cascade="all, delete-orphan")
    articles = relationship("LegalArticleORM", back_populates="version", cascade="all, delete-orphan")


class LegalChapterORM(Base):
    """Chương trong văn bản luật."""
    __tablename__ = "legal_chapters"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False)
    version_id = Column(String(64), ForeignKey("legal_versions.id", ondelete="CASCADE"), nullable=False)
    chapter_number = Column(String(20), nullable=True)  # I, II, III...
    title = Column(Text, nullable=False)
    order_num = Column(Integer, default=1)
    parent_id = Column(String(64), nullable=True)

    # Relationships
    document = relationship("LegalDocumentORM", back_populates="chapters")
    version = relationship("LegalVersionORM", back_populates="chapters")
    articles = relationship("LegalArticleORM", back_populates="chapter")


class LegalArticleORM(Base):
    """Điều trong văn bản luật."""
    __tablename__ = "legal_articles"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False)
    version_id = Column(String(64), ForeignKey("legal_versions.id", ondelete="CASCADE"), nullable=False)
    chapter_id = Column(String(64), ForeignKey("legal_chapters.id", ondelete="SET NULL"), nullable=True)
    article_number = Column(String(20), nullable=False)  # 1, 2, 6...
    title = Column(Text, nullable=False)
    order_num = Column(Integer, default=1)
    parent_id = Column(String(64), nullable=True)

    # Relationships
    document = relationship("LegalDocumentORM", back_populates="articles")
    version = relationship("LegalVersionORM", back_populates="articles")
    chapter = relationship("LegalChapterORM", back_populates="articles")
    clauses = relationship("LegalClauseORM", back_populates="article", cascade="all, delete-orphan")
    points = relationship("LegalPointORM", back_populates="article", cascade="all, delete-orphan")
    chunks = relationship("LegalChunkORM", back_populates="article", cascade="all, delete-orphan")


class LegalClauseORM(Base):
    """Khoản trong Điều luật."""
    __tablename__ = "legal_clauses"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False)
    version_id = Column(String(64), ForeignKey("legal_versions.id", ondelete="CASCADE"), nullable=False)
    chapter_id = Column(String(64), ForeignKey("legal_chapters.id", ondelete="SET NULL"), nullable=True)
    article_id = Column(String(64), ForeignKey("legal_articles.id", ondelete="CASCADE"), nullable=False)
    clause_number = Column(String(20), nullable=False)  # 1, 2, 9...
    content = Column(Text, nullable=False)
    order_num = Column(Integer, default=1)
    parent_id = Column(String(64), nullable=True)

    # Relationships
    article = relationship("LegalArticleORM", back_populates="clauses")
    points = relationship("LegalPointORM", back_populates="clause", cascade="all, delete-orphan")


class LegalPointORM(Base):
    """Điểm trong Khoản của Điều luật."""
    __tablename__ = "legal_points"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False)
    version_id = Column(String(64), ForeignKey("legal_versions.id", ondelete="CASCADE"), nullable=False)
    chapter_id = Column(String(64), ForeignKey("legal_chapters.id", ondelete="SET NULL"), nullable=True)
    article_id = Column(String(64), ForeignKey("legal_articles.id", ondelete="CASCADE"), nullable=False)
    clause_id = Column(String(64), ForeignKey("legal_clauses.id", ondelete="CASCADE"), nullable=False)
    point_label = Column(String(10), nullable=False)  # a, b, c, d, đ...
    content = Column(Text, nullable=False)
    order_num = Column(Integer, default=1)
    parent_id = Column(String(64), nullable=True)

    # Relationships
    article = relationship("LegalArticleORM", back_populates="points")
    clause = relationship("LegalClauseORM", back_populates="points")


class LegalChunkORM(Base):
    """Chunk pháp lý phân cấp - Source of Truth cho RAG."""
    __tablename__ = "legal_chunks"

    id = Column(String(64), primary_key=True)  # Trùng với Qdrant point.id
    document_id = Column(String(64), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False)
    version_id = Column(String(64), ForeignKey("legal_versions.id", ondelete="CASCADE"), nullable=False)
    chapter_id = Column(String(64), ForeignKey("legal_chapters.id", ondelete="SET NULL"), nullable=True)
    article_id = Column(String(64), ForeignKey("legal_articles.id", ondelete="CASCADE"), nullable=False)
    clause_id = Column(String(64), ForeignKey("legal_clauses.id", ondelete="SET NULL"), nullable=True)
    point_id = Column(String(64), ForeignKey("legal_points.id", ondelete="SET NULL"), nullable=True)
    parent_chunk_id = Column(String(64), ForeignKey("legal_chunks.id", ondelete="SET NULL"), nullable=True)
    
    chunk_type = Column(String(30), nullable=False)  # point, clause, article
    chunk_index = Column(Integer, default=1)
    content = Column(Text, nullable=False)
    normalized_content = Column(Text, nullable=True)
    token_count = Column(Integer, default=0)
    
    metadata_json = Column("metadata", JSON_TYPE, default=dict)
    validity = Column(JSON_TYPE, default=dict)  # {"effective_from": ..., "effective_to": ..., "status": "active", "is_current": True}
    source = Column(JSON_TYPE, default=dict)  # {"file_name": ..., "page_start": ..., "page_end": ...}
    search = Column(JSON_TYPE, default=dict)  # {"embedding_model": ..., "qdrant_collection": ...}
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    document = relationship("LegalDocumentORM", back_populates="chunks")
    article = relationship("LegalArticleORM", back_populates="chunks")


# Index definitions for PostgreSQL
Index("idx_articles_doc_ver", LegalArticleORM.document_id, LegalArticleORM.version_id, LegalArticleORM.article_number)
Index("idx_clauses_art", LegalClauseORM.article_id, LegalClauseORM.clause_number)
Index("idx_points_clause", LegalPointORM.clause_id, LegalPointORM.point_label)
Index("idx_chunks_doc_art", LegalChunkORM.document_id, LegalChunkORM.article_id)
Index("idx_chunks_parent", LegalChunkORM.parent_chunk_id)


# ==============================================================================
# 2. PYDANTIC SCHEMAS (Data Validation, API & Qdrant Payloads)
# ==============================================================================

class DocumentMetadata(BaseModel):
    document_number: str
    article_number: Optional[str] = None
    clause_number: Optional[str] = None
    point_label: Optional[str] = None
    document_type: str = "luat"
    domain: str = "giao_thong_duong_bo"
    topics: List[str] = Field(default_factory=list)
    violation_types: List[str] = Field(default_factory=list)


class ValidityMetadata(BaseModel):
    effective_from: str  # YYYY-MM-DD
    effective_to: Optional[str] = None  # YYYY-MM-DD or None
    status: str = "active"  # active, expired, replaced
    is_current: bool = True


class SourceMetadata(BaseModel):
    file_name: str
    file_hash: Optional[str] = None
    page_start: Optional[int] = 1
    page_end: Optional[int] = 1
    source_type: str = "pdf"


class QdrantChunkPayload(BaseModel):
    """Minimal Payload stored in Qdrant (Only filterable fields, no raw text)."""
    chunk_id: str
    document_id: str
    version_id: str
    article_id: str
    clause_id: Optional[str] = None
    point_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    document_number: str
    article_number: str
    clause_number: Optional[str] = None
    point_label: Optional[str] = None
    document_type: str = "luat"
    domain: str = "giao_thong_duong_bo"
    topics: List[str] = Field(default_factory=list)
    violation_types: List[str] = Field(default_factory=list)

    effective_from: str
    effective_to: Optional[str] = None
    status: str = "active"
    is_current: bool = True

    page_start: Optional[int] = 1
    page_end: Optional[int] = 1


class LegalChunkDTO(BaseModel):
    """Full DTO for Ingestion and Synchronization."""
    id: str
    document_id: str
    version_id: str
    chapter_id: Optional[str] = None
    article_id: str
    clause_id: Optional[str] = None
    point_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    
    chunk_type: str  # point, clause, article
    chunk_index: int = 1
    content: str
    normalized_content: Optional[str] = None
    token_count: int = 0
    
    metadata: DocumentMetadata
    validity: ValidityMetadata
    source: SourceMetadata
    search: Dict[str, Any] = Field(default_factory=dict)
