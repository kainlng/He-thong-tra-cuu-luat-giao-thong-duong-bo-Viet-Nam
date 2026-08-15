import sys
import uuid
from typing import List, Dict, Any, Optional
import numpy as np

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, Range
except ImportError:
    QdrantClient = None
    rest_models = None

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_LOCAL_PATH,
    EMBEDDING_DIMENSION,
)


class QdrantManager:
    """Quản trị cơ sở dữ liệu Vector Qdrant - Chỉ lưu Vector và Payload tối giản để Filter."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = QDRANT_COLLECTION_NAME,
        vector_dim: int = EMBEDDING_DIMENSION,
    ):
        self.url = url
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = None

        if QdrantClient is None:
            print("[Qdrant Cảnh báo] Thư viện qdrant-client chưa được cài đặt.")
            return

        try:
            if url.startswith("http://") or url.startswith("https://"):
                print(f"[Qdrant] Đang kết nối tới Qdrant Server: {url}...")
                is_local = "localhost" in url or "127.0.0.1" in url
                effective_api_key = None if is_local else (api_key if api_key else None)
                self.client = QdrantClient(
                    url=url,
                    api_key=effective_api_key,
                    timeout=10,
                    check_compatibility=False,
                )
                # Test connection
                self.client.get_collections()
                print(f"[Qdrant] Kết nối thành công tới Server {url}")
            else:
                raise ValueError("URL không hợp lệ, chuyển sang local storage.")
        except Exception as exc:
            print(f"[Qdrant Cảnh báo] Không thể kết nối tới Qdrant Server ({exc}). Chuyển sang Qdrant Local/In-Memory...")
            try:
                self.client = QdrantClient(path=QDRANT_LOCAL_PATH)
            except Exception:
                self.client = QdrantClient(location=":memory:")

        self._init_collection()

    def _init_collection(self):
        """Tạo collection và Payload Indexes nếu chưa tồn tại."""
        if not self.client:
            return

        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            print(f"[Qdrant] Đang tạo mới Collection: '{self.collection_name}' (dim={self.vector_dim}, metric=Cosine)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            self._create_payload_indexes()
        else:
            print(f"[Qdrant] Collection '{self.collection_name}' đã sẵn sàng.")

    def _create_payload_indexes(self):
        """Đánh chỉ mục trên Payload để tăng tốc truy vấn Pre-filtering."""
        payload_fields = [
            ("status", rest_models.PayloadSchemaType.KEYWORD),
            ("is_current", rest_models.PayloadSchemaType.BOOL),
            ("document_id", rest_models.PayloadSchemaType.KEYWORD),
            ("article_id", rest_models.PayloadSchemaType.KEYWORD),
            ("domain", rest_models.PayloadSchemaType.KEYWORD),
            ("document_type", rest_models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_type in payload_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception:
                pass

    def chunk_id_to_uuid(self, chunk_id: str) -> str:
        """Chuyển đổi chuỗi chunk_id thành UUIDv5 hợp lệ cho Qdrant Point ID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Upsert danh sách Chunks và Vector vào Qdrant với minimal payload (không chứa raw text)."""
        if not self.client or len(chunks) == 0:
            return

        points = []
        for i, chunk in enumerate(chunks):
            point_uuid = self.chunk_id_to_uuid(chunk["id"])
            vector = embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else list(embeddings[i])

            # Tạo payload tối giản
            meta = chunk.get("metadata", {})
            validity = chunk.get("validity", {})
            source = chunk.get("source", {})

            payload = {
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "version_id": chunk["version_id"],
                "chapter_id": chunk.get("chapter_id"),
                "article_id": chunk["article_id"],
                "clause_id": chunk.get("clause_id"),
                "point_id": chunk.get("point_id"),
                "parent_chunk_id": chunk.get("parent_chunk_id"),
                "document_number": meta.get("document_number", ""),
                "article_number": meta.get("article_number", ""),
                "clause_number": meta.get("clause_number"),
                "point_label": meta.get("point_label"),
                "document_type": meta.get("document_type", "luat"),
                "domain": meta.get("domain", "giao_thong_duong_bo"),
                "topics": meta.get("topics", []),
                "violation_types": meta.get("violation_types", []),
                "effective_from": validity.get("effective_from", "2025-01-01"),
                "effective_to": validity.get("effective_to"),
                "status": validity.get("status", "active"),
                "is_current": validity.get("is_current", True),
                "page_start": source.get("page_start", 1),
                "page_end": source.get("page_end", 1),
            }

            points.append(
                rest_models.PointStruct(
                    id=point_uuid,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert to Qdrant
        batch_size = 64
        for start_idx in range(0, len(points), batch_size):
            batch = points[start_idx : start_idx + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        print(f"[Qdrant] Đã upsert thành công {len(points)} vectors vào collection '{self.collection_name}'.")

    def search_candidates(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        status_filter: Optional[str] = "active",
        is_current_only: bool = True,
        document_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm Top K candidate chunks gần nhất kết hợp Metadata Filter."""
        if not self.client:
            return []

        # Xây dựng Qdrant filter conditions
        must_conditions = []

        if status_filter:
            must_conditions.append(
                FieldCondition(key="status", match=MatchValue(value=status_filter))
            )

        if is_current_only:
            must_conditions.append(
                FieldCondition(key="is_current", match=MatchValue(value=True))
            )

        if document_type:
            must_conditions.append(
                FieldCondition(key="document_type", match=MatchValue(value=document_type))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        vec = query_vector[0].tolist() if len(query_vector.shape) > 1 else query_vector.tolist()

        candidates = []
        # 1. Thử dùng API query_points (cho Qdrant >= 1.10)
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vec,
                query_filter=qdrant_filter,
                limit=top_k,
            )
            for hit in response.points:
                candidates.append({
                    "score": float(hit.score),
                    "chunk_id": hit.payload.get("chunk_id"),
                    "article_id": hit.payload.get("article_id"),
                    "document_id": hit.payload.get("document_id"),
                    "document_number": hit.payload.get("document_number"),
                    "topics": hit.payload.get("topics", []),
                    "violation_types": hit.payload.get("violation_types", []),
                    "payload": hit.payload,
                })
            return candidates
        except Exception:
            pass

        # 2. Fallback sang REST HTTP Search API (Tương thích tuyệt đối mọi phiên bản Qdrant v1.x)
        try:
            import urllib.request
            import json

            search_endpoint = f"{self.url.rstrip('/')}/collections/{self.collection_name}/points/search"
            payload_body = {
                "vector": vec,
                "limit": top_k,
                "with_payload": True,
            }
            if must_conditions:
                filter_dict = {"must": []}
                if status_filter:
                    filter_dict["must"].append({"key": "status", "match": {"value": status_filter}})
                if is_current_only:
                    filter_dict["must"].append({"key": "is_current", "match": {"value": True}})
                if document_type:
                    filter_dict["must"].append({"key": "document_type", "match": {"value": document_type}})
                payload_body["filter"] = filter_dict

            req = urllib.request.Request(
                search_endpoint,
                data=json.dumps(payload_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for hit in data.get("result", []):
                    payload = hit.get("payload", {})
                    candidates.append({
                        "score": float(hit.get("score", 0.0)),
                        "chunk_id": payload.get("chunk_id"),
                        "article_id": payload.get("article_id"),
                        "document_id": payload.get("document_id"),
                        "document_number": payload.get("document_number"),
                        "topics": payload.get("topics", []),
                        "violation_types": payload.get("violation_types", []),
                        "payload": payload,
                    })
                return candidates
        except Exception as rest_exc:
            print(f"[Qdrant Cảnh báo] Lỗi tìm kiếm REST ({rest_exc})")
            return []
