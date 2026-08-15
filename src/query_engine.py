import re
from datetime import datetime, date
from typing import List, Dict, Any, Optional

from src.indexer import LlamaIndexEmbedder
from src.qdrant_manager import QdrantManager
from src.postgres_manager import PostgresManager
from src.context_expander import HierarchicalContextExpander
from src.reranker import LegalReranker
from src.config import RETRIEVAL_TOP_K, RERANKER_TOP_N


class LegalQueryEngine:
    """Động cơ truy vấn pháp luật phân cấp kết hợp Qdrant (Filter + Vector) -> PostgreSQL (Context Expansion) -> Reranker."""

    def __init__(
        self,
        embedder: Optional[LlamaIndexEmbedder] = None,
        qdrant_mgr: Optional[QdrantManager] = None,
        postgres_mgr: Optional[PostgresManager] = None,
        reranker: Optional[LegalReranker] = None,
    ):
        self.embedder = embedder or LlamaIndexEmbedder()
        self.qdrant = qdrant_mgr or QdrantManager()
        self.postgres = postgres_mgr or PostgresManager()
        self.expander = HierarchicalContextExpander(self.postgres)
        self.reranker = reranker or LegalReranker()

    def parse_time_intent(self, query: str) -> Dict[str, Any]:
        """Phân tích ý định thời gian trong câu hỏi để áp dụng bộ lọc hiệu lực phù hợp."""
        # 1. Phát hiện năm cụ thể (ví dụ: năm 2023, năm 2020)
        year_match = re.search(r"năm\s+(20\d{2}|19\d{2})", query, re.IGNORECASE)
        if year_match:
            year = int(year_match.group(1))
            query_date = date(year, 6, 1)
            return {
                "is_current_only": False,
                "target_date": query_date.strftime("%Y-%m-%d"),
                "status": None,  # Tìm cả văn bản cũ đã hết hiệu lực tại thời điểm đó
                "reason": f"Phát hiện mốc thời gian: Năm {year}",
            }

        # 2. Phát hiện ngày cụ thể (ví dụ: ngày 01/01/2024)
        date_match = re.search(r"ngày\s+(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", query, re.IGNORECASE)
        if date_match:
            d, m, y = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
            try:
                query_date = date(y, m, d)
                return {
                    "is_current_only": False,
                    "target_date": query_date.strftime("%Y-%m-%d"),
                    "status": None,
                    "reason": f"Phát hiện mốc ngày cụ thể: {query_date}",
                }
            except ValueError:
                pass

        # Mặc định: Tra cứu quy định hiện hành đang có hiệu lực
        return {
            "is_current_only": True,
            "target_date": date.today().strftime("%Y-%m-%d"),
            "status": "active",
            "reason": "Quy định hiện hành (Active)",
        }

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        top_n: int = RERANKER_TOP_N,
        custom_date: Optional[str] = None,
        force_current: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Quy trình Retrieval hoàn chỉnh:
        1. Phân tích Time Intent & Filter
        2. Qdrant Vector Search (Dense Candidates)
        3. PostgreSQL Context Expansion
        4. Reranking (FlashRank / Cross-Encoder)
        """
        # Bước 1: Xác định điều kiện lọc
        time_info = self.parse_time_intent(query)
        is_current = time_info["is_current_only"] if force_current is None else force_current
        status = time_info["status"] if force_current is None else ("active" if force_current else None)

        # Bước 2: Sinh embedding câu hỏi & Dense Search trong Qdrant
        query_vec = self.embedder.get_query_embedding(query)
        qdrant_candidates = self.qdrant.search_candidates(
            query_vector=query_vec,
            top_k=top_k,
            status_filter=status,
            is_current_only=is_current,
        )

        if not qdrant_candidates:
            # Thử nới lỏng bộ lọc nếu không tìm thấy
            qdrant_candidates = self.qdrant.search_candidates(
                query_vector=query_vec,
                top_k=top_k,
                status_filter=None,
                is_current_only=False,
            )

        # Bước 3: Mở rộng ngữ cảnh phân cấp từ PostgreSQL
        expanded_passages = self.expander.expand_candidates(qdrant_candidates)

        # Loại bỏ trùng lặp Điều luật
        unique_passages = []
        seen_art_ids = set()
        for p in expanded_passages:
            art_id = p.get("article_id")
            if art_id not in seen_art_ids:
                seen_art_ids.add(art_id)
                unique_passages.append(p)

        # Bước 4: Xếp hạng lại bằng Reranker
        top_contexts = self.reranker.rerank(query, unique_passages, top_n=top_n)

        return top_contexts
