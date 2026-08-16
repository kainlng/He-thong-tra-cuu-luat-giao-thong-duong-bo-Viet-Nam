import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LegalReranker:
    """Xếp hạng lại (Reranking) danh sách các đoạn ngữ cảnh pháp lý dựa trên độ tương quan với câu hỏi."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker = None
        self._cross_encoder = None
        self._engine_type = "heuristic"

        # 1. Thử tải FlashRank (Reranker siêu nhẹ, cực nhanh, tối ưu CPU)
        try:
            from flashrank import Ranker
            print(f"[Reranker] Đang khởi tạo FlashRank model: {model_name}...")
            self._ranker = Ranker(model_name=model_name)
            self._engine_type = "flashrank"
            print("[Reranker] FlashRank engine đã sẵn sàng.")
            return
        except Exception as exc:
            logger.warning(f"FlashRank init note: {exc}. Thử chuyển sang CrossEncoder...")

        # 2. Thử tải CrossEncoder từ sentence-transformers
        try:
            from sentence_transformers import CrossEncoder
            print(f"[Reranker] Đang khởi tạo SentenceTransformer CrossEncoder: BAAI/bge-reranker-base...")
            self._cross_encoder = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
            self._engine_type = "cross_encoder"
            print("[Reranker] CrossEncoder engine đã sẵn sàng.")
            return
        except Exception as exc:
            logger.warning(f"CrossEncoder init note: {exc}. Sử dụng bộ xếp hạng trọng số ngữ nghĩa.")

    def rerank(self, query: str, passages: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
        """Chấm điểm lại các đoạn văn bản (Passages) và trả về Top N đoạn chính xác nhất."""
        if not passages:
            return []

        if len(passages) <= top_n:
            return passages

        if self._engine_type == "flashrank" and self._ranker:
            try:
                from flashrank import RerankRequest
                flash_passages = [
                    {"id": idx, "text": p["passage_text"], "meta": p}
                    for idx, p in enumerate(passages)
                ]
                req = RerankRequest(query=query, passages=flash_passages)
                ranked_items = self._ranker.rerank(req)
                
                results = []
                for item in ranked_items[:top_n]:
                    orig_p = item["meta"]
                    orig_p["rerank_score"] = float(item["score"])
                    results.append(orig_p)
                return results
            except Exception as exc:
                print(f"[Reranker Warning] FlashRank rerank error: {exc}. Fallback to initial scores.")

        elif self._engine_type == "cross_encoder" and self._cross_encoder:
            try:
                pairs = [[query, p["passage_text"]] for p in passages]
                scores = self._cross_encoder.predict(pairs)
                for idx, score in enumerate(scores):
                    passages[idx]["rerank_score"] = float(score)
                sorted_passages = sorted(passages, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                return sorted_passages[:top_n]
            except Exception as exc:
                print(f"[Reranker Warning] CrossEncoder error: {exc}.")

        # Fallback sorting by initial dense score
        sorted_passages = sorted(passages, key=lambda x: x.get("initial_score", 0.0), reverse=True)
        return sorted_passages[:top_n]
