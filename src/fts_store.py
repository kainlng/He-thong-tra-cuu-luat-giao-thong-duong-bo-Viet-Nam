import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Any

VECTOR_DB_DIR = Path(__file__).resolve().parent.parent / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
FTS_DB_PATH = VECTOR_DB_DIR / "legal_fts.db"


class SQLiteFTSStore:
    """SQLite FTS5 Full-Text Search Store providing BM25 keyword search & verification."""

    def __init__(self, db_path: Path = FTS_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize SQLite FTS5 virtual table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS legal_fts USING fts5(
                    source,
                    header,
                    article_num,
                    chunk_index UNINDEXED,
                    location UNINDEXED,
                    text,
                    tokenize='unicode61 remove_diacritics 0'
                );
            """)
            conn.commit()

    def reset_db(self):
        """Drop existing FTS content for re-indexing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS legal_fts;")
            conn.commit()
        self._init_db()

    def add_documents(self, docs_metadata: List[Dict[str, Any]]):
        """Insert legal document chunks into FTS5 table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for doc in docs_metadata:
                cursor.execute(
                    """
                    INSERT INTO legal_fts (source, header, article_num, chunk_index, location, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc.get("source", ""),
                        doc.get("header", ""),
                        doc.get("article_num", ""),
                        doc.get("chunk_index", 0),
                        doc.get("location", ""),
                        doc.get("text", ""),
                    ),
                )
            conn.commit()
        print(f"Indexed {len(docs_metadata)} documents into SQLite FTS5 database at {self.db_path}")

    def clean_query(self, query: str) -> str:
        """Sanitize query for FTS5 syntax."""
        cleaned = re.sub(r'[^\w\s]', ' ', query, flags=re.UNICODE)
        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
        if not words:
            return query
        # Use OR query for broad BM25 term coverage
        return " OR ".join(words)

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search legal chunks using FTS5 native BM25 rank scoring."""
        fts_query = self.clean_query(query)
        results = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # SQLite fts5 bm25() returns negative score where lower (more negative) is better match
                cursor.execute(
                    """
                    SELECT source, header, article_num, chunk_index, location, text,
                           bm25(legal_fts, 2.0, 5.0, 1.0) AS bm25_score
                    FROM legal_fts
                    WHERE legal_fts MATCH ?
                    ORDER BY bm25_score ASC
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                )
                rows = cursor.fetchall()
            except sqlite3.OperationalError as exc:
                print(f"FTS5 Search Operational Warning: {exc}. Falling back to LIKE query.")
                cursor.execute(
                    """
                    SELECT source, header, article_num, chunk_index, location, text, 0.0 AS bm25_score
                    FROM legal_fts
                    WHERE text LIKE ? OR header LIKE ?
                    LIMIT ?
                    """,
                    (f"%{query}%", f"%{query}%", top_k),
                )
                rows = cursor.fetchall()

            for row in rows:
                # Transform negative SQLite BM25 score into positive rank score for RRF
                raw_bm25 = float(row["bm25_score"])
                pos_score = abs(raw_bm25) if raw_bm25 < 0 else (1.0 / (1.0 + raw_bm25))

                results.append({
                    "score": pos_score,
                    "metadata": {
                        "source": row["source"],
                        "header": row["header"],
                        "article_num": row["article_num"],
                        "chunk_index": row["chunk_index"],
                        "location": row["location"],
                        "text": row["text"],
                    },
                })

        return results

    def verify_citation(self, article_num: str) -> bool:
        """Verify if a cited article number (e.g. '25' or '12') exists in the FTS store."""
        if not article_num:
            return True

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM legal_fts WHERE article_num = ? OR header LIKE ?",
                (article_num, f"%Điều {article_num}%"),
            )
            count = cursor.fetchone()[0]
            return count > 0
