"""
SQLite-based memory store with embedding similarity search.
Stores problem-solution pairs and feedback for pattern reuse.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from rag.embedder import embed


class MemoryStore:
    """
    Memory store for problem-solution pairs and feedback.
    Supports similarity search via embeddings and feedback tracking.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str((Path(__file__).parent / "memory.db").resolve())
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    topic TEXT,
                    solution TEXT,
                    verdict TEXT,
                    feedback INTEGER,
                    comment TEXT,
                    embedding TEXT,
                    timestamp TEXT NOT NULL
                )
            """
            )

    def add(self, query: str, topic: str, solution: str, verdict: str) -> int:
        """Store a new problem-solution entry. Returns row id."""
        emb = embed(query)[0]
        emb_json = json.dumps(emb)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO memory (query, topic, solution, verdict, embedding, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (query, topic, solution, verdict, emb_json, datetime.now().isoformat()),
            )
            row_id = cur.lastrowid
            conn.commit()
            print(
                f"✅ Added entry to memory: id={row_id}, topic={topic}, query_len={len(query)}, db_path={self.db_path}"
            )
        return row_id

    def update_feedback(self, ok: bool, comment: str = None, entry_id: int = None):
        """Update an entry with user feedback. entry_id: row id or None for latest."""
        feedback_value = 1 if ok else 0
        with sqlite3.connect(self.db_path) as conn:
            if entry_id is not None:
                cursor = conn.execute(
                    "UPDATE memory SET feedback = ?, comment = ? WHERE id = ?",
                    (feedback_value, comment, entry_id),
                )
                rows_affected = cursor.rowcount
                print(
                    f"🔄 Updated entry {entry_id}: feedback={feedback_value}, comment_len={len(comment or '')}, rows_affected={rows_affected}"
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE memory SET feedback = ?, comment = ?
                    WHERE id = (SELECT MAX(id) FROM memory)
                    """,
                    (feedback_value, comment),
                )
                rows_affected = cursor.rowcount
                print(
                    f"🔄 Updated latest entry: feedback={feedback_value}, rows_affected={rows_affected}"
                )
            conn.commit()
            print(f"✅ Database committed - feedback saved to {self.db_path}")

    def fetch_all(self) -> list:
        """Fetch all memory entries."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id, query, solution, verdict, feedback, comment, embedding
                FROM memory ORDER BY timestamp DESC
                """
            )
            return cur.fetchall()

    def similar(self, query: str, top_k: int = 2, thresh: float = 0.75) -> list:
        """Return list of (similarity_score, past_query, past_solution)."""
        new_emb = np.array(embed(query)[0])
        rows = self.fetch_all()
        similarities = []
        for row in rows:
            _, q, sol, _, _, _, emb_json = row
            if not emb_json:
                continue
            emb = np.array(json.loads(emb_json))
            norm_new = np.linalg.norm(new_emb)
            norm_emb = np.linalg.norm(emb)
            if norm_new > 0 and norm_emb > 0:
                sim = float(new_emb @ emb / (norm_new * norm_emb))
                similarities.append((sim, q, sol))
        similarities.sort(reverse=True, key=lambda x: x[0])
        filtered = [item for item in similarities if item[0] >= thresh]
        return filtered[:top_k]

    def get_stats(self) -> dict:
        """Return memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM memory")
            total = int(cur.fetchone()[0] or 0)
            cur = conn.execute("SELECT COUNT(*) FROM memory WHERE feedback = 1")
            positive = int(cur.fetchone()[0] or 0)
            cur = conn.execute(
                "SELECT COUNT(*) FROM memory WHERE feedback = 0 AND feedback IS NOT NULL"
            )
            negative = int(cur.fetchone()[0] or 0)
            cur = conn.execute("SELECT COUNT(*) FROM memory WHERE feedback IS NULL")
            no_feedback = int(cur.fetchone()[0] or 0)
        return {
            "total": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "no_feedback": no_feedback,
        }
