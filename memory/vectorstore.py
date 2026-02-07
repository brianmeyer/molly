import json
import logging
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

log = logging.getLogger(__name__)

EMBEDDING_DIM = 768


def _serialize_float32(vec: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class VectorStore:
    """sqlite-vec backed vector store for conversation chunks + operational logs."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._create_tables()
        log.info("VectorStore initialized at %s", self.db_path)

    def _create_tables(self):
        self.conn.executescript(f"""
            -- Personal memory: conversation chunks with embeddings
            CREATE TABLE IF NOT EXISTS conversation_chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT DEFAULT 'whatsapp',
                chat_jid TEXT,
                topic_tags TEXT DEFAULT '',
                entity_refs TEXT DEFAULT ''
            );

            -- Operational memory: tool call logs
            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                tool_name TEXT,
                parameters TEXT,
                success INTEGER,
                latency_ms INTEGER,
                error_message TEXT,
                user_feedback TEXT,
                created_at TEXT
            );

            -- Operational memory: skill execution logs
            CREATE TABLE IF NOT EXISTS skill_executions (
                id TEXT PRIMARY KEY,
                skill_name TEXT,
                trigger TEXT,
                outcome TEXT,
                user_approval TEXT,
                edits_made TEXT,
                created_at TEXT
            );

            -- Operational memory: corrections
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                context TEXT,
                molly_output TEXT,
                user_correction TEXT,
                pattern TEXT,
                created_at TEXT
            );

            -- Operational memory: preference signals
            CREATE TABLE IF NOT EXISTS preference_signals (
                id TEXT PRIMARY KEY,
                signal_type TEXT,
                context TEXT,
                created_at TEXT
            );
        """)

        # Create virtual vec table (separate statement — can't be in executescript)
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec
            USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)
        self.conn.commit()

    def store_chunk(
        self,
        content: str,
        embedding: list[float],
        source: str = "whatsapp",
        chat_jid: str = "",
        topic_tags: str = "",
        entity_refs: str = "",
    ) -> str:
        """Store a conversation chunk with its embedding. Returns the chunk ID."""
        chunk_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            INSERT INTO conversation_chunks
                (id, content, created_at, source, chat_jid, topic_tags, entity_refs)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, content, now, source, chat_jid, topic_tags, entity_refs),
        )
        self.conn.execute(
            "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
            (chunk_id, _serialize_float32(embedding)),
        )
        self.conn.commit()
        return chunk_id

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """ANN search for the most similar conversation chunks."""
        cursor = self.conn.execute(
            """
            SELECT
                cv.id,
                cv.distance,
                cc.content,
                cc.created_at,
                cc.source,
                cc.chat_jid,
                cc.topic_tags,
                cc.entity_refs
            FROM chunks_vec cv
            JOIN conversation_chunks cc ON cc.id = cv.id
            WHERE cv.embedding MATCH ?
                AND k = ?
            ORDER BY cv.distance
            """,
            (_serialize_float32(query_embedding), top_k),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "distance": r["distance"],
                "content": r["content"],
                "created_at": r["created_at"],
                "source": r["source"],
                "chat_jid": r["chat_jid"],
                "topic_tags": r["topic_tags"],
                "entity_refs": r["entity_refs"],
            }
            for r in rows
        ]

    def log_tool_call(
        self,
        tool_name: str,
        parameters: str = "",
        success: bool = True,
        latency_ms: int = 0,
        error_message: str = "",
    ):
        """Log a tool call to operational memory."""
        call_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO tool_calls
               (id, tool_name, parameters, success, latency_ms, error_message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (call_id, tool_name, parameters[:500], int(success), latency_ms, error_message, now),
        )
        self.conn.commit()

    def chunk_count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM conversation_chunks")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
            log.info("VectorStore closed")
