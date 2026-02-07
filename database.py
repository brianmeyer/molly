import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        log.info(f"Database initialized at {self.db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                jid TEXT PRIMARY KEY,
                name TEXT,
                last_message_time TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_jid TEXT NOT NULL,
                sender TEXT NOT NULL,
                sender_name TEXT,
                content TEXT,
                timestamp TEXT NOT NULL,
                is_from_me INTEGER DEFAULT 0,
                FOREIGN KEY (chat_jid) REFERENCES chats(jid)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_messages_chat
                ON messages(chat_jid);
        """)
        self.conn.commit()

    def store_message(
        self,
        msg_id: str,
        chat_jid: str,
        sender: str,
        sender_name: str,
        content: str,
        timestamp: str,
        is_from_me: bool,
    ):
        # Upsert chat
        self.conn.execute(
            """
            INSERT INTO chats (jid, name, last_message_time)
            VALUES (?, ?, ?)
            ON CONFLICT(jid) DO UPDATE SET
                name = COALESCE(excluded.name, chats.name),
                last_message_time = excluded.last_message_time
            """,
            (chat_jid, sender_name, timestamp),
        )
        # Upsert message
        self.conn.execute(
            """
            INSERT OR REPLACE INTO messages
                (id, chat_jid, sender, sender_name, content, timestamp, is_from_me)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, chat_jid, sender, sender_name, content, timestamp, int(is_from_me)),
        )
        self.conn.commit()

    def get_recent_messages(self, chat_jid: str, limit: int = 20) -> list[dict]:
        cursor = self.conn.execute(
            """
            SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me
            FROM messages
            WHERE chat_jid = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (chat_jid, limit),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_message_count(self, since: str | None = None) -> int:
        if since:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM messages WHERE timestamp > ?", (since,)
            )
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM messages")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()
            log.info("Database closed")
