import logging
import sqlite3
from pathlib import Path

import db_pool
from utils import normalize_timestamp

log = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = db_pool.sqlite_connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        migrated = self._normalize_legacy_timestamps()
        if migrated:
            log.info("Normalized %d legacy timestamps to ISO-8601", migrated)
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
        normalized_timestamp = normalize_timestamp(timestamp)
        # Upsert chat
        self.conn.execute(
            """
            INSERT INTO chats (jid, name, last_message_time)
            VALUES (?, ?, ?)
            ON CONFLICT(jid) DO UPDATE SET
                name = COALESCE(excluded.name, chats.name),
                last_message_time = MAX(COALESCE(chats.last_message_time, ''), excluded.last_message_time)
            """,
            (chat_jid, sender_name, normalized_timestamp),
        )
        # Upsert message
        self.conn.execute(
            """
            INSERT OR REPLACE INTO messages
                (id, chat_jid, sender, sender_name, content, timestamp, is_from_me)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg_id,
                chat_jid,
                sender,
                sender_name,
                content,
                normalized_timestamp,
                int(is_from_me),
            ),
        )
        self.conn.commit()

    def _normalize_legacy_timestamps(self) -> int:
        """Backfill pre-existing message/chat timestamps into ISO-8601 format."""
        updates = 0

        message_rows = self.conn.execute("SELECT id, timestamp FROM messages").fetchall()
        message_updates = []
        for row in message_rows:
            old_ts = str(row["timestamp"] or "")
            new_ts = normalize_timestamp(old_ts)
            if new_ts != old_ts:
                message_updates.append((new_ts, row["id"]))
        if message_updates:
            self.conn.executemany(
                "UPDATE messages SET timestamp = ? WHERE id = ?",
                message_updates,
            )
            updates += len(message_updates)

        chat_rows = self.conn.execute(
            "SELECT jid, last_message_time FROM chats WHERE last_message_time IS NOT NULL"
        ).fetchall()
        chat_updates = []
        for row in chat_rows:
            old_ts = str(row["last_message_time"] or "")
            new_ts = normalize_timestamp(old_ts)
            if new_ts != old_ts:
                chat_updates.append((new_ts, row["jid"]))
        if chat_updates:
            self.conn.executemany(
                "UPDATE chats SET last_message_time = ? WHERE jid = ?",
                chat_updates,
            )
            updates += len(chat_updates)

        if updates:
            self.conn.commit()
        return updates

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
