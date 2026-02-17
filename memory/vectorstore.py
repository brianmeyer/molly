import json
import logging
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import threading

import sqlite_vec

import db_pool
from utils import track_latency

log = logging.getLogger(__name__)

EMBEDDING_DIM = 768


def _serialize_float32(vec: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _ensure_columns(
    conn: sqlite3.Connection,
    table_name: str,
    required_columns: dict[str, str],
):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {str(row[1]) for row in rows}
    for column, definition in required_columns.items():
        if column in existing:
            continue
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {definition}")


def ensure_issue_registry_tables(conn: sqlite3.Connection):
    """Initialize maintenance issue tables used by monitoring/reporting."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS maintenance_issues (
            fingerprint TEXT PRIMARY KEY,
            check_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            last_detail TEXT,
            source TEXT
        );

        CREATE TABLE IF NOT EXISTS maintenance_issue_events (
            issue_fingerprint TEXT NOT NULL,
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT
        );
        """
    )
    _ensure_columns(
        conn,
        "maintenance_issues",
        {
            "check_id": "TEXT NOT NULL DEFAULT ''",
            "severity": "TEXT NOT NULL DEFAULT 'green'",
            "status": "TEXT NOT NULL DEFAULT 'resolved'",
            "first_seen": "TEXT NOT NULL DEFAULT ''",
            "last_seen": "TEXT NOT NULL DEFAULT ''",
            "consecutive_failures": "INTEGER NOT NULL DEFAULT 0",
            "last_detail": "TEXT",
            "source": "TEXT",
        },
    )
    _ensure_columns(
        conn,
        "maintenance_issue_events",
        {
            "issue_fingerprint": "TEXT NOT NULL DEFAULT ''",
            "event_type": "TEXT NOT NULL DEFAULT 'observed'",
            "created_at": "TEXT NOT NULL DEFAULT ''",
            "payload": "TEXT",
        },
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_issues_status "
        "ON maintenance_issues(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_issues_last_seen "
        "ON maintenance_issues(last_seen)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_issue_events_fingerprint "
        "ON maintenance_issue_events(issue_fingerprint)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_issue_events_created_at "
        "ON maintenance_issue_events(created_at)"
    )
    conn.commit()


class VectorStore:
    """sqlite-vec backed vector store for conversation chunks + operational logs."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()

    def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = db_pool.sqlite_connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent readers + single writer without blocking
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
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

            -- Operational memory: detected skill gaps
            CREATE TABLE IF NOT EXISTS skill_gaps (
                id INTEGER PRIMARY KEY,
                user_message TEXT,
                tools_used TEXT,
                session_id TEXT,
                created_at TEXT,
                addressed INTEGER DEFAULT 0
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
                source TEXT,
                surfaced_summary TEXT,
                sender_pattern TEXT,
                owner_feedback TEXT,
                context TEXT,
                timestamp TEXT,
                created_at TEXT
            );

            -- Phase 7: self-improvement proposals + deployments + assessments
            CREATE TABLE IF NOT EXISTS self_improvement_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                payload TEXT,
                status TEXT DEFAULT 'proposed',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Sender tier overrides for triage pre-filtering
            CREATE TABLE IF NOT EXISTS sender_tiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_pattern TEXT NOT NULL UNIQUE,
                tier TEXT NOT NULL DEFAULT 'normal',
                source TEXT DEFAULT 'manual',
                updated_at TEXT NOT NULL
            );
        """)
        self._ensure_preference_signal_columns()
        self._ensure_sender_tiers_table()
        ensure_issue_registry_tables(self.conn)

        # Create virtual vec table (separate statement — can't be in executescript)
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec
            USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)

        # Phase 5B: FTS5 full-text search index for BM25 keyword matching.
        # content= uses external content (conversation_chunks) to avoid
        # duplicate storage; content_rowid maps to the TEXT primary key via
        # a hidden rowid we maintain manually.
        self._ensure_fts5()
        self.conn.commit()

    def _ensure_preference_signal_columns(self):
        """Backfill new preference_signals columns for existing databases."""
        cursor = self.conn.execute("PRAGMA table_info(preference_signals)")
        existing = {row["name"] for row in cursor.fetchall()}
        required = {
            "source": "TEXT",
            "surfaced_summary": "TEXT",
            "sender_pattern": "TEXT",
            "owner_feedback": "TEXT",
            "timestamp": "TEXT",
        }
        for column, col_type in required.items():
            if column not in existing:
                self.conn.execute(
                    f"ALTER TABLE preference_signals ADD COLUMN {column} {col_type}"
                )

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

        with self._write_lock:
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
            # Phase 5B: keep FTS5 index in sync
            self.conn.execute(
                "INSERT INTO chunks_fts (chunk_id, content, source) VALUES (?, ?, ?)",
                (chunk_id, content, source or "unknown"),
            )
            self.conn.commit()
        return chunk_id

    def store_chunks_batch(
        self,
        chunks: list[dict],
    ) -> list[str]:
        """Store multiple conversation chunks in a single transaction.

        Each dict must have: content, embedding, source, chat_jid.
        Returns list of chunk IDs.
        """
        if not chunks:
            return []

        chunk_ids = []
        now = datetime.now(timezone.utc).isoformat()

        with self._write_lock:
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                self.conn.execute(
                    """
                    INSERT INTO conversation_chunks
                        (id, content, created_at, source, chat_jid, topic_tags, entity_refs)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, chunk["content"], now, chunk["source"],
                     chunk["chat_jid"], "", ""),
                )
                self.conn.execute(
                    "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                    (chunk_id, _serialize_float32(chunk["embedding"])),
                )
                # Phase 5B: keep FTS5 index in sync
                self.conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, content, source) VALUES (?, ?, ?)",
                    (chunk_id, chunk["content"], chunk.get("source", "unknown")),
                )

            self.conn.commit()
        log.debug("Batch stored %d chunks in single transaction", len(chunk_ids))
        return chunk_ids

    @track_latency("vectorstore")
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
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO tool_calls
                   (id, tool_name, parameters, success, latency_ms, error_message, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (call_id, tool_name, parameters[:500], int(success), latency_ms, error_message, now),
            )
            self.conn.commit()

    def log_preference_signal(
        self,
        source: str,
        surfaced_summary: str,
        sender_pattern: str,
        owner_feedback: str,
        signal_type: str = "dismissive_feedback",
    ):
        """Log dismissive owner feedback tied to a surfaced proactive item."""
        signal_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        context = json.dumps(
            {
                "source": source,
                "surfaced_summary": surfaced_summary,
                "sender_pattern": sender_pattern,
                "owner_feedback": owner_feedback,
            },
            ensure_ascii=True,
        )
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO preference_signals
                   (id, signal_type, source, surfaced_summary, sender_pattern, owner_feedback, context, timestamp, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal_id,
                    signal_type,
                    source[:32],
                    surfaced_summary[:1000],
                    sender_pattern[:500],
                    owner_feedback[:500],
                    context[:2000],
                    now,
                    now,
                ),
            )
            self.conn.commit()

    def _ensure_sender_tiers_table(self):
        """Migration safety: ensure sender_tiers exists for older databases."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sender_tiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_pattern TEXT NOT NULL UNIQUE,
                tier TEXT NOT NULL DEFAULT 'normal',
                source TEXT DEFAULT 'manual',
                updated_at TEXT NOT NULL
            )
        """)

    def _ensure_fts5(self):
        """Create FTS5 virtual table for BM25 keyword search (Phase 5B).

        Uses a standalone FTS5 table (not external-content) so we can
        populate and query it independently of conversation_chunks.  The
        rowid is the chunk UUID stored as TEXT in a mapping column.
        """
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                content,
                source UNINDEXED,
                tokenize = 'porter unicode61'
            )
        """)
        # Backfill: if chunks exist but FTS is empty, rebuild the index.
        try:
            fts_count = self.conn.execute(
                "SELECT COUNT(*) FROM chunks_fts"
            ).fetchone()[0]
            chunk_count = self.conn.execute(
                "SELECT COUNT(*) FROM conversation_chunks"
            ).fetchone()[0]
            if fts_count == 0 and chunk_count > 0:
                log.info(
                    "Backfilling FTS5 index: %d chunks to index", chunk_count
                )
                self._rebuild_fts5()
        except Exception:
            log.debug("FTS5 backfill check failed", exc_info=True)

    def _rebuild_fts5(self):
        """Rebuild the FTS5 index from conversation_chunks."""
        with self._write_lock:
            self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute("""
                INSERT INTO chunks_fts (chunk_id, content, source)
                SELECT id, content, COALESCE(source, 'unknown')
                FROM conversation_chunks
            """)
            self.conn.commit()
        log.info("FTS5 index rebuilt")

    def search_fts(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 keyword search using FTS5.

        Returns results ranked by BM25 score (lower = more relevant,
        following SQLite FTS5 convention where rank is negative BM25).
        """
        if not query or not query.strip():
            return []

        # Escape FTS5 special characters for safety
        # FTS5 special chars: " * ^ : ( ) AND OR NOT NEAR
        safe_query = query.replace('"', ' ')
        for ch in ('*', '^', ':', '(', ')'):
            safe_query = safe_query.replace(ch, ' ')

        # Tokenize into individual terms and join with OR for broad matching.
        # Previous approach wrapped in quotes for exact-phrase search which
        # returned 0 results when the phrase didn't appear verbatim.
        tokens = [t.strip() for t in safe_query.split() if t.strip()]
        if not tokens:
            return []
        fts_expr = " OR ".join(f'"{t}"' for t in tokens)

        try:
            cursor = self.conn.execute(
                """
                SELECT
                    cf.chunk_id,
                    cf.content,
                    cf.source,
                    cf.rank AS bm25_rank,
                    cc.created_at,
                    cc.chat_jid,
                    cc.topic_tags,
                    cc.entity_refs
                FROM chunks_fts cf
                JOIN conversation_chunks cc ON cc.id = cf.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY cf.rank
                LIMIT ?
                """,
                (fts_expr, top_k),
            )
            rows = cursor.fetchall()
        except Exception:
            # FTS match can fail on malformed queries; fall back to LIKE
            log.debug("FTS5 MATCH failed, trying LIKE fallback", exc_info=True)
            try:
                cursor = self.conn.execute(
                    """
                    SELECT
                        id AS chunk_id,
                        content,
                        source,
                        0.0 AS bm25_rank,
                        created_at,
                        chat_jid,
                        topic_tags,
                        entity_refs
                    FROM conversation_chunks
                    WHERE content LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", top_k),
                )
                rows = cursor.fetchall()
            except Exception:
                log.error("FTS5 LIKE fallback also failed", exc_info=True)
                return []

        return [
            {
                "id": r["chunk_id"],
                "content": r["content"],
                "source": r["source"],
                "bm25_rank": float(r["bm25_rank"]),
                "created_at": r["created_at"],
                "chat_jid": r["chat_jid"],
                "topic_tags": r["topic_tags"],
                "entity_refs": r["entity_refs"],
            }
            for r in rows
        ]

    @track_latency("vectorstore")
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> list[dict]:
        """Three-layer retrieval: vector similarity + BM25 keyword matching.

        Scores are normalized to [0, 1] and combined:
            final_score = vector_weight * vector_sim + bm25_weight * bm25_sim

        Weights must sum to ~1.0 (±0.05 tolerance). Both must be non-negative.

        Returns up to ``top_k`` results sorted by final_score descending.
        """
        # Validate weights
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError(f"Weights must be non-negative: vector={vector_weight}, bm25={bm25_weight}")
        weight_sum = vector_weight + bm25_weight
        if abs(weight_sum - 1.0) > 0.05:
            raise ValueError(
                f"Weights must sum to ~1.0 (got {weight_sum:.2f}): "
                f"vector={vector_weight}, bm25={bm25_weight}"
            )
        # Layer 1: Vector search (fetch extra candidates for fusion)
        vec_k = min(top_k * 3, 50)
        vec_results = self.search(query_embedding, top_k=vec_k)

        # Layer 2: BM25 keyword search
        fts_results = self.search_fts(query_text, top_k=vec_k)

        # Build score maps keyed by chunk ID
        # Vector: distance → similarity (1 - distance), clamp to [0, 1]
        vec_scores: dict[str, float] = {}
        vec_data: dict[str, dict] = {}
        for r in vec_results:
            cid = r["id"]
            sim = max(0.0, min(1.0, 1.0 - float(r.get("distance", 1.0))))
            vec_scores[cid] = sim
            vec_data[cid] = r

        # BM25: SQLite FTS5 rank is negative (more negative = more relevant).
        # Normalize to [0, 1] using min-max within result set.
        fts_scores: dict[str, float] = {}
        fts_data: dict[str, dict] = {}
        if fts_results:
            raw_ranks = [float(r.get("bm25_rank", 0.0)) for r in fts_results]
            min_rank = min(raw_ranks)  # most relevant (most negative)
            max_rank = max(raw_ranks)  # least relevant

            if min_rank == max_rank:
                # Single result or all equal ranks: assign score 1.0 to all
                for r in fts_results:
                    cid = r["id"]
                    fts_scores[cid] = 1.0
                    fts_data[cid] = r
            else:
                rank_range = max_rank - min_rank
                for r in fts_results:
                    cid = r["id"]
                    raw = float(r.get("bm25_rank", 0.0))
                    # Invert: most negative → highest score
                    norm = (max_rank - raw) / rank_range
                    fts_scores[cid] = max(0.0, min(1.0, norm))
                    fts_data[cid] = r

        # Merge: union of all candidate chunk IDs
        all_ids = set(vec_scores.keys()) | set(fts_scores.keys())
        scored: list[tuple[str, float]] = []
        for cid in all_ids:
            v = vec_scores.get(cid, 0.0)
            b = fts_scores.get(cid, 0.0)
            final = vector_weight * v + bm25_weight * b
            scored.append((cid, final))

        # Sort by final score descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = scored[:top_k]

        # Build output with combined metadata
        results: list[dict] = []
        for cid, final_score in top_ids:
            # Prefer vector data (has distance), fall back to FTS data
            data = vec_data.get(cid) or fts_data.get(cid, {})
            results.append({
                "id": cid,
                "content": data.get("content", ""),
                "created_at": data.get("created_at", ""),
                "source": data.get("source", ""),
                "chat_jid": data.get("chat_jid", ""),
                "topic_tags": data.get("topic_tags", ""),
                "entity_refs": data.get("entity_refs", ""),
                "distance": float(data.get("distance", 1.0 - vec_scores.get(cid, 0.0))),
                "vector_score": vec_scores.get(cid, 0.0),
                "bm25_score": fts_scores.get(cid, 0.0),
                "hybrid_score": final_score,
            })

        return results

    def upsert_sender_tier(
        self, sender_pattern: str, tier: str, source: str = "manual",
    ):
        """Insert or update a sender tier override."""
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO sender_tiers (sender_pattern, tier, source, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(sender_pattern)
                   DO UPDATE SET tier = excluded.tier,
                                 source = excluded.source,
                                 updated_at = excluded.updated_at""",
                (sender_pattern.strip().lower(), tier.strip().lower(), source[:32], now),
            )
            self.conn.commit()

    def get_sender_tier(self, sender_pattern: str) -> str | None:
        """Look up the tier for a single sender pattern. Returns None if not set."""
        row = self.conn.execute(
            "SELECT tier FROM sender_tiers WHERE sender_pattern = ?",
            (sender_pattern.strip().lower(),),
        ).fetchone()
        return row["tier"] if row else None

    def get_sender_tiers(self) -> list[dict]:
        """Return all non-normal sender tier overrides."""
        rows = self.conn.execute(
            "SELECT sender_pattern, tier, source, updated_at "
            "FROM sender_tiers WHERE tier != 'normal' ORDER BY updated_at DESC"
        ).fetchall()
        return [
            {
                "sender_pattern": r["sender_pattern"],
                "tier": r["tier"],
                "source": r["source"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def get_triage_context_signals(self, limit: int = 20) -> dict:
        """Aggregate sender tiers + frequently dismissed senders for triage prompt."""
        tiers = self.get_sender_tiers()

        # Find senders dismissed 2+ times
        dismissed_rows = self.conn.execute(
            """SELECT sender_pattern, COUNT(*) as cnt
               FROM preference_signals
               WHERE signal_type = 'dismissive_feedback'
                 AND sender_pattern IS NOT NULL
                 AND sender_pattern != ''
               GROUP BY sender_pattern
               HAVING cnt >= 2
               ORDER BY cnt DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        dismissed = [
            {"sender_pattern": r["sender_pattern"], "dismissals": r["cnt"]}
            for r in dismissed_rows
        ]

        return {"sender_tiers": tiers, "dismissed_senders": dismissed}

    def log_skill_gap(
        self,
        user_message: str,
        tools_used: list[str],
        session_id: str = "",
        addressed: bool = False,
    ) -> int:
        """Log a missing-skill candidate based on turn-level workflow activity."""
        now = datetime.now(timezone.utc).isoformat()
        tools_payload = json.dumps(tools_used, ensure_ascii=True)
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO skill_gaps
                   (user_message, tools_used, session_id, created_at, addressed)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    user_message[:4000],
                    tools_payload[:4000],
                    session_id[:200],
                    now,
                    int(bool(addressed)),
                ),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def chunk_count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM conversation_chunks")
        return cursor.fetchone()[0]

    def log_skill_execution(
        self,
        skill_name: str,
        trigger: str,
        outcome: str,
        user_approval: str = "",
        edits_made: str = "",
    ):
        execution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO skill_executions
                   (id, skill_name, trigger, outcome, user_approval, edits_made, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    execution_id,
                    skill_name[:200],
                    trigger[:500],
                    outcome[:1000],
                    user_approval[:200],
                    edits_made[:1000],
                    now,
                ),
            )
            self.conn.commit()

    def log_correction(
        self,
        context: str,
        molly_output: str,
        user_correction: str,
        pattern: str = "",
    ):
        correction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO corrections
                   (id, context, molly_output, user_correction, pattern, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    correction_id,
                    context[:2000],
                    molly_output[:2000],
                    user_correction[:2000],
                    pattern[:200],
                    now,
                ),
            )
            self.conn.commit()

    def log_self_improvement_event(
        self,
        event_type: str,
        category: str,
        title: str,
        payload: str = "",
        status: str = "proposed",
    ) -> str:
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO self_improvement_events
                   (id, event_type, category, title, payload, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    event_type[:64],
                    category[:64],
                    title[:200],
                    payload[:10000],
                    status[:32],
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return event_id

    def update_self_improvement_event_status(
        self,
        event_id: str,
        status: str,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            self.conn.execute(
                "UPDATE self_improvement_events SET status = ?, updated_at = ? WHERE id = ?",
                (status[:32], now, event_id),
            )
            self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            log.info("VectorStore closed")
