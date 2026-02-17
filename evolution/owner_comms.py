"""Owner communication service for SelfImprovementEngine.

Handles all interactions with the human owner:
- ``_request_owner_decision()`` — synchronous approval flow (YES/NO/EDIT)
- ``_notify_owner()`` — one-way messages to owner
- ``_log_improvement_event()`` — event logging to vectorstore or fallback DB
- ``_append_self_improvement_event_fallback()`` — direct SQLite fallback writer
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
import db_pool

log = logging.getLogger(__name__)



class OwnerCommsService:
    """Owner communication service — notify, approve, log events.

    Receives an ``EngineContext`` instead of relying on implicit ``self.molly``.
    """

    def __init__(self, ctx):
        from evolution.context import EngineContext
        self.ctx: EngineContext = ctx

    async def request_owner_decision(
        self,
        category: str,
        description: str,
        required_keyword: str = "YES",
        allow_edit: bool = True,
        include_rejection_reason: bool = False,
    ) -> bool | str | tuple[str, str]:
        molly = self.ctx.molly
        if not molly or not getattr(molly, "approvals", None) or not getattr(molly, "wa", None):
            return False
        owner_jid = molly._get_owner_dm_jid()
        if not owner_jid:
            return False
        return await molly.approvals.request_custom_approval(
            category=category,
            description=description,
            chat_jid=owner_jid,
            molly=molly,
            required_keyword=required_keyword,
            allow_edit=allow_edit,
            return_reasoned_denial=include_rejection_reason,
        )

    async def notify_owner(self, text: str) -> None:
        molly = self.ctx.molly
        if not molly or not getattr(molly, "wa", None):
            return
        owner_jid = molly._get_owner_dm_jid()
        if not owner_jid:
            return
        molly._track_send(molly.wa.send_message(owner_jid, text[:3900]))

    def log_improvement_event(
        self,
        event_type: str,
        category: str,
        title: str,
        payload: str,
        status: str,
    ) -> str | None:
        logged = False
        try:
            from memory.retriever import get_vectorstore

            vs = get_vectorstore()
            target_db = Path(config.MOLLYGRAPH_PATH).expanduser()
            vectorstore_db = Path(vs.db_path).expanduser()
            try:
                same_db = target_db.resolve() == vectorstore_db.resolve()
            except Exception:
                same_db = str(target_db) == str(vectorstore_db)
            if not same_db:
                log.debug(
                    "Vectorstore DB path mismatch for self_improvement_events log; "
                    "using fallback writer (%s != %s)",
                    vectorstore_db,
                    target_db,
                )
            else:
                event_id = vs.log_self_improvement_event(
                    event_type=event_type,
                    category=category,
                    title=title,
                    payload=payload,
                    status=status,
                )
                logged = True
                return event_id
        except Exception:
            log.debug("Failed to log self improvement event", exc_info=True)

        if not logged:
            self.append_self_improvement_event_fallback(
                event_type=event_type,
                category=category,
                title=title,
                payload=payload,
                status=status,
            )
        return None

    def append_self_improvement_event_fallback(
        self,
        *,
        event_type: str,
        category: str,
        title: str,
        payload: str,
        status: str,
    ) -> None:
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS self_improvement_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload TEXT,
                    status TEXT DEFAULT 'proposed',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO self_improvement_events
                (id, event_type, category, title, payload, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    str(event_type)[:64],
                    str(category)[:64],
                    str(title)[:255],
                    str(payload)[:20000],
                    str(status)[:64],
                    now,
                    now,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            log.debug("Fallback self improvement event write failed", exc_info=True)
