import json
import logging
import time
from datetime import datetime, date

import config

log = logging.getLogger(__name__)

HEARTBEAT_SENTINEL = "HEARTBEAT_OK"
_skill_reload_count = 0

# Morning digest window: first heartbeat between 7:00-7:59
DIGEST_HOUR = 7


def _send_surface(
    molly,
    chat_jid: str,
    text: str,
    source: str,
    surfaced_summary: str = "",
    sender_pattern: str = "",
):
    """Send surfaced info and attach metadata for passive preference logging."""
    if hasattr(molly, "send_surface_message"):
        molly.send_surface_message(
            chat_jid,
            text,
            source=source,
            surfaced_summary=surfaced_summary or text,
            sender_pattern=sender_pattern,
        )
        return

    # Test/legacy fallback: preserve old behavior if helper isn't present.
    if getattr(molly, "wa", None):
        molly._track_send(molly.wa.send_message(chat_jid, text))


def _update_state_key(key: str, value) -> None:
    """Atomically update a single key in state.json.

    Re-reads the file before writing to avoid clobbering concurrent updates
    from the heartbeat or MCP tools.
    """
    config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state_data = (
        json.loads(config.STATE_FILE.read_text())
        if config.STATE_FILE.exists()
        else {}
    )
    state_data[key] = value
    config.STATE_FILE.write_text(json.dumps(state_data, indent=2))


def should_heartbeat(last_heartbeat: datetime | None) -> bool:
    """Check if it's time for a heartbeat based on interval and active hours."""
    now = datetime.now()
    hour = now.hour
    if hour < config.ACTIVE_HOURS[0] or hour >= config.ACTIVE_HOURS[1]:
        return False
    if last_heartbeat is None:
        return True
    elapsed = (now - last_heartbeat).total_seconds()
    return elapsed >= config.HEARTBEAT_INTERVAL


async def run_heartbeat(molly):
    """Run a heartbeat check: HEARTBEAT.md evaluation + skill triggers + iMessage/email monitoring."""
    global _skill_reload_count
    import skills

    try:
        reloaded = skills.check_for_changes()
        if reloaded:
            _skill_reload_count += 1
        log.info(
            "Heartbeat skill hot-reload: status=%s total_reloads=%d",
            skills.get_reload_status(),
            _skill_reload_count,
        )
    except Exception:
        log.error("Heartbeat skill hot-reload check failed", exc_info=True)

    # Check for new iMessages and surface urgent ones
    await _check_imessages(molly)

    # Check for new emails and surface urgent ones
    await _check_email(molly)

    # Always send proactive messages to the owner's DM, not a random group
    chat_jid = molly._get_owner_dm_jid()
    if not chat_jid:
        log.debug("No owner DM JID available for heartbeat")
        return

    # Check for due commitments and send reminders
    await _check_due_commitments(molly, chat_jid)

    # Phase 3D: proactive skills — only run if enabled in HEARTBEAT_SKILLS
    if "daily-digest" in skills.HEARTBEAT_SKILLS:
        await _check_morning_digest(molly, chat_jid)

    if "meeting-prep" in skills.HEARTBEAT_SKILLS:
        await _check_meeting_prep(molly, chat_jid)

    # Standard heartbeat: HEARTBEAT.md evaluation
    from agent import handle_message

    if not config.HEARTBEAT_FILE.exists():
        log.debug("No HEARTBEAT.md found, skipping heartbeat prompt")
        return

    checklist = config.HEARTBEAT_FILE.read_text()
    prompt = (
        "HEARTBEAT CHECK:\n\n"
        f"{checklist}\n\n"
        "Evaluate the checklist above against your current knowledge. "
        "If there is nothing actionable or worth messaging Brian about, "
        "respond with exactly: HEARTBEAT_OK\n"
        "If there is something worth sending, respond with that message only."
    )

    heartbeat_key = f"heartbeat:{chat_jid}"
    session_id = molly.sessions.get(heartbeat_key)

    try:
        response, new_session_id = await handle_message(prompt, chat_jid, session_id, source="heartbeat")

        if new_session_id:
            molly.sessions[heartbeat_key] = new_session_id
            molly.save_sessions()

        if response and HEARTBEAT_SENTINEL not in response:
            log.info("Heartbeat: sending proactive message to %s", chat_jid)
            _send_surface(
                molly,
                chat_jid,
                response,
                source="calendar",
                surfaced_summary=response,
                sender_pattern="heartbeat:checklist",
            )
        else:
            log.info("Heartbeat: nothing to report")
    except Exception:
        log.error("Heartbeat failed", exc_info=True)


# ---------------------------------------------------------------------------
# Commitment delivery: nudge for due/overdue commitments
# ---------------------------------------------------------------------------

# How long after sending a nudge before we send another for the same commitment.
_NUDGE_COOLDOWN_S = 3600  # 1 hour

# How old a dateless commitment must be before we surface it.
_DATELESS_STALE_HOURS = 24


def _in_quiet_hours(now_local) -> bool:
    """Return True if now_local falls within configured quiet hours."""
    from datetime import time as _time

    try:
        sh, sm = config.QUIET_HOURS_START.split(":")
        eh, em = config.QUIET_HOURS_END.split(":")
        start = _time(int(sh), int(sm))
        end = _time(int(eh), int(em))
    except (ValueError, AttributeError):
        return False

    t = now_local.time().replace(second=0, microsecond=0)
    if start <= end:
        return start <= t < end
    # Wraps midnight, e.g. 22:00 -> 07:00
    return t >= start or t < end


async def _check_due_commitments(molly, chat_jid: str):
    """Send WhatsApp nudges for commitments that are due or overdue."""
    engine = getattr(molly, "automations", None)
    if engine is None:
        return
    try:
        from datetime import timezone as _tz
        from zoneinfo import ZoneInfo

        now_utc = datetime.now(_tz.utc)
        now_local = now_utc.astimezone(ZoneInfo(config.TIMEZONE))

        # Respect quiet hours — don't nudge while the user is sleeping.
        if _in_quiet_hours(now_local):
            log.debug("Skipping commitment nudges during quiet hours")
            return

        # Read commitment state through the engine's state lock.
        async with engine._state_lock:
            tracker_state = engine._state.get("automations", {}).get("commitment-tracker", {})
            commitments = tracker_state.get("commitments", [])
            if not isinstance(commitments, list):
                return

        nudge_messages = []
        for record in commitments:
            if not isinstance(record, dict):
                continue
            if str(record.get("status", "open")).lower() != "open":
                continue

            # Skip if recently nudged
            last_nudged = str(record.get("last_nudged_at", "")).strip()
            if last_nudged:
                try:
                    nudged_dt = datetime.fromisoformat(last_nudged.replace("Z", "+00:00"))
                    if nudged_dt.tzinfo is None:
                        nudged_dt = nudged_dt.replace(tzinfo=_tz.utc)
                    if (now_utc - nudged_dt).total_seconds() < _NUDGE_COOLDOWN_S:
                        continue
                except (ValueError, TypeError):
                    pass

            title = str(record.get("title", "(untitled)"))
            due_at_raw = str(record.get("due_at", "")).strip()

            if due_at_raw:
                try:
                    due_dt = datetime.fromisoformat(due_at_raw.replace("Z", "+00:00"))
                    if due_dt.tzinfo is None:
                        due_dt = due_dt.replace(tzinfo=_tz.utc)
                    if due_dt <= now_utc:
                        overdue_mins = int((now_utc - due_dt).total_seconds() / 60)
                        if overdue_mins < 60:
                            nudge_messages.append((record, f"Reminder: {title}"))
                        else:
                            hours = overdue_mins // 60
                            nudge_messages.append((record, f"Overdue ({hours}h): {title}"))
                except (ValueError, TypeError):
                    pass
            else:
                # Dateless commitment: surface if older than _DATELESS_STALE_HOURS
                created_raw = str(record.get("created_at", "")).strip()
                if created_raw:
                    try:
                        created_dt = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=_tz.utc)
                        age_hours = (now_utc - created_dt).total_seconds() / 3600
                        if age_hours >= _DATELESS_STALE_HOURS:
                            nudge_messages.append(
                                (record, f"Needs attention (no due date, {int(age_hours)}h old): {title}")
                            )
                    except (ValueError, TypeError):
                        pass

        if not nudge_messages:
            return

        # Mark nudged BEFORE sending so a crash during send doesn't cause
        # duplicate nudges on the next heartbeat.
        async with engine._state_lock:
            tracker_state = engine._state.get("automations", {}).get("commitment-tracker", {})
            all_commitments = tracker_state.get("commitments", [])
            nudged_ids = {str(r.get("id", "")) for r, _ in nudge_messages}
            changed = False
            for record in all_commitments:
                if not isinstance(record, dict):
                    continue
                if str(record.get("id", "")) in nudged_ids:
                    record["last_nudged_at"] = now_utc.isoformat()
                    record["updated_at"] = now_utc.isoformat()
                    changed = True
            if changed:
                await engine._save_state_locked()

        # Build a single combined nudge message
        lines = ["Commitment reminders:"]
        for _, msg in nudge_messages[:10]:  # Cap at 10 per heartbeat
            lines.append(f"- {msg}")

        _send_surface(
            molly,
            chat_jid,
            "\n".join(lines),
            source="commitment",
            surfaced_summary=f"{len(nudge_messages)} commitment(s) due",
            sender_pattern="heartbeat:commitments",
        )

        log.info("Sent %d commitment nudge(s) to %s", len(nudge_messages), chat_jid)

    except Exception:
        log.error("Commitment delivery check failed", exc_info=True)


# ---------------------------------------------------------------------------
# Phase 3D: Morning digest
# ---------------------------------------------------------------------------

async def _check_morning_digest(molly, chat_jid: str):
    """Send the daily digest if it's the morning window and we haven't sent one today."""
    now = datetime.now()
    if now.hour != DIGEST_HOUR:
        return

    # Check if we already sent a digest today
    state_data = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
    last_digest = state_data.get("last_digest_date", "")
    today = date.today().isoformat()

    if last_digest == today:
        return

    log.info("Morning digest: triggering for %s", today)

    # Mark as sent (even if it fails, don't retry every 30 min)
    state_data["last_digest_date"] = today
    config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.STATE_FILE.write_text(json.dumps(state_data, indent=2))

    from agent import handle_message

    prompt = (
        "Good morning. Run the daily digest. "
        "Pull today's calendar, check unread emails from the last 12 hours, "
        "review pending follow-ups, and send me a concise summary of what's on my plate today."
    )

    try:
        response, new_session_id = await handle_message(
            prompt, chat_jid, None,
            approval_manager=molly.approvals,
            molly_instance=molly,
            source="heartbeat",
        )

        if new_session_id:
            molly.sessions[f"digest:{chat_jid}"] = new_session_id
            molly.save_sessions()

        if response:
            _send_surface(
                molly,
                chat_jid,
                response,
                source="calendar",
                surfaced_summary=response,
                sender_pattern="digest:daily",
            )
            log.info("Morning digest sent (%d chars)", len(response))
    except Exception:
        log.error("Morning digest failed", exc_info=True)


# ---------------------------------------------------------------------------
# Phase 3D: Meeting prep
# ---------------------------------------------------------------------------

async def _check_meeting_prep(molly, chat_jid: str):
    """Check for upcoming meetings and send prep 30 min before."""
    try:
        from tools.google_auth import get_credentials

        creds = get_credentials()
        if not creds or not creds.valid:
            return
    except Exception:
        return  # Google auth not set up

    try:
        from googleapiclient.discovery import build
        from tools.google_auth import get_credentials
        from datetime import timedelta, timezone

        creds = get_credentials()
        service = build("calendar", "v3", credentials=creds)

        now = datetime.now(timezone.utc)
        window_start = now + timedelta(minutes=25)
        window_end = now + timedelta(minutes=35)

        events_result = service.events().list(
            calendarId="primary",
            timeMin=window_start.isoformat(),
            timeMax=window_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = events_result.get("items", [])

        for event in events:
            attendees = event.get("attendees", [])
            if not attendees:
                continue  # Skip events with no attendees

            event_id = event["id"]

            # Check if we already prepped this event
            state_data = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
            prepped = state_data.get("prepped_events", [])
            if event_id in prepped:
                continue

            # Mark as prepped
            prepped.append(event_id)
            # Keep only last 50 to avoid unbounded growth
            state_data["prepped_events"] = prepped[-50:]
            config.STATE_FILE.write_text(json.dumps(state_data, indent=2))

            title = event.get("summary", "Untitled meeting")
            start = event.get("start", {}).get("dateTime", "")
            attendee_names = [
                a.get("displayName", a.get("email", "Unknown"))
                for a in attendees
                if not a.get("self", False)
            ]

            log.info("Meeting prep: triggering for '%s' with %d attendees", title, len(attendee_names))

            from agent import handle_message

            prompt = (
                f"Prep me for my upcoming meeting: '{title}' starting at {start}. "
                f"Attendees: {', '.join(attendee_names)}. "
                f"Look up each attendee in contacts and memory, check recent email threads, "
                f"and give me a prep brief."
            )

            try:
                response, new_session_id = await handle_message(
                    prompt, chat_jid, None,
                    approval_manager=molly.approvals,
                    molly_instance=molly,
                    source="heartbeat",
                )
                if response:
                    _send_surface(
                        molly,
                        chat_jid,
                        response,
                        source="calendar",
                        surfaced_summary=f"Meeting prep for {title}",
                        sender_pattern=f"meeting:{title}",
                    )
                    log.info("Meeting prep sent for '%s' (%d chars)", title, len(response))
            except Exception:
                log.error("Meeting prep failed for '%s'", title, exc_info=True)

    except Exception:
        log.debug("Meeting prep check failed", exc_info=True)


async def _check_imessages(molly):
    """Check for new iMessages since last heartbeat and surface important ones.

    Runs the triage model on each new message. Urgent ones get forwarded
    to Brian via WhatsApp. All new messages feed into the memory pipeline.
    """
    try:
        from tools.imessage import get_new_messages_since

        # Load high-water mark
        state_data = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
        last_check = float(state_data.get("imessage_heartbeat_hw", 0))

        if last_check == 0:
            # First run — default to 7 days ago (not epoch 0 which scans entire history)
            last_check = time.time() - (7 * 86400)
            state_data["imessage_heartbeat_hw"] = last_check
            config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.STATE_FILE.write_text(json.dumps(state_data, indent=2))
            log.info("iMessage monitoring: initialized high-water mark to 7 days ago")

        messages = get_new_messages_since(last_check)
        if not messages:
            return

        # Safety cap: don't process more than 50 messages per heartbeat
        # to avoid blocking the main loop for ages
        if len(messages) > 50:
            log.warning(
                "iMessage heartbeat: %d messages found, capping at 50 (check timestamp conversion)",
                len(messages),
            )
            messages = messages[-50:]  # most recent 50

        log.info("iMessage heartbeat: %d new messages since last check", len(messages))

        # Triage each message and surface urgent ones
        owner_jid = molly._get_owner_dm_jid()
        if not owner_jid or not molly.wa:
            return

        from memory.triage import triage_message
        from memory.processor import embed_and_store, extract_to_graph

        for msg in messages:
            if msg["is_from_me"] or not msg["text"]:
                continue

            # Run triage
            result = await triage_message(
                msg["text"],
                sender_name=msg["sender"],
                group_name="iMessage",
            )

            if result and result.classification == "urgent":
                preview = msg["text"][:200]
                notify = (
                    f"iMessage from {msg['sender']}\n"
                    f"{preview}\n\n"
                    f"Reason: {result.reason}"
                )
                _send_surface(
                    molly,
                    owner_jid,
                    notify,
                    source="imessage",
                    surfaced_summary=f"{msg['sender']}: {preview}",
                    sender_pattern=f"sender:{msg['sender']}",
                )

            # Embed + graph extract for relevant/urgent messages
            if result and result.classification in ("urgent", "relevant"):
                imsg_text = f"iMessage from {msg['sender']}: {msg['text']}"
                await embed_and_store(imsg_text, chat_jid="imessage", source="imessage")
                await extract_to_graph(imsg_text, chat_jid="imessage", source="imessage")

        # Update high-water mark AFTER processing so a crash doesn't lose messages
        state_data["imessage_heartbeat_hw"] = time.time()
        config.STATE_FILE.write_text(json.dumps(state_data, indent=2))

    except Exception:
        log.debug("iMessage heartbeat check failed", exc_info=True)


async def _check_imessage_mentions(molly):
    """Fast poll for @molly mentions in Brian's sent iMessages.

    Runs on IMESSAGE_MENTION_POLL_INTERVAL (default 60s), separate from the
    30-minute heartbeat.  When @molly is detected:
      1. Fetch surrounding thread context (previous 5-8 messages)
      2. Route through handle_message() so Claude can use tools
      3. Send the response to Brian via WhatsApp
    """
    try:
        from tools.imessage import get_mention_messages_since, get_thread_context

        # Separate high-water mark from the heartbeat's imessage_heartbeat_hw
        state_data = (
            json.loads(config.STATE_FILE.read_text())
            if config.STATE_FILE.exists()
            else {}
        )
        last_check = float(state_data.get("imessage_mention_hw", 0))

        if last_check == 0:
            # First run: start from now (don't scan history for old mentions)
            _update_state_key("imessage_mention_hw", time.time())
            log.info("iMessage mention polling: initialized high-water mark to now")
            return

        mentions = get_mention_messages_since(last_check)
        if not mentions:
            _update_state_key("imessage_mention_hw", time.time())
            return

        log.info("iMessage mention poll: %d @molly mention(s) found", len(mentions))

        owner_jid = molly._get_owner_dm_jid()
        if not owner_jid or not molly.wa:
            log.warning(
                "iMessage mention poll: WhatsApp not available, skipping %d mention(s)",
                len(mentions),
            )
            return

        from agent import handle_message

        for mention in mentions:
            chat_id = mention.get("chat_id")
            if chat_id is None:
                log.warning("Could not resolve chat_id for mention message %s", mention.get("id"))
                continue

            # Fetch surrounding thread context
            context_messages = get_thread_context(
                chat_id=chat_id,
                before_message_id=mention["id"],
                count=config.IMESSAGE_MENTION_CONTEXT_COUNT,
            )

            trigger_text = mention.get("text", "")

            # Build context string from surrounding messages (truncate each to 500 chars)
            context_lines = []
            for ctx_msg in context_messages:
                sender = "Me" if ctx_msg.get("is_from_me") else ctx_msg.get("sender", "Unknown")
                text = ctx_msg.get("text", "")[:500]
                context_lines.append(f"{sender}: {text}")

            thread_context = "\n".join(context_lines) if context_lines else "(no prior context)"

            prompt = (
                f"Brian mentioned you (@molly) in an iMessage conversation. "
                f"Here is the thread context (most recent messages):\n\n"
                f"{thread_context}\n\n"
                f"Brian's @molly message: {trigger_text}\n\n"
                f"Based on this context, figure out what Brian needs and take action. "
                f"If the intent is clear (calendar invite, reminder, contact lookup, "
                f"email draft, etc.), go ahead and do it. "
                f"If you need clarification, ask a specific question. "
                f"Respond concisely — this will be sent to Brian via WhatsApp."
            )

            # Use a synthetic iMessage chat ID for session isolation
            imessage_chat_id = f"imessage:chat:{chat_id}"
            session_id = molly.sessions.get(imessage_chat_id)

            try:
                response, new_session_id = await handle_message(
                    prompt,
                    imessage_chat_id,
                    session_id,
                    approval_manager=molly.approvals,
                    molly_instance=molly,
                    source="imessage-mention",
                )

                if new_session_id:
                    molly.sessions[imessage_chat_id] = new_session_id
                    molly.save_sessions()

                if response:
                    clean_text = config.TRIGGER_PATTERN.sub("", trigger_text).strip()
                    _send_surface(
                        molly,
                        owner_jid,
                        response,
                        source="imessage",
                        surfaced_summary=f"@molly response: {clean_text[:100]}",
                        sender_pattern="imessage:mention",
                    )
                    log.info(
                        "iMessage @molly response sent (%d chars) for mention in chat %s: %s",
                        len(response),
                        chat_id,
                        trigger_text[:100],
                    )
            except Exception:
                log.error("Failed to process iMessage @molly mention", exc_info=True)

        # Update high-water mark AFTER processing (re-read to avoid race with heartbeat)
        _update_state_key("imessage_mention_hw", time.time())

    except Exception:
        log.debug("iMessage mention check failed", exc_info=True)


async def _check_email(molly):
    """Check for new emails since last heartbeat and surface important ones.

    Polls Gmail for unread messages. Processes in three phases for throughput:
      1. Fetch all email metadata (sequential Gmail API)
      2. Triage all emails in parallel (asyncio.gather)
      3. Batch embed passing texts + parallel graph extraction

    Target: 7 emails in under 60 seconds.
    """
    try:
        from tools.google_auth import get_gmail_service

        service = get_gmail_service()
        if not service:
            return

        # Rate limit: only poll every EMAIL_POLL_INTERVAL seconds.
        # Keep legacy fallback for existing state files that only have
        # email_heartbeat_hw.
        state_data = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
        last_poll = float(
            state_data.get(
                "email_heartbeat_last_poll",
                state_data.get("email_heartbeat_hw", 0),
            )
        )
        now = time.time()

        if last_poll > 0 and (now - last_poll) < config.EMAIL_POLL_INTERVAL:
            return

        try:
            high_water_ts_ms = int(
                state_data.get(
                    "email_heartbeat_hw_ts_ms",
                    float(state_data.get("email_heartbeat_hw", 0)) * 1000,
                )
            )
        except Exception:
            high_water_ts_ms = 0
        if high_water_ts_ms < 0:
            high_water_ts_ms = 0

        high_water_ids_list = [
            str(value).strip()
            for value in state_data.get("email_heartbeat_hw_ids", [])
            if str(value).strip()
        ]
        legacy_high_water_id = str(state_data.get("email_heartbeat_hw_id", "")).strip()
        if legacy_high_water_id and legacy_high_water_id not in high_water_ids_list:
            high_water_ids_list.append(legacy_high_water_id)
        high_water_ids = set(high_water_ids_list)

        # Use precise high-water timestamp for query, with a one-second overlap
        # to avoid dropping messages on second-level boundaries.
        if high_water_ts_ms > 0:
            after_s = max(0, (high_water_ts_ms // 1000) - 1)
        else:
            after_s = int(now - (24 * 3600))

        query = f"is:unread after:{after_s}"

        result = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=20)
            .execute()
        )

        messages = result.get("messages", [])

        if not messages:
            # No messages — update poll timestamp but keep high-water cursor.
            state_data["email_heartbeat_last_poll"] = now
            state_data["email_heartbeat_hw"] = now
            config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.STATE_FILE.write_text(json.dumps(state_data, indent=2))
            return

        log.info("Email heartbeat: %d new unread emails", len(messages))

        import asyncio
        from memory.triage import triage_message
        from memory.processor import batch_embed_and_store, extract_to_graph

        owner_jid = molly._get_owner_dm_jid()

        # --- Phase 1: Fetch all email metadata ---
        email_data = []  # list of (msg_id, internal_ts_ms, sender, subject, snippet, email_text)
        for msg_ref in messages:
            try:
                msg = (
                    service.users()
                    .messages()
                    .get(userId="me", id=msg_ref["id"], format="metadata",
                         metadataHeaders=["From", "Subject", "Date"])
                    .execute()
                )
                headers = {
                    h["name"].lower(): h["value"]
                    for h in msg.get("payload", {}).get("headers", [])
                }

                msg_id = str(msg.get("id", "")).strip()
                if not msg_id:
                    continue

                try:
                    internal_ts_ms = int(msg.get("internalDate", 0))
                except Exception:
                    internal_ts_ms = 0

                # Query includes a boundary overlap second; filter duplicates
                # with exact (timestamp,id) high-water checks.
                if high_water_ts_ms > 0:
                    if internal_ts_ms > 0:
                        if internal_ts_ms < high_water_ts_ms:
                            continue
                        if internal_ts_ms == high_water_ts_ms and msg_id in high_water_ids:
                            continue
                    elif msg_id in high_water_ids:
                        continue

                sender = headers.get("from", "Unknown")
                subject = headers.get("subject", "(no subject)")
                snippet = msg.get("snippet", "")
                email_text = f"From: {sender}\nSubject: {subject}\n{snippet}"
                email_data.append((msg_id, internal_ts_ms, sender, subject, snippet, email_text))
            except Exception:
                log.debug("Failed to fetch email %s", msg_ref.get("id"), exc_info=True)

        if not email_data:
            state_data["email_heartbeat_last_poll"] = now
            state_data["email_heartbeat_hw"] = now
            config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.STATE_FILE.write_text(json.dumps(state_data, indent=2))
            return

        # --- Phase 2: Triage all emails in parallel ---
        triage_tasks = [
            triage_message(email_text, sender_name=sender, group_name="Email")
            for _msg_id, _internal_ts_ms, sender, _subject, _snippet, email_text in email_data
        ]
        triage_results = await asyncio.gather(*triage_tasks, return_exceptions=True)

        # --- Phase 3: Surface urgent notifications + collect texts for memory ---
        texts_to_embed = []
        texts_to_extract = []

        for i, triage_result in enumerate(triage_results):
            if isinstance(triage_result, Exception):
                log.debug("Triage failed for email %d", i, exc_info=triage_result)
                continue

            msg_id, internal_ts_ms, sender, subject, snippet, email_text = email_data[i]

            # Queue non-noise emails for digest
            if triage_result.classification != "noise":
                from memory.email_digest import append_digest_item
                append_digest_item(
                    msg_id=msg_id, sender=sender, subject=subject,
                    snippet=snippet, classification=triage_result.classification,
                    score=triage_result.score, reason=triage_result.reason,
                    internal_ts_ms=internal_ts_ms,
                )

            if triage_result and triage_result.classification == "urgent":
                if owner_jid and molly.wa:
                    notify = (
                        f"Email from {sender}\n"
                        f"Subject: {subject}\n"
                        f"{snippet[:200]}\n\n"
                        f"Reason: {triage_result.reason}"
                    )
                    _send_surface(
                        molly,
                        owner_jid,
                        notify,
                        source="email",
                        surfaced_summary=f"{subject}: {snippet[:200]}",
                        sender_pattern=f"sender:{sender}",
                    )

            if triage_result and triage_result.classification in ("urgent", "relevant"):
                texts_to_embed.append(email_text)
                texts_to_extract.append(email_text)

        # --- Phase 4: Batch embed + parallel graph extraction ---
        if texts_to_embed:
            # Single batched embedding call instead of N individual calls
            embed_task = batch_embed_and_store(
                texts_to_embed, chat_jid="email", source="email",
            )
            # Parallel graph extraction
            graph_tasks = [
                extract_to_graph(text, chat_jid="email", source="email")
                for text in texts_to_extract
            ]
            await asyncio.gather(embed_task, *graph_tasks, return_exceptions=True)

            log.info(
                "Email heartbeat: processed %d emails, %d embedded+extracted",
                len(email_data), len(texts_to_embed),
            )

        # Update high-water mark AFTER processing so a crash doesn't lose emails.
        latest_ts_ms = 0
        latest_ids: list[str] = []
        for msg_id, internal_ts_ms, _sender, _subject, _snippet, _email_text in email_data:
            if internal_ts_ms > latest_ts_ms:
                latest_ts_ms = internal_ts_ms
                latest_ids = [msg_id]
            elif internal_ts_ms == latest_ts_ms:
                latest_ids.append(msg_id)

        latest_ids = list(dict.fromkeys([msg_id for msg_id in latest_ids if msg_id]))
        latest_ids = latest_ids[-50:]
        if latest_ts_ms <= 0 and email_data:
            latest_ts_ms = max(high_water_ts_ms, int(now * 1000))
            latest_ids = [
                msg_id
                for msg_id, _internal_ts_ms, _sender, _subject, _snippet, _email_text in email_data
                if msg_id
            ][-50:]

        if latest_ts_ms > high_water_ts_ms:
            merged_ids = latest_ids
        elif latest_ts_ms == high_water_ts_ms:
            merged_ids = list(dict.fromkeys([*high_water_ids_list, *latest_ids]))[-50:]
        else:
            merged_ids = high_water_ids_list[-50:]

        if latest_ts_ms > 0:
            state_data["email_heartbeat_hw_ts_ms"] = latest_ts_ms
            state_data["email_heartbeat_hw_ids"] = merged_ids
            state_data["email_heartbeat_hw_id"] = merged_ids[-1] if merged_ids else ""

        state_data["email_heartbeat_last_poll"] = now
        state_data["email_heartbeat_hw"] = now
        config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.STATE_FILE.write_text(json.dumps(state_data, indent=2))

    except Exception:
        log.warning("Email heartbeat check failed", exc_info=True)
