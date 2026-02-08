import json
import logging
import time
from datetime import datetime, date

import config

log = logging.getLogger(__name__)

HEARTBEAT_SENTINEL = "HEARTBEAT_OK"

# Morning digest window: first heartbeat between 7:00-7:59
DIGEST_HOUR = 7


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
    # Check for new iMessages and surface urgent ones
    await _check_imessages(molly)

    # Check for new emails and surface urgent ones
    await _check_email(molly)

    # Always send proactive messages to the owner's DM, not a random group
    chat_jid = molly._get_owner_dm_jid()
    if not chat_jid:
        log.debug("No owner DM JID available for heartbeat")
        return

    # Phase 3D: proactive skills — only run if enabled in HEARTBEAT_SKILLS
    from skills import HEARTBEAT_SKILLS

    if "daily-digest" in HEARTBEAT_SKILLS:
        await _check_morning_digest(molly, chat_jid)

    if "meeting-prep" in HEARTBEAT_SKILLS:
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
            molly._track_send(molly.wa.send_message(chat_jid, response))
        else:
            log.info("Heartbeat: nothing to report")
    except Exception:
        log.error("Heartbeat failed", exc_info=True)


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
            molly._track_send(molly.wa.send_message(chat_jid, response))
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
                    molly._track_send(molly.wa.send_message(chat_jid, response))
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
                molly._track_send(molly.wa.send_message(owner_jid, notify))

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


async def _check_email(molly):
    """Check for new emails since last heartbeat and surface important ones.

    Polls Gmail for unread messages. Runs triage on each email.
    Urgent ones get forwarded to Brian via WhatsApp. Relevant ones
    feed into the memory pipeline.
    """
    try:
        from tools.google_auth import get_gmail_service

        service = get_gmail_service()
        if not service:
            return

        # Rate limit: only poll every EMAIL_POLL_INTERVAL seconds
        state_data = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
        last_check = float(state_data.get("email_heartbeat_hw", 0))
        now = time.time()

        if last_check > 0 and (now - last_check) < config.EMAIL_POLL_INTERVAL:
            return

        # Build Gmail query for recent unread messages
        if last_check == 0:
            # First run — look back 24 hours
            lookback_seconds = 24 * 3600
        else:
            lookback_seconds = int(now - last_check)

        # Gmail query: unread messages newer than lookback period
        lookback_hours = max(1, lookback_seconds // 3600)
        query = f"is:unread newer_than:{lookback_hours}h"

        result = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=20)
            .execute()
        )

        messages = result.get("messages", [])

        if not messages:
            # No messages — still update high-water to track timing
            state_data["email_heartbeat_hw"] = now
            config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.STATE_FILE.write_text(json.dumps(state_data, indent=2))
            return

        log.info("Email heartbeat: %d new unread emails", len(messages))

        from memory.triage import triage_message
        from memory.processor import embed_and_store, extract_to_graph

        owner_jid = molly._get_owner_dm_jid()

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
                sender = headers.get("from", "Unknown")
                subject = headers.get("subject", "(no subject)")
                snippet = msg.get("snippet", "")

                email_text = f"From: {sender}\nSubject: {subject}\n{snippet}"

                # Run triage
                triage_result = await triage_message(
                    email_text,
                    sender_name=sender,
                    group_name="Email",
                )

                if triage_result and triage_result.classification == "urgent":
                    if owner_jid and molly.wa:
                        notify = (
                            f"Email from {sender}\n"
                            f"Subject: {subject}\n"
                            f"{snippet[:200]}\n\n"
                            f"Reason: {triage_result.reason}"
                        )
                        molly._track_send(molly.wa.send_message(owner_jid, notify))

                # Embed + graph extract for relevant/urgent emails
                if triage_result and triage_result.classification in ("urgent", "relevant"):
                    await embed_and_store(email_text, chat_jid="email", source="email")
                    await extract_to_graph(email_text, chat_jid="email", source="email")

            except Exception:
                log.debug("Failed to process email %s", msg_ref.get("id"), exc_info=True)

        # Update high-water mark AFTER processing so a crash doesn't lose emails
        state_data["email_heartbeat_hw"] = now
        config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.STATE_FILE.write_text(json.dumps(state_data, indent=2))

    except Exception:
        log.warning("Email heartbeat check failed", exc_info=True)
