import logging
from datetime import datetime

import config

log = logging.getLogger(__name__)

HEARTBEAT_SENTINEL = "HEARTBEAT_OK"


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
    """Run a heartbeat check. Sends message to first registered chat if actionable."""
    from agent import handle_message

    if not config.HEARTBEAT_FILE.exists():
        log.debug("No HEARTBEAT.md found, skipping heartbeat")
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

    # Heartbeat goes to the first registered chat
    if not molly.registered_chats:
        log.debug("No registered chats for heartbeat")
        return

    chat_jid = next(iter(molly.registered_chats))
    heartbeat_key = f"heartbeat:{chat_jid}"
    session_id = molly.sessions.get(heartbeat_key)

    try:
        response, new_session_id = await handle_message(prompt, chat_jid, session_id)

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
