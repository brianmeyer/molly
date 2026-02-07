import logging
from datetime import date, datetime, timezone

import config

log = logging.getLogger(__name__)


async def handle_command(text: str, chat_jid: str, molly) -> str | None:
    """Process a slash command. Returns response text or None."""
    parts = text.strip().split(maxsplit=1)
    cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/clear":
        molly.sessions.pop(chat_jid, None)
        molly.save_sessions()
        return "Session cleared. Fresh start."

    if cmd == "/memory":
        memory_path = config.WORKSPACE / "MEMORY.md"
        if memory_path.exists():
            content = memory_path.read_text().strip()
            return content if content else "MEMORY.md is empty."
        return "MEMORY.md not found."

    if cmd == "/graph":
        if not args:
            return "Usage: /graph <entity name>"
        return f"Graph queries available in Phase 2. (searched: {args})"

    if cmd == "/forget":
        if not args:
            return "Usage: /forget <topic>"
        return f"Forget functionality available in Phase 2. (topic: {args})"

    if cmd == "/status":
        uptime = "unknown"
        if hasattr(molly, "start_time") and molly.start_time:
            delta = datetime.now() - molly.start_time
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime = f"{hours}h {minutes}m {seconds}s"

        connected = molly.wa.connected if molly.wa else False
        session_count = len(molly.sessions)
        chat_count = len(molly.registered_chats)
        today = date.today().isoformat()
        msg_count = molly.db.get_message_count(since=today) if molly.db else 0

        lines = [
            f"*Molly Status*",
            f"- Model: {config.CLAUDE_MODEL}",
            f"- Connected: {connected}",
            f"- Uptime: {uptime}",
            f"- Registered chats: {chat_count}",
            f"- Active sessions: {session_count}",
            f"- Messages today: {msg_count}",
        ]
        if hasattr(molly, "last_heartbeat") and molly.last_heartbeat:
            lines.append(f"- Last heartbeat: {molly.last_heartbeat.strftime('%H:%M')}")
        return "\n".join(lines)

    return None
