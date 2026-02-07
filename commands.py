import logging
from datetime import date, datetime, timezone

import config

log = logging.getLogger(__name__)


async def handle_command(text: str, chat_jid: str, molly) -> str | None:
    """Process a slash command. Returns response text or None."""
    parts = text.strip().split(maxsplit=1)
    cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        return (
            "Molly Commands\n"
            "\n"
            "/help - Show this list\n"
            "/clear - Reset conversation session\n"
            "/memory - Show what Molly remembers (MEMORY.md)\n"
            "/graph <entity> - Look up a person, project, or topic in the knowledge graph\n"
            "/forget <topic> - Remove an entity and its relationships from the graph\n"
            "/status - Show uptime, model, connection info, and message stats"
        )

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
        try:
            from memory.graph import query_entity, entity_count, relationship_count

            entity = query_entity(args)
            if not entity:
                total_e = entity_count()
                return f"No entity found for '{args}'. ({total_e} entities in graph)"

            lines = [f"*{entity['name']}* ({entity.get('entity_type', '?')})"]
            lines.append(f"Mentions: {entity.get('mention_count', 0)}")
            lines.append(f"First seen: {entity.get('first_mentioned', '?')[:10]}")
            lines.append(f"Last seen: {entity.get('last_mentioned', '?')[:10]}")
            if entity.get("aliases"):
                lines.append(f"Aliases: {', '.join(entity['aliases'])}")
            for rel in entity.get("relationships", []):
                rtype = rel["type"].replace("_", " ").lower()
                if rel["direction"] == "outgoing":
                    lines.append(f"  → {rtype} {rel['target']}")
                else:
                    lines.append(f"  ← {rel['source']} {rtype}")
            return "\n".join(lines)
        except Exception as e:
            log.error("/graph failed", exc_info=True)
            return f"Graph query failed: {e}"

    if cmd == "/forget":
        if not args:
            return "Usage: /forget <topic>"
        try:
            from memory.graph import delete_entity

            deleted = delete_entity(args)
            if deleted:
                return f"Forgot '{args}' — entity and all relationships removed from graph."
            return f"No entity '{args}' found in graph."
        except Exception as e:
            log.error("/forget failed", exc_info=True)
            return f"Forget failed: {e}"

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
