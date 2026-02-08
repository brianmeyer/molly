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
            "/graph - Show graph summary (entity count, top connected, recent)\n"
            "/graph <entity> - Look up a person, project, or topic in the knowledge graph\n"
            "/forget <topic> - Remove an entity and its relationships from the graph\n"
            "/pending - Show actions waiting for approval\n"
            "/register - Register this group for full processing + responses\n"
            "/register listen - Register this group for listen-only monitoring\n"
            "/unregister - Stop responding in this group (still stores messages)\n"
            "/groups - List all registered and monitored groups\n"
            "/status - Show uptime, model, connection info, and message stats\n"
            "/skills - List all available skills and their triggers\n"
            "/skill <name> - Show details of a specific skill\n"
            "/digest - Run the daily digest now"
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
            # Bare /graph — show overall graph summary
            try:
                from memory.graph import get_graph_summary

                summary = get_graph_summary()
                lines = [
                    "*Knowledge Graph*",
                    f"- Entities: {summary['entity_count']}",
                    f"- Relationships: {summary['relationship_count']}",
                ]
                if summary["top_connected"]:
                    lines.append("")
                    lines.append("Top connected:")
                    for e in summary["top_connected"]:
                        lines.append(
                            f"  {e['name']} ({e.get('type', '?')}) — "
                            f"{e['connections']} connections, {e.get('mentions', 0)} mentions"
                        )
                if summary["recent"]:
                    lines.append("")
                    lines.append("Recently added:")
                    for e in summary["recent"]:
                        added = (e.get("added") or "?")[:10]
                        lines.append(f"  {e['name']} ({e.get('type', '?')}) — {added}")
                return "\n".join(lines)
            except Exception as e:
                log.error("/graph summary failed", exc_info=True)
                return f"Graph summary failed: {e}"

        try:
            from memory.graph import query_entity, entity_count

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

    if cmd == "/register":
        # Check if this is a group (not a DM)
        is_owner_dm = chat_jid.split("@")[0] in config.OWNER_IDS
        if is_owner_dm:
            return "This is a DM — it's already fully active. /register is for group chats."

        mode = "respond"
        if args.strip().lower() == "listen":
            mode = "listen"

        # Try to pull group metadata via Neonize
        group_name = ""
        members = []
        member_count = 0
        if molly.wa:
            group_info = molly.wa.get_group_info(chat_jid)
            if group_info:
                group_name = group_info["group_name"]
                members = group_info["participants"]
                member_count = group_info.get("participant_count", len(members))

        # Fallback name: use JID if Neonize lookup failed
        if not group_name:
            group_name = chat_jid.split("@")[0]

        chat_entry = {
            "name": group_name,
            "chat_id": chat_jid.split("@")[0],
            "mode": mode,
        }
        if members:
            chat_entry["members"] = [
                {"name": m["name"], "phone": m["phone"], "is_admin": m["is_admin"]}
                for m in members
            ]
        if member_count:
            chat_entry["member_count"] = member_count

        molly.registered_chats[chat_jid] = chat_entry
        molly.save_registered_chats()

        if mode == "respond":
            log.info("Registered group '%s' (%s) in respond mode (%d members)", group_name, chat_jid, len(members))
            msg = f"Registered '{group_name}' for full processing + responses."
            if members:
                msg += f"\n{len(members)} members indexed."
            return msg
        else:
            log.info("Registered group '%s' (%s) in listen mode (%d members)", group_name, chat_jid, len(members))
            msg = f"Registered '{group_name}' for listen-only monitoring. I'll absorb everything but stay quiet."
            if members:
                msg += f"\n{len(members)} members indexed."
            return msg

    if cmd == "/unregister":
        if chat_jid in molly.registered_chats:
            name = molly.registered_chats[chat_jid].get("name", chat_jid)
            del molly.registered_chats[chat_jid]
            molly.save_registered_chats()
            log.info("Unregistered group %s (%s)", name, chat_jid)
            return f"Unregistered. I'll still store and embed messages but won't respond or extract here."
        return "This chat isn't registered."

    if cmd == "/groups":
        if not molly.registered_chats:
            return "No chats registered. Use /register in a group chat."

        groups = []
        dms = []
        updated = False

        for jid, info in molly.registered_chats.items():
            mode = info.get("mode", "respond")
            is_group = jid.endswith("@g.us")

            if is_group:
                # Refresh group name from Neonize
                name = info.get("name", "")
                if molly.wa and molly.wa.connected:
                    try:
                        group_info = molly.wa.get_group_info(jid)
                        if group_info and group_info["group_name"]:
                            if info.get("name") != group_info["group_name"]:
                                info["name"] = group_info["group_name"]
                                updated = True
                            name = group_info["group_name"]
                            # Also refresh member count
                            info["member_count"] = group_info["participant_count"]
                    except Exception:
                        pass

                if not name:
                    name = jid.split("@")[0]

                label = f"  {name} ({mode})"
                if info.get("member_count"):
                    label += f" — {info['member_count']} members"
                groups.append(label)
            else:
                # DM — resolve phone to contact name
                phone = jid.split("@")[0]
                name = None

                # Try Apple Contacts
                try:
                    from tools.contacts import resolve_phone_to_name
                    name = resolve_phone_to_name(phone)
                except Exception:
                    pass

                # Fall back to stored name (may be a pushname)
                if not name:
                    stored = info.get("name", "")
                    if stored and stored != phone:
                        name = stored

                # Final fallback: formatted phone number
                if not name:
                    name = f"+{phone}" if not phone.startswith("+") else phone

                # Persist resolved name if better than what's stored
                if name != info.get("name") and not name.startswith("+"):
                    info["name"] = name
                    updated = True

                is_owner = phone in config.OWNER_IDS
                label = f"  {name} ({mode})"
                if is_owner:
                    label += " [you]"
                dms.append(label)

        if updated:
            molly.save_registered_chats()

        lines = []
        if groups:
            lines.append(f"Groups ({len(groups)}):")
            lines.extend(groups)
        if dms:
            if lines:
                lines.append("")
            lines.append(f"DMs ({len(dms)}):")
            lines.extend(dms)

        return "\n".join(lines) if lines else "No chats registered."

    if cmd == "/pending":
        pending = molly.approvals.get_all_pending()
        if not pending:
            return "No actions waiting for approval."
        lines = [f"Pending Approvals ({len(pending)})\n"]
        for p in pending:
            elapsed = (datetime.now() - p.created_at).total_seconds()
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            lines.append(f"- [{p.category}] waiting {mins}m {secs}s")
            # Show first line of description
            first_line = p.description.split("\n")[0]
            if first_line:
                lines.append(f"  {first_line}")
        return "\n".join(lines)

    if cmd == "/skills":
        from skills import list_skills

        skill_list = list_skills()
        if not skill_list:
            return "No skills installed. Add .md files to workspace/skills/."
        lines = [f"Available Skills ({len(skill_list)})\n"]
        for s in skill_list:
            lines.append(f"*{s['name']}*")
            # Show first line of trigger section
            trigger_preview = s["trigger"].split("\n")[0].strip("- ")
            lines.append(f"  Trigger: {trigger_preview}")
        return "\n".join(lines)

    if cmd == "/skill":
        if not args:
            return "Usage: /skill <name>"
        from skills import get_skill_by_name

        skill = get_skill_by_name(args)
        if not skill:
            return f"No skill found matching '{args}'. Use /skills to list all."
        return skill.content

    if cmd == "/digest":
        from agent import handle_message

        prompt = (
            "Run the daily digest now. "
            "Pull today's calendar, check unread emails, and summarize what's on my plate."
        )
        # If the skill exists, it will be matched by the prompt via normal flow.
        # But we also inject it explicitly here to be sure.
        if molly.wa:
            molly.wa.send_typing(chat_jid)
        try:
            session_id = molly.sessions.get(chat_jid)
            response, new_session_id = await handle_message(
                prompt, chat_jid, session_id,
                approval_manager=molly.approvals,
                molly_instance=molly,
                source="whatsapp",
            )
            if new_session_id:
                molly.sessions[chat_jid] = new_session_id
                molly.save_sessions()
            return response or "Couldn't generate the digest right now."
        except Exception as e:
            log.error("/digest failed", exc_info=True)
            return f"Digest failed: {e}"
        finally:
            if molly.wa:
                molly.wa.send_typing_stopped(chat_jid)

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

        # Count chat modes
        respond_count = sum(
            1 for jid, info in molly.registered_chats.items()
            if info.get("mode") == "respond" and jid.split("@")[0] not in config.OWNER_IDS
        )
        listen_count = sum(
            1 for info in molly.registered_chats.values()
            if info.get("mode") == "listen"
        )

        lines = [
            f"*Molly Status*",
            f"- Model: {config.CLAUDE_MODEL}",
            f"- Connected: {connected}",
            f"- Uptime: {uptime}",
            f"- Groups responding: {respond_count}",
            f"- Groups monitoring: {listen_count}",
            f"- Active sessions: {session_count}",
            f"- Messages today: {msg_count}",
        ]
        if hasattr(molly, "last_heartbeat") and molly.last_heartbeat:
            lines.append(f"- Last heartbeat: {molly.last_heartbeat.strftime('%H:%M')}")
        return "\n".join(lines)

    return None
