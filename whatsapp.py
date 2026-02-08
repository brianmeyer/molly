import logging
from typing import Callable

import config
from formatting import render_for_whatsapp, split_for_whatsapp
from neonize import NewClient
from neonize.proto.Neonize_pb2 import Connected, GroupInfo, Message as MessageEv
from neonize.utils import extract_text
from neonize.utils.enum import ChatPresence, ChatPresenceMedia
from neonize.utils.jid import Jid2String, build_jid

log = logging.getLogger(__name__)
SendResult = str | list[str] | None


def _resolve_sender_name(sender_jid: str, pushname: str) -> str:
    """Resolve a sender name using pushname → phone fallback."""
    phone = sender_jid.split("@")[0]

    # Prefer WhatsApp pushname when available.
    if pushname:
        return pushname

    # Final fallback: raw phone number.
    return phone


class WhatsAppClient:
    """Neonize WhatsApp client wrapper.

    Runs synchronously (call connect() from a thread).
    Fires message_callback on incoming messages from the neonize thread —
    caller is responsible for thread-safe bridging (e.g. asyncio.Queue).
    """

    def __init__(
        self,
        auth_dir,
        message_callback: Callable | None = None,
        non_whatsapp_sender: Callable[[str, str], str | None] | None = None,
    ):
        self.client = NewClient(str(auth_dir / "molly.db"))
        self._message_callback = message_callback
        self._non_whatsapp_sender = non_whatsapp_sender
        self.connected = False
        self._setup_events()

    def _setup_events(self):
        @self.client.event(Connected)
        def on_connected(client: NewClient, event: Connected):
            self.connected = True
            me = client.me
            log.info("WhatsApp connected (device: %s)", me)

        @self.client.event(MessageEv)
        def on_message(client: NewClient, event: MessageEv):
            try:
                self._handle_message_event(event)
            except Exception:
                log.error("Error handling WhatsApp message", exc_info=True)

    def _handle_message_event(self, event: MessageEv):
        info = event.Info
        chat_jid = Jid2String(info.MessageSource.Chat)
        sender_jid = Jid2String(info.MessageSource.Sender)
        is_from_me = info.MessageSource.IsFromMe
        is_group = info.MessageSource.IsGroup
        msg_id = info.ID
        # Store as ISO format for consistent comparison with search queries
        raw_ts = info.Timestamp
        if hasattr(raw_ts, 'isoformat'):
            timestamp = raw_ts.isoformat()
        else:
            timestamp = str(raw_ts)
        pushname = info.Pushname or ""

        # Extract text content from any message type
        content = extract_text(event.Message)
        if not content:
            return

        # Resolve sender name: pushname → phone number
        sender_name = _resolve_sender_name(sender_jid, pushname)

        log.debug(
            "Message from %s in %s: %s",
            sender_name,
            chat_jid,
            content[:80],
        )

        if self._message_callback:
            self._message_callback(
                {
                    "msg_id": msg_id,
                    "chat_jid": chat_jid,
                    "sender_jid": sender_jid,
                    "sender_name": sender_name,
                    "content": content,
                    "timestamp": timestamp,
                    "is_from_me": is_from_me,
                    "is_group": is_group,
                }
            )

    # --- Outbound ---

    @staticmethod
    def _is_whatsapp_jid(jid_str: str) -> bool:
        return isinstance(jid_str, str) and "@" in jid_str

    @staticmethod
    def _parse_jid(jid_str: str):
        """Convert 'user@server' string back to a neonize JID protobuf."""
        if not WhatsAppClient._is_whatsapp_jid(jid_str):
            return None
        parts = jid_str.split("@", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return build_jid(parts[0], parts[1])

    def _route_non_whatsapp_message(self, chat_jid: str, text: str) -> str | None:
        if not self._non_whatsapp_sender:
            return None
        try:
            return self._non_whatsapp_sender(chat_jid, text)
        except Exception:
            log.error("Non-WhatsApp send failed for %s", chat_jid, exc_info=True)
            return None

    def _prepare_outbound_chunks(self, text: str) -> list[str]:
        content = text or ""
        if config.WHATSAPP_PLAIN_RENDER:
            content = render_for_whatsapp(content)
        content = content.strip()
        if not content:
            return []

        if not config.WHATSAPP_CHUNKING_ENABLED:
            chunks = [content]
        else:
            chunks = split_for_whatsapp(content, max_chars=config.WHATSAPP_CHUNK_CHARS)
        if not chunks:
            return []

        # Preserve legacy signature while avoiding clutter on every chunk.
        chunks[-1] = f"{chunks[-1]}\n\n-MollyAI"
        return chunks

    def send_message(self, chat_jid: str, text: str) -> SendResult:
        """Send a text message. Returns one or many message IDs for send tracking."""
        if chat_jid.startswith("web:"):
            routed = self._route_non_whatsapp_message(chat_jid, text)
            if routed is not None:
                return routed
            log.warning("No Web UI transport for %s; skipping WhatsApp send", chat_jid)
            return None

        try:
            jid = self._parse_jid(chat_jid)
            if jid is None:
                log.warning("Skipping send to non-WhatsApp target: %s", chat_jid)
                return None

            chunks = self._prepare_outbound_chunks(text)
            if not chunks:
                log.warning("Skipping empty outbound message to %s", chat_jid)
                return None

            msg_ids: list[str] = []
            for chunk in chunks:
                resp = self.client.send_message(jid, chunk)
                if resp and resp.ID:
                    msg_ids.append(resp.ID)

            if not msg_ids:
                return None
            if len(msg_ids) == 1:
                return msg_ids[0]
            return msg_ids
        except Exception:
            log.error("Failed to send message to %s", chat_jid, exc_info=True)
            return None

    def send_typing(self, chat_jid: str):
        try:
            jid = self._parse_jid(chat_jid)
            if jid is None:
                return
            self.client.send_chat_presence(
                jid,
                ChatPresence.CHAT_PRESENCE_COMPOSING,
                ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
            )
        except Exception:
            log.debug("Failed to send typing to %s", chat_jid, exc_info=True)

    def send_typing_stopped(self, chat_jid: str):
        try:
            jid = self._parse_jid(chat_jid)
            if jid is None:
                return
            self.client.send_chat_presence(
                jid,
                ChatPresence.CHAT_PRESENCE_PAUSED,
                ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
            )
        except Exception:
            log.debug("Failed to stop typing for %s", chat_jid, exc_info=True)

    def get_group_info(self, group_jid: str) -> dict | None:
        """Fetch group metadata: name, topic, participants with resolved names.

        Returns dict with group_name, topic, and participants list, or None on error.
        """
        try:
            jid = self._parse_jid(group_jid)
            if jid is None:
                return None
            info: GroupInfo = self.client.get_group_info(jid)

            participants = []
            for p in info.Participants:
                p_jid = Jid2String(p.JID)
                p_phone = p_jid.split("@")[0]
                p_name = _resolve_sender_name(p_jid, p.DisplayName or "")
                participants.append({
                    "jid": p_jid,
                    "phone": p_phone,
                    "name": p_name,
                    "is_admin": p.IsAdmin,
                    "is_super_admin": p.IsSuperAdmin,
                })

            return {
                "group_name": info.GroupName.Name if info.GroupName else "",
                "topic": info.GroupTopic.Topic if info.GroupTopic else "",
                "participant_count": len(participants),
                "participants": participants,
            }
        except Exception:
            log.error("Failed to get group info for %s", group_jid, exc_info=True)
            return None

    def connect(self):
        """Blocking — run in a background thread."""
        log.info("Connecting to WhatsApp...")
        self.client.connect()
