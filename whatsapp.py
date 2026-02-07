import logging
from typing import Callable

from neonize import NewClient
from neonize.proto.Neonize_pb2 import Connected, GroupInfo, Message as MessageEv
from neonize.utils import extract_text
from neonize.utils.enum import ChatPresence, ChatPresenceMedia
from neonize.utils.jid import Jid2String, build_jid

log = logging.getLogger(__name__)


def _resolve_sender_name(sender_jid: str, pushname: str) -> str:
    """Resolve a sender's name using Apple Contacts → pushname → phone fallback.

    Called from the neonize message handler thread. Results are cached
    in-memory by the contacts module.
    """
    phone = sender_jid.split("@")[0]

    # Try Apple Contacts first
    try:
        from tools.contacts import resolve_phone_to_name
        contact_name = resolve_phone_to_name(phone)
        if contact_name:
            return contact_name
    except Exception:
        log.debug("Contact resolution unavailable", exc_info=True)

    # Fall back to WhatsApp pushname
    if pushname:
        return pushname

    # Final fallback: raw phone number
    return phone


class WhatsAppClient:
    """Neonize WhatsApp client wrapper.

    Runs synchronously (call connect() from a thread).
    Fires message_callback on incoming messages from the neonize thread —
    caller is responsible for thread-safe bridging (e.g. asyncio.Queue).
    """

    def __init__(self, auth_dir, message_callback: Callable | None = None):
        self.client = NewClient(str(auth_dir / "molly.db"))
        self._message_callback = message_callback
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
        timestamp = str(info.Timestamp)
        pushname = info.Pushname or ""

        # Extract text content from any message type
        content = extract_text(event.Message)
        if not content:
            return

        # Resolve sender name: Apple Contacts → pushname → phone number
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
    def _parse_jid(jid_str: str):
        """Convert 'user@server' string back to a neonize JID protobuf."""
        parts = jid_str.split("@", 1)
        return build_jid(parts[0], parts[1])

    def send_message(self, chat_jid: str, text: str) -> str | None:
        """Send a text message. Returns the message ID so callers can track it."""
        try:
            jid = self._parse_jid(chat_jid)
            resp = self.client.send_message(jid, f"{text}\n\n-MollyAI")
            return resp.ID if resp else None
        except Exception:
            log.error("Failed to send message to %s", chat_jid, exc_info=True)
            return None

    def send_typing(self, chat_jid: str):
        try:
            jid = self._parse_jid(chat_jid)
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
