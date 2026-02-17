import asyncio
import json
import logging
import os
import re
import shlex
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import config

log = logging.getLogger(__name__)

# Tag pattern for prompt-level approval (belt-and-suspenders with can_use_tool)
APPROVAL_TAG_RE = re.compile(
    r"\[APPROVAL_REQUIRED:\s*([^|]+?)\s*\|\s*(.+?)\s*\]",
    re.DOTALL,
)

YES_WORDS = frozenset({
    "yes", "y", "approve", "approved", "go", "proceed", "ok", "do it",
    "send", "yep", "yea", "yeah",
})
NO_WORDS = frozenset({
    "no", "n", "deny", "denied", "cancel", "stop", "don't", "dont",
    "nope", "nah",
})
APPROVE_ALL_WORDS = frozenset({
    "all",
    "yes all",
    "approve all",
    "approved all",
    "go all",
    "proceed all",
    "all yes",
})

# Primary fallback for approvals from non-WhatsApp sessions.
OWNER_PRIMARY_WHATSAPP_JID = os.getenv(
    "MOLLY_OWNER_WHATSAPP_JID", ""
)
OWNER_APPROVAL_JID_FALLBACKS: tuple[str, ...] = (OWNER_PRIMARY_WHATSAPP_JID,)
WHATSAPP_JID_SERVER_SUFFIX = ".whatsapp.net"
WHATSAPP_JID_ALLOWED_SERVERS = {"lid", "g.us", "broadcast"}


# ---------------------------------------------------------------------------
# Action tier classification
# ---------------------------------------------------------------------------

_APPLE_MCP_READ_OPS: dict[str, set[str]] = {
    "reminders": {"list", "search", "listbyid"},
    "notes": {"search", "list"},
    "messages": {"read", "unread"},
    "mail": {"unread", "search", "mailboxes", "accounts", "latest"},
    "calendar": {"search", "open", "list"},
    "maps": {"search", "directions", "listguides"},
}

_APPLE_MCP_WRITE_OPS: dict[str, set[str]] = {
    "reminders": {"create", "open"},
    "notes": {"create"},
    "messages": {"send", "schedule"},
    "mail": {"send"},
    "calendar": {"create"},
    "maps": {"save", "pin", "addtoguide", "createguide"},
}

_APPLE_MCP_TOOL_ALIAS_TIERS: dict[str, str] = {
    "list_reminders": "AUTO",
    "search_reminders": "AUTO",
    "create_reminder": "CONFIRM",
    "complete_reminder": "CONFIRM",
    "delete_reminder": "CONFIRM",
}


def _get_operation(tool_input: dict[str, Any] | None) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    raw = tool_input.get("operation")
    if not isinstance(raw, str):
        return None
    op = raw.strip().lower()
    return op or None


# ---------------------------------------------------------------------------
# Bash command safety classification (BUG-08)
# ---------------------------------------------------------------------------

# Single-word commands that are always safe (read-only, no side effects)
_SAFE_BASH_COMMANDS: frozenset[str] = frozenset({
    # File reading
    "cat", "head", "tail", "wc", "less", "more",
    # Directory listing
    "ls", "pwd", "tree",
    # Search
    "grep", "rg", "ag", "ack", "find", "which", "whereis", "type", "file",
    # Text processing
    "sort", "uniq", "cut", "tr", "diff", "comm",
    # System info
    "date", "whoami", "hostname", "uname", "env", "printenv",
    "df", "du", "free", "uptime", "ps", "id", "groups", "stat",
    # Utility
    "echo", "printf", "jq", "yq",
    "readlink", "realpath", "basename", "dirname",
    "cal", "man", "help",
    "true", "false", "test", "[",
})

# Multi-word command prefixes that are safe (read-only operations)
_SAFE_BASH_PREFIXES: tuple[tuple[str, ...], ...] = (
    # Git read-only
    ("git", "status"),
    ("git", "log"),
    ("git", "diff"),
    ("git", "show"),
    ("git", "branch"),
    ("git", "tag"),
    ("git", "remote"),
    ("git", "describe"),
    ("git", "rev-parse"),
    ("git", "ls-files"),
    ("git", "ls-tree"),
    ("git", "cat-file"),
    ("git", "shortlog"),
    ("git", "blame"),
    ("git", "stash", "list"),
    # Docker read-only
    ("docker", "ps"),
    ("docker", "images"),
    ("docker", "logs"),
    ("docker", "inspect"),
    ("docker", "stats"),
    ("docker", "version"),
    ("docker", "info"),
    # Package info
    ("npm", "list"),
    ("npm", "ls"),
    ("npm", "outdated"),
    ("npm", "view"),
    ("pip", "list"),
    ("pip", "show"),
    ("pip", "freeze"),
    ("brew", "list"),
    ("brew", "info"),
    # Version checks
    ("python", "--version"),
    ("python3", "--version"),
    ("node", "--version"),
    ("ruby", "--version"),
    ("java", "--version"),
    ("go", "version"),
    ("rustc", "--version"),
    ("cargo", "--version"),
    ("npm", "--version"),
    ("pip", "--version"),
    ("pip3", "--version"),
    ("git", "--version"),
    ("docker", "--version"),
)

# Tokens that indicate danger — if ANY of these appear anywhere in tokens,
# the command is NOT safe regardless of the base command.
_BASH_DANGER_TOKENS: frozenset[str] = frozenset({
    # Destructive
    "rm", "rmdir", "mv", "cp",
    "chmod", "chown", "chgrp",
    "mkfs", "dd", "fdisk", "mount", "umount",
    # Process control
    "kill", "killall", "pkill",
    "shutdown", "reboot", "halt", "poweroff",
    # Privilege escalation
    "sudo", "su", "doas",
    # Network
    "curl", "wget", "ssh", "scp", "sftp", "rsync",
    # Shell escapes
    "eval", "exec", "nohup", "disown",
    # System modification
    "crontab", "at",
    "useradd", "userdel", "usermod", "groupadd", "groupdel",
    "iptables", "ufw",
    "systemctl", "service", "launchctl",
    # Write-capable
    "tee",
    # Command execution via find options
    "-exec", "-execdir", "-ok", "-okdir",
})

# Commands allowed to write within config.WORKSPACE
_WORKSPACE_WRITE_COMMANDS: frozenset[str] = frozenset({"mkdir", "touch"})

# Disallowed shell syntax fragments (always denied)
_BASH_BLOCKED_SYNTAX: tuple[str, ...] = (
    "`", "$(", "${", "\n", "<(", ">(",
)


def _contains_disallowed_shell_syntax(raw: str) -> bool:
    for token in _BASH_BLOCKED_SYNTAX:
        if token in raw:
            return True
    # Reject redirects (>, >>, <, <<, 2>, &>, etc.)
    if ">" in raw or "<" in raw:
        return True
    return False


def _split_compound_commands(raw: str) -> list[str] | None:
    """Split on top-level shell command operators (&&, ||, ;, |)."""
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0

    while i < len(raw):
        ch = raw[i]
        nxt = raw[i : i + 2]

        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and i + 1 < len(raw):
                i += 1
                buf.append(raw[i])
            i += 1
            continue

        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if ch == "\\" and i + 1 < len(raw):
            buf.append(ch)
            i += 1
            buf.append(raw[i])
            i += 1
            continue

        if nxt in {"&&", "||"}:
            segment = "".join(buf).strip()
            if segment:
                segments.append(segment)
            buf = []
            i += 2
            continue

        if ch in {";", "|"}:
            segment = "".join(buf).strip()
            if segment:
                segments.append(segment)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    if quote is not None:
        return None

    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def _workspace_path_ok(path_arg: str) -> bool:
    workspace_real = os.path.realpath(str(config.WORKSPACE))
    resolved = os.path.realpath(os.path.expanduser(path_arg))
    return resolved == workspace_real or resolved.startswith(workspace_real + os.sep)


def _is_workspace_cd(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "cd":
        return False
    if len(tokens) != 2:
        return False
    return _workspace_path_ok(tokens[1])


def _is_safe_read_segment(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if _is_workspace_cd(tokens):
        return True
    for token in tokens:
        if token in _BASH_DANGER_TOKENS:
            return False

    base_cmd = tokens[0]
    if base_cmd in _SAFE_BASH_COMMANDS:
        return True
    for prefix in _SAFE_BASH_PREFIXES:
        if len(tokens) >= len(prefix) and tuple(tokens[: len(prefix)]) == prefix:
            return True
    return False


def _is_workspace_write_segment(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if _is_workspace_cd(tokens):
        return True
    base_cmd = tokens[0]
    if base_cmd not in _WORKSPACE_WRITE_COMMANDS:
        return False

    path_args = [t for t in tokens[1:] if not t.startswith("-")]
    if not path_args:
        return False
    return all(_workspace_path_ok(path_arg) for path_arg in path_args)


def is_safe_bash_command(tool_input: dict[str, Any] | None) -> bool:
    """Determine if a Bash command is safe (read-only) and can be auto-approved.

    Compound commands are split and each sub-command is validated independently.
    This prevents bypasses such as `cd /etc && cat passwd`.

    Conservative by design: returns False for anything it cannot parse or
    does not recognize.
    """
    if not isinstance(tool_input, dict):
        return False

    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return False

    raw = command.strip()
    if _contains_disallowed_shell_syntax(raw):
        return False

    segments = _split_compound_commands(raw)
    if not segments:
        return False

    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if not _is_safe_read_segment(tokens):
            return False
    return True


def is_bash_workspace_safe(tool_input: dict[str, Any] | None) -> bool:
    """Determine if a Bash command writes only within the Molly workspace.

    Allows ``mkdir`` and ``touch`` when every path argument resolves to a
    location under ``config.WORKSPACE``.  These commands only create new
    directories or empty files — they cannot overwrite or delete content.

    Path arguments are resolved with ``os.path.realpath`` to prevent
    ``../`` escape attacks.
    """
    if not isinstance(tool_input, dict):
        return False

    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return False

    raw = command.strip()
    if _contains_disallowed_shell_syntax(raw):
        return False

    segments = _split_compound_commands(raw)
    if not segments:
        return False

    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if not _is_workspace_write_segment(tokens):
            return False
    return True


def get_action_tier(tool_name: str, tool_input: dict[str, Any] | None = None) -> str:
    """Classify a tool into AUTO, CONFIRM, or BLOCKED.

    Unknown tools default to BLOCKED for safety.
    """
    operation = _get_operation(tool_input)
    alias_tier = _APPLE_MCP_TOOL_ALIAS_TIERS.get(tool_name)
    if alias_tier:
        return alias_tier

    if tool_name in _APPLE_MCP_READ_OPS:
        if operation in _APPLE_MCP_READ_OPS[tool_name]:
            return "AUTO"
        if operation in _APPLE_MCP_WRITE_OPS.get(tool_name, set()):
            return "CONFIRM"
        # Conservative fallback for unknown/omitted operations.
        return "CONFIRM"

    # Bash command-level classification: safe read-only commands and
    # workspace-scoped writes are AUTO, everything else stays CONFIRM.
    if tool_name == "Bash" and (
        is_safe_bash_command(tool_input) or is_bash_workspace_safe(tool_input)
    ):
        return "AUTO"

    for tier in ("AUTO", "CONFIRM", "BLOCKED"):
        if tool_name in config.ACTION_TIERS.get(tier, set()):
            return tier
    # Unknown tools are blocked by default
    return "BLOCKED"


def is_auto_approved_path(tool_name: str, tool_input: dict) -> bool:
    """Check if a CONFIRM-tier file operation targets an auto-approved path.

    Writes/edits to workspace/memory/ (daily logs, deep knowledge files)
    are auto-approved so Molly can manage her own memory without interrupting Brian.
    """
    if tool_name not in ("Write", "Edit"):
        return False

    file_path = tool_input.get("file_path", "")
    workspace = str(config.WORKSPACE)

    for safe_suffix in config.AUTO_APPROVE_PATHS:
        safe_path = f"{workspace}/{safe_suffix}"
        if file_path.startswith(safe_path):
            return True

    return False


def _log_approval_decision(
    tool_name: str,
    decision: str,
    response_time_s: float = 0.0,
):
    """Log an approval decision to operational memory (best-effort)."""
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        vs.log_tool_call(
            tool_name=f"approval:{tool_name}",
            parameters=json.dumps({"decision": decision}),
            success=(decision == "approved"),
            latency_ms=int(response_time_s * 1000),
            error_message="" if decision == "approved" else decision,
        )
    except Exception:
        log.debug("Failed to log approval decision: %s", tool_name, exc_info=True)


def format_approval_message(tool_name: str, tool_input: dict) -> str:
    """Format a human-readable approval message for WhatsApp."""
    lines = ["Approval needed\n"]
    lines.append(f"Action: {tool_name}")

    if tool_name == "Write":
        lines.append(f"File: {tool_input.get('file_path', 'unknown')}")
        content = tool_input.get("content", "")
        preview = content[:200] + "..." if len(content) > 200 else content
        lines.append(f"Preview: {preview}")

    elif tool_name == "Edit":
        lines.append(f"File: {tool_input.get('file_path', 'unknown')}")
        old = tool_input.get("old_string", "")[:100]
        new = tool_input.get("new_string", "")[:100]
        lines.append(f"Replacing: {old}")
        lines.append(f"With: {new}")

    elif tool_name in ("gmail_send", "gmail_reply"):
        lines.append(f"To: {tool_input.get('to', 'unknown')}")
        lines.append(f"Subject: \"{tool_input.get('subject', '')}\"")
        body = tool_input.get("body", "")
        preview = body[:200] + "..." if len(body) > 200 else body
        lines.append(f"Preview: \"{preview}\"")

    elif tool_name == "gmail_draft":
        lines.append(f"To: {tool_input.get('to', 'unknown')}")
        lines.append(f"Subject: \"{tool_input.get('subject', '')}\"")

    elif tool_name in ("calendar_create", "calendar_update"):
        lines.append(f"Event: {tool_input.get('title', tool_input.get('summary', 'unknown'))}")
        lines.append(f"When: {tool_input.get('start', 'unknown')}")
        attendees = tool_input.get("attendees")
        if attendees:
            lines.append(f"Attendees: {', '.join(attendees)}")

    elif tool_name == "calendar_delete":
        lines.append(f"Event: {tool_input.get('event_id', 'unknown')}")

    elif tool_name in ("reminders", "create_reminder"):
        operation = str(tool_input.get("operation", "create")).strip().lower() if tool_name == "reminders" else "create"
        lines.append(f"Operation: {operation or 'create'}")
        lines.append(f"List: {tool_input.get('list', tool_input.get('list_name', 'Molly'))}")
        lines.append(f"Title: {tool_input.get('title', 'unknown')}")
        if tool_input.get("due_at"):
            lines.append(f"Due: {tool_input.get('due_at')}")
        if tool_input.get("notes"):
            preview = str(tool_input.get("notes", ""))
            if len(preview) > 200:
                preview = preview[:200] + "..."
            lines.append(f"Notes: {preview}")

    else:
        # Generic format for other CONFIRM tools
        for key, value in list(tool_input.items())[:5]:
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            lines.append(f"{key}: {val_str}")

    lines.append(
        "\nReply YES to approve this action, ALL to approve all actions for this request, "
        "NO to cancel, or EDIT: [changes]"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pending approval data
# ---------------------------------------------------------------------------

@dataclass
class RequestApprovalState:
    """Per-request approval cache and inflight coalescing state."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    approved_all_confirm: bool = False
    approved_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    inflight_tool_approvals: dict[str, asyncio.Future] = field(default_factory=dict)
    tool_asks: int = 0
    prompts_sent: int = 0
    auto_approved: int = 0
    turn_tool_calls: list[str] = field(default_factory=list)
    executed_tool_calls: list[str] = field(default_factory=list)

    def reset_for_retry(self):
        """Clear transient deny/inflight state before a transport retry."""
        self.denied_tools.clear()
        self.inflight_tool_approvals.clear()
        self.turn_tool_calls.clear()
        self.executed_tool_calls.clear()


@dataclass
class PendingApproval:
    id: str
    request_id: str
    category: str
    description: str
    chat_jid: str
    response_chat_jid: str
    session_id: str | None
    future: asyncio.Future
    required_keyword: str = ""
    allow_edit: bool = True
    allow_approve_all: bool = False
    created_at: datetime = field(default_factory=datetime.now)


def _is_approve_all_reply(normalized_text: str) -> bool:
    compact = " ".join(normalized_text.strip().split())
    return compact in APPROVE_ALL_WORDS


# ---------------------------------------------------------------------------
# Approval manager
# ---------------------------------------------------------------------------

class ApprovalManager:
    """Manages approval requests sent via WhatsApp and resolved by yes/no replies."""

    def __init__(self):
        self._pending: dict[str, PendingApproval] = {}  # approval_id -> approval
        self._pending_by_request_chat: dict[str, set[str]] = {}  # request_chat_jid -> approval_ids
        self._pending_by_response_chat: dict[str, set[str]] = {}  # response_chat_jid -> approval_ids

    @staticmethod
    def _is_whatsapp_jid(chat_jid: str) -> bool:
        if not isinstance(chat_jid, str) or "@" not in chat_jid:
            return False
        user, server = chat_jid.split("@", 1)
        if not user or not server:
            return False
        return (
            server.endswith(WHATSAPP_JID_SERVER_SUFFIX)
            or server in WHATSAPP_JID_ALLOWED_SERVERS
        )

    def _iter_owner_approval_jids(self, molly) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(candidate: str | None):
            jid = (candidate or "").strip()
            if not jid or jid in seen or not self._is_whatsapp_jid(jid):
                return
            seen.add(jid)
            candidates.append(jid)

        primary_user = OWNER_PRIMARY_WHATSAPP_JID.split("@", 1)[0]
        _add(OWNER_PRIMARY_WHATSAPP_JID)

        def _add_owner_variants(owner_id: str):
            oid = owner_id.strip()
            if not oid:
                return
            if "@" in oid:
                _add(oid)
                return
            _add(f"{oid}@s.whatsapp.net")
            _add(f"{oid}@lid")

        if molly and hasattr(molly, "_get_owner_dm_jid"):
            try:
                _add(molly._get_owner_dm_jid())
            except Exception:
                log.debug("Failed resolving owner DM JID from Molly", exc_info=True)

        registered = getattr(molly, "registered_chats", {}) if molly else {}
        if isinstance(registered, dict):
            for jid in sorted(registered):
                if jid.split("@", 1)[0] in config.OWNER_IDS:
                    _add(jid)

        def _owner_sort_key(owner_id: str) -> tuple[int, str]:
            oid = owner_id.strip()
            if not oid:
                return (2, "")
            if oid == primary_user or oid.startswith(f"{primary_user}@"):
                return (0, oid)
            return (1, oid)

        for owner_id in sorted(config.OWNER_IDS, key=_owner_sort_key):
            _add_owner_variants(owner_id)

        for fallback in OWNER_APPROVAL_JID_FALLBACKS:
            _add(fallback)

        return candidates

    def _resolve_response_chat_jid(self, request_chat_jid: str, molly) -> str:
        if self._is_whatsapp_jid(request_chat_jid):
            return request_chat_jid

        for candidate in self._iter_owner_approval_jids(molly):
            if self._is_whatsapp_jid(candidate):
                if candidate != request_chat_jid:
                    log.info(
                        "[approval INFO] - Rerouting approval to WhatsApp (original chat: %s) -> %s",
                        request_chat_jid,
                        candidate,
                    )
                return candidate

        log.warning(
            "No owner WhatsApp JID available for non-WhatsApp approval source %s",
            request_chat_jid,
        )
        return request_chat_jid

    def _track_send(self, molly, message_result):
        if hasattr(molly, "_track_send"):
            try:
                molly._track_send(message_result)
            except Exception:
                log.debug("Failed to track outbound approval message", exc_info=True)

    def _retarget_pending_response_chat(self, approval: PendingApproval, response_chat_jid: str):
        if approval.response_chat_jid == response_chat_jid:
            return
        self._discard_pending_index(
            self._pending_by_response_chat,
            approval.response_chat_jid,
            approval.id,
        )
        approval.response_chat_jid = response_chat_jid
        self._add_pending_index(
            self._pending_by_response_chat,
            response_chat_jid,
            approval.id,
        )

    def _send_approval_message(
        self,
        molly,
        request_chat_jid: str,
        preferred_chat_jid: str,
        text: str,
    ) -> str | None:
        wa = getattr(molly, "wa", None)
        if not wa:
            log.warning("Approval message not sent (WhatsApp unavailable): %s", preferred_chat_jid)
            return None

        targets: list[str] = []
        seen: set[str] = set()

        def _add_target(candidate: str | None):
            target = (candidate or "").strip()
            if not target or target in seen:
                return
            seen.add(target)
            targets.append(target)

        _add_target(preferred_chat_jid)
        _add_target(request_chat_jid)
        for candidate in self._iter_owner_approval_jids(molly):
            _add_target(candidate)

        for target in targets:
            if not self._is_whatsapp_jid(target):
                continue
            message_result = wa.send_message(target, text)
            if message_result:
                self._track_send(molly, message_result)
                return target
            log.warning(
                "Approval send failed for %s (request_chat=%s preferred_chat=%s)",
                target,
                request_chat_jid,
                preferred_chat_jid,
            )

        log.error(
            "Approval message delivery failed (request_chat=%s preferred_chat=%s)",
            request_chat_jid,
            preferred_chat_jid,
        )
        return None

    @staticmethod
    def _add_pending_index(index: dict[str, set[str]], chat_jid: str, approval_id: str) -> None:
        ids = index.setdefault(chat_jid, set())
        ids.add(approval_id)

    @staticmethod
    def _discard_pending_index(index: dict[str, set[str]], chat_jid: str, approval_id: str) -> None:
        ids = index.get(chat_jid)
        if not ids:
            return
        ids.discard(approval_id)
        if not ids:
            index.pop(chat_jid, None)

    def _pending_ids_for_chat(self, chat_jid: str) -> set[str]:
        response_ids = set(self._pending_by_response_chat.get(chat_jid, set()))
        request_ids = set(self._pending_by_request_chat.get(chat_jid, set()))
        return response_ids | request_ids

    @staticmethod
    def _extract_request_id(text: str) -> str | None:
        match = re.search(r"\b(?:id|request)\s*[:#]?\s*([a-f0-9]{8})\b", text.lower())
        if match:
            return match.group(1)
        bare = re.fullmatch(r"\s*([a-f0-9]{8})\s*", text.lower())
        if bare:
            return bare.group(1)
        return None

    def _lookup_pending(
        self,
        chat_jid: str,
        request_id: str | None = None,
    ) -> PendingApproval | None:
        ids = self._pending_ids_for_chat(chat_jid)
        if not ids:
            return None

        candidates: list[PendingApproval] = []
        for pending_id in ids:
            approval = self._pending.get(pending_id)
            if approval is None:
                continue
            if request_id and approval.request_id != request_id:
                continue
            candidates.append(approval)

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # When there are multiple pending approvals and no explicit ID,
        # prefer the most recent prompt for backward compatibility.
        return max(candidates, key=lambda item: item.created_at)

    def _register_pending(self, approval: PendingApproval):
        self._pending[approval.id] = approval
        self._add_pending_index(self._pending_by_request_chat, approval.chat_jid, approval.id)
        self._add_pending_index(
            self._pending_by_response_chat,
            approval.response_chat_jid,
            approval.id,
        )

    def _remove_pending(self, approval: PendingApproval):
        existing = self._pending.pop(approval.id, None)
        if not existing:
            return
        self._discard_pending_index(self._pending_by_request_chat, existing.chat_jid, existing.id)
        self._discard_pending_index(
            self._pending_by_response_chat,
            existing.response_chat_jid,
            existing.id,
        )

    def _pop_pending_for_chat(
        self,
        chat_jid: str,
        request_id: str | None = None,
    ) -> PendingApproval | None:
        pending = self._lookup_pending(chat_jid, request_id=request_id)
        if not pending:
            return None
        self._remove_pending(pending)
        return pending

    # --- Tag detection (prompt-level fallback) ---

    @staticmethod
    def find_approval_tag(text: str) -> tuple[str, str] | None:
        """If text contains [APPROVAL_REQUIRED: cat | desc], return (cat, desc)."""
        match = APPROVAL_TAG_RE.search(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None

    @staticmethod
    def strip_approval_tag(text: str) -> str:
        """Remove the [APPROVAL_REQUIRED: ...] tag from text."""
        return APPROVAL_TAG_RE.sub("", text).strip()

    # --- Tool-level approval (code-enforced via can_use_tool) ---

    async def request_tool_approval(
        self,
        tool_name: str,
        tool_input: dict,
        chat_jid: str,
        molly,
        request_state: RequestApprovalState | None = None,
    ) -> bool | str:
        """Send an approval request on WhatsApp for a tool call and wait.

        Returns True if approved, False if denied or timed out,
        or a string with edit instructions if Brian replied "edit: ...".
        """
        if request_state is not None:
            request_state.tool_asks += 1

        if request_state is not None and request_state.approved_all_confirm:
            request_state.auto_approved += 1
            log.info("Tool auto-approved (request-level ALL grant): %s", tool_name)
            return True

        response_chat_jid = self._resolve_response_chat_jid(chat_jid, molly)
        approval_request_id = uuid.uuid4().hex[:8]

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if request_state is not None:
            request_state.inflight_tool_approvals[approval_request_id] = future

        description = format_approval_message(tool_name, tool_input)
        if request_state is not None:
            session_header = (
                f"🔧 {tool_name} requested\n"
                f"Request ID: {approval_request_id}\n"
                "Reply YES <id> to approve this tool call."
            )
            description = f"{session_header}\n\n{description}"
        else:
            description = (
                f"🔧 {tool_name} requested\n"
                f"Request ID: {approval_request_id}\n"
                "Reply YES <id> to approve this tool call.\n\n"
                f"{description}"
            )
        if response_chat_jid != chat_jid:
            description = f"{description}\n\nOrigin chat: {chat_jid}"

        approval = PendingApproval(
            id=str(uuid.uuid4()),
            request_id=approval_request_id,
            category=tool_name,
            description=description,
            chat_jid=chat_jid,
            response_chat_jid=response_chat_jid,
            session_id=None,
            future=future,
            required_keyword="",
            allow_edit=True,
            allow_approve_all=True,
        )
        self._register_pending(approval)

        # Send the structured approval message. If delivery fails, deny now
        # instead of waiting for a timeout that Brian can never answer.
        sent_chat_jid = self._send_approval_message(
            molly,
            request_chat_jid=chat_jid,
            preferred_chat_jid=response_chat_jid,
            text=description,
        )
        if not sent_chat_jid:
            self._remove_pending(approval)
            if not future.done():
                future.set_result(False)
            if request_state is not None:
                self._apply_tool_approval_result(tool_name, False, request_state)
            _log_approval_decision(tool_name, "delivery_failed", 0.0)
            log.warning(
                "Tool approval prompt delivery failed [%s] request_chat=%s response_chat=%s",
                tool_name,
                chat_jid,
                response_chat_jid,
            )
            return False
        if sent_chat_jid != approval.response_chat_jid:
            self._retarget_pending_response_chat(approval, sent_chat_jid)
        if request_state is not None:
            request_state.prompts_sent += 1
        log.info(
            "Tool approval requested [%s] request_chat=%s response_chat=%s",
            tool_name,
            chat_jid,
            approval.response_chat_jid,
        )

        # Wait for yes/no/edit or timeout
        t0 = asyncio.get_event_loop().time()
        try:
            result = await asyncio.wait_for(
                asyncio.shield(future), timeout=config.APPROVAL_TIMEOUT
            )
            elapsed = asyncio.get_event_loop().time() - t0
            final = self._apply_tool_approval_result(tool_name, result, request_state)
            if final is True:
                if isinstance(result, tuple) and result[0] == "approve_all":
                    _log_approval_decision(tool_name, "approved_all", elapsed)
                else:
                    _log_approval_decision(tool_name, "approved", elapsed)
                return True
            if isinstance(final, str):
                # Edit request — deny this call with the edit instruction so
                # the agent can modify parameters and retry
                _log_approval_decision(tool_name, "edited", elapsed)
                return final
            _log_approval_decision(tool_name, "denied", elapsed)
            return False
        except asyncio.TimeoutError:
            self._remove_pending(approval)
            if not future.done():
                future.set_result(False)
            elapsed = asyncio.get_event_loop().time() - t0
            self._apply_tool_approval_result(tool_name, False, request_state)
            _log_approval_decision(tool_name, "timed_out", elapsed)
            self._send_approval_message(
                molly,
                request_chat_jid=chat_jid,
                preferred_chat_jid=approval.response_chat_jid,
                text=f"Approval timed out for: {tool_name}",
            )
            log.info("Tool approval timed out: %s", tool_name)
            return False
        finally:
            if request_state is not None:
                current = request_state.inflight_tool_approvals.get(approval_request_id)
                if current is future:
                    request_state.inflight_tool_approvals.pop(approval_request_id, None)

    @staticmethod
    def _apply_tool_approval_result(
        tool_name: str,
        result: Any,
        request_state: RequestApprovalState | None,
    ) -> bool | str:
        if isinstance(result, tuple) and result and result[0] == "edit":
            return result[1]

        if isinstance(result, tuple) and result and result[0] == "approve_all":
            if request_state is not None:
                request_state.approved_all_confirm = True
            return True

        if result is True:
            return True

        return False

    # --- Tag-based approval (prompt-level fallback) ---

    async def request(
        self,
        category: str,
        description: str,
        chat_jid: str,
        session_id: str | None,
        molly,
    ) -> bool:
        """Send a tag-based approval request and wait for yes/no.

        Returns True if approved, False if denied or timed out.
        """
        response_chat_jid = self._resolve_response_chat_jid(chat_jid, molly)
        self._cancel_pending(chat_jid)
        if response_chat_jid != chat_jid:
            self._cancel_pending(response_chat_jid)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        approval = PendingApproval(
            id=str(uuid.uuid4()),
            request_id=uuid.uuid4().hex[:8],
            category=category,
            description=description,
            chat_jid=chat_jid,
            response_chat_jid=response_chat_jid,
            session_id=session_id,
            future=future,
            required_keyword="",
            allow_edit=True,
        )
        self._register_pending(approval)

        msg = (
            f"Approval needed\n\n"
            f"{description}\n\n"
            f"Category: {category}\n"
            f"Reply YES to proceed, NO to cancel, or EDIT: [changes]"
        )
        if response_chat_jid != chat_jid:
            msg = f"{msg}\n\nOrigin chat: {chat_jid}"
        sent_chat_jid = self._send_approval_message(
            molly,
            request_chat_jid=chat_jid,
            preferred_chat_jid=response_chat_jid,
            text=msg,
        )
        if not sent_chat_jid:
            self._remove_pending(approval)
            if not future.done():
                future.set_result(False)
            return False
        if sent_chat_jid != approval.response_chat_jid:
            self._retarget_pending_response_chat(approval, sent_chat_jid)
        log.info("Tag approval requested [%s]: %s", category, description)

        try:
            result = await asyncio.wait_for(
                future, timeout=config.APPROVAL_TIMEOUT
            )
            if result is True:
                return True
            return False
        except asyncio.TimeoutError:
            self._remove_pending(approval)
            self._send_approval_message(
                molly,
                request_chat_jid=chat_jid,
                preferred_chat_jid=approval.response_chat_jid,
                text=f"Approval timed out for: {description}",
            )
            log.info("Tag approval timed out: %s", description)
            return False

    async def request_custom_approval(
        self,
        category: str,
        description: str,
        chat_jid: str,
        molly,
        required_keyword: str = "YES",
        timeout_s: int | None = None,
        allow_edit: bool = True,
        return_reasoned_denial: bool = False,
    ) -> bool | str | tuple[str, str]:
        """Request approval with a custom required keyword (for DEPLOY gating).

        Returns:
            True if approved,
            False if denied/timeout,
            str edit instruction if allow_edit and owner replied EDIT: ...,
            ("deny", reason) when return_reasoned_denial=True and owner replied NO with a reason.
        """
        response_chat_jid = self._resolve_response_chat_jid(chat_jid, molly)
        self._cancel_pending(chat_jid)
        if response_chat_jid != chat_jid:
            self._cancel_pending(response_chat_jid)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        key = (required_keyword or "YES").strip().upper()
        approval = PendingApproval(
            id=str(uuid.uuid4()),
            request_id=uuid.uuid4().hex[:8],
            category=category,
            description=description,
            chat_jid=chat_jid,
            response_chat_jid=response_chat_jid,
            session_id=None,
            future=future,
            required_keyword=key,
            allow_edit=allow_edit,
        )
        self._register_pending(approval)

        edit_hint = ", EDIT: [changes]" if allow_edit else ""
        msg = (
            f"{description}\n\n"
            f"Reply {key} to approve, NO to reject{edit_hint}"
        )
        if response_chat_jid != chat_jid:
            msg = f"{msg}\n\nOrigin chat: {chat_jid}"
        sent_chat_jid = self._send_approval_message(
            molly,
            request_chat_jid=chat_jid,
            preferred_chat_jid=response_chat_jid,
            text=msg,
        )
        if not sent_chat_jid:
            self._remove_pending(approval)
            if not future.done():
                future.set_result(False)
            _log_approval_decision(category, "delivery_failed", 0.0)
            return False
        if sent_chat_jid != approval.response_chat_jid:
            self._retarget_pending_response_chat(approval, sent_chat_jid)
        log.info("Custom approval requested [%s] keyword=%s", category, key)

        t0 = asyncio.get_event_loop().time()
        wait_timeout = timeout_s if timeout_s is not None else config.APPROVAL_TIMEOUT
        try:
            result = await asyncio.wait_for(future, timeout=wait_timeout)
            elapsed = asyncio.get_event_loop().time() - t0
            if result is True:
                _log_approval_decision(category, "approved", elapsed)
                return True
            if isinstance(result, tuple) and result[0] == "edit":
                _log_approval_decision(category, "edited", elapsed)
                return result[1]
            if isinstance(result, tuple) and result[0] == "deny":
                _log_approval_decision(category, "denied", elapsed)
                if return_reasoned_denial:
                    return ("deny", str(result[1] if len(result) > 1 else "").strip())
                return False
            _log_approval_decision(category, "denied", elapsed)
            return False
        except asyncio.TimeoutError:
            self._remove_pending(approval)
            elapsed = asyncio.get_event_loop().time() - t0
            _log_approval_decision(category, "timed_out", elapsed)
            self._send_approval_message(
                molly,
                request_chat_jid=chat_jid,
                preferred_chat_jid=approval.response_chat_jid,
                text=f"Approval timed out for: {category}",
            )
            return False

    # --- Resolution ---

    def try_resolve(self, text: str, chat_jid: str) -> bool:
        """Check if an incoming message is a yes/no/edit for a pending approval.

        Returns True if the message was consumed as an approval response.
        """
        request_id = self._extract_request_id(text)
        pending_ids = self._pending_ids_for_chat(chat_jid)
        if len(pending_ids) > 1 and not request_id:
            # Multiple approvals are pending; require an explicit request ID.
            return False

        pending = self._lookup_pending(chat_jid, request_id=request_id)
        if not pending:
            return False

        normalized = text.strip().lower()
        if request_id:
            normalized = re.sub(
                rf"\b(?:id|request)\s*[:#]?\s*{re.escape(request_id)}\b",
                "",
                normalized,
            )
            normalized = re.sub(rf"\b{re.escape(request_id)}\b", "", normalized)
            normalized = " ".join(normalized.split())
        keyword = pending.required_keyword.strip().lower()

        if _is_approve_all_reply(normalized):
            if not pending.allow_approve_all:
                return False
            if keyword and keyword != "yes":
                return False
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if not approval:
                return False
            if not approval.future.done():
                approval.future.set_result(("approve_all", True))
            log.info("Approval GRANTED for ALL (request scope): %s", approval.category)
            return True

        if keyword and normalized == keyword:
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if not approval:
                return False
            if not approval.future.done():
                approval.future.set_result(True)
            log.info("Approval GRANTED by keyword '%s': %s", keyword, approval.category)
            return True

        if normalized in YES_WORDS:
            if keyword and keyword != "yes":
                return False
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if not approval:
                return False
            if not approval.future.done():
                approval.future.set_result(True)
            log.info("Approval GRANTED: %s", approval.category)
            return True

        if normalized in NO_WORDS or normalized.startswith("no:") or normalized.startswith("no "):
            reason = ""
            if normalized not in NO_WORDS:
                reason = normalized[2:].lstrip(": ").strip()
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if not approval:
                return False
            if not approval.future.done():
                approval.future.set_result(("deny", reason))
            if reason:
                log.info("Approval DENIED: %s (reason=%s)", approval.category, reason)
            else:
                log.info("Approval DENIED: %s", approval.category)
            return True

        if normalized.startswith("edit:") or normalized.startswith("edit "):
            if not pending.allow_edit:
                return False
            raw = text.strip()
            if normalized.startswith("edit:"):
                edit_instruction = raw[5:].strip()
            else:
                edit_instruction = raw[4:].strip()
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if not approval:
                return False
            if not approval.future.done():
                approval.future.set_result(("edit", edit_instruction))
            log.info("Approval EDIT: %s → %s", approval.category, edit_instruction)
            return True

        return False

    # --- Query ---

    def has_pending(self, chat_jid: str) -> bool:
        return bool(self._pending_ids_for_chat(chat_jid))

    def get_pending(self, chat_jid: str) -> PendingApproval | None:
        return self._lookup_pending(chat_jid)

    def get_all_pending(self) -> list[PendingApproval]:
        return list(self._pending.values())

    def cancel_pending(self, chat_jid: str):
        """Public wrapper for clearing pending approval state for one chat."""
        self._cancel_pending(chat_jid)

    def _cancel_pending(self, chat_jid: str):
        pending_ids = list(self._pending_ids_for_chat(chat_jid))
        for pending_id in pending_ids:
            approval = self._pending.get(pending_id)
            if approval is None:
                continue
            self._remove_pending(approval)
            if not approval.future.done():
                approval.future.set_result(False)
