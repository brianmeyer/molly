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

log = logging.getLogger("approval")

APPROVAL_TAG_RE = re.compile(r"\[APPROVAL_REQUIRED:\s*([^|]+?)\s*\|\s*(.+?)\s*\]", re.DOTALL)
YES_WORDS = frozenset({"yes", "y", "approve", "approved", "go", "proceed", "ok", "do it", "send", "yep", "yea", "yeah"})
NO_WORDS = frozenset({"no", "n", "deny", "denied", "cancel", "stop", "don't", "dont", "nope", "nah"})
APPROVE_ALL_WORDS = frozenset({"all", "yes all", "approve all", "approved all", "go all", "proceed all", "all yes"})

OWNER_PRIMARY_WHATSAPP_JID = os.getenv("MOLLY_OWNER_WHATSAPP_JID", "")
OWNER_APPROVAL_JID_FALLBACKS: tuple[str, ...] = (OWNER_PRIMARY_WHATSAPP_JID,)
WHATSAPP_JID_SERVER_SUFFIX = ".whatsapp.net"
WHATSAPP_JID_ALLOWED_SERVERS = {"lid", "g.us", "broadcast"}

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

_SAFE_BASH_COMMANDS = frozenset({
    "cat", "head", "tail", "wc", "less", "more", "ls", "pwd", "tree", "grep", "rg", "ag", "ack", "find", "which",
    "whereis", "type", "file", "sort", "uniq", "cut", "tr", "diff", "comm", "date", "whoami", "hostname", "uname", "env",
    "printenv", "df", "du", "free", "uptime", "ps", "id", "groups", "stat", "echo", "printf", "jq", "yq", "readlink",
    "realpath", "basename", "dirname", "cal", "man", "help", "true", "false", "test", "[",
})
_SAFE_BASH_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("git", "status"), ("git", "log"), ("git", "diff"), ("git", "show"), ("git", "branch"), ("git", "tag"),
    ("git", "remote"), ("git", "describe"), ("git", "rev-parse"), ("git", "ls-files"), ("git", "ls-tree"),
    ("git", "cat-file"), ("git", "shortlog"), ("git", "blame"), ("git", "stash", "list"), ("docker", "ps"),
    ("docker", "images"), ("docker", "logs"), ("docker", "inspect"), ("docker", "stats"), ("docker", "version"),
    ("docker", "info"), ("npm", "list"), ("npm", "ls"), ("npm", "outdated"), ("npm", "view"), ("pip", "list"),
    ("pip", "show"), ("pip", "freeze"), ("brew", "list"), ("brew", "info"), ("python", "--version"),
    ("python3", "--version"), ("node", "--version"), ("ruby", "--version"), ("java", "--version"),
    ("go", "version"), ("rustc", "--version"), ("cargo", "--version"), ("npm", "--version"), ("pip", "--version"),
    ("pip3", "--version"), ("git", "--version"), ("docker", "--version"),
)
_BASH_DANGER_TOKENS = frozenset({
    "rm", "rmdir", "mv", "cp", "chmod", "chown", "chgrp", "mkfs", "dd", "fdisk", "mount", "umount", "kill", "killall", "pkill",
    "shutdown", "reboot", "halt", "poweroff", "sudo", "su", "doas", "curl", "wget", "ssh", "scp", "sftp", "rsync", "eval", "exec",
    "nohup", "disown", "crontab", "at", "useradd", "userdel", "usermod", "groupadd", "groupdel", "iptables", "ufw", "systemctl",
    "service", "launchctl", "tee", "-exec", "-execdir", "-ok", "-okdir",
})
_WORKSPACE_WRITE_COMMANDS = frozenset({"mkdir", "touch"})
_BASH_BLOCKED_SYNTAX = ("`", "$(", "${", "\n", "<(", ">(")


def _get_operation(tool_input: dict[str, Any] | None) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    raw = tool_input.get("operation")
    if not isinstance(raw, str):
        return None
    op = raw.strip().lower()
    return op or None


def _contains_disallowed_shell_syntax(raw: str) -> bool:
    if any(token in raw for token in _BASH_BLOCKED_SYNTAX):
        return True
    return ">" in raw or "<" in raw


def _split_compound_commands(raw: str) -> list[str] | None:
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
    return bool(tokens and tokens[0] == "cd" and len(tokens) == 2 and _workspace_path_ok(tokens[1]))


def _is_safe_read_segment(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if _is_workspace_cd(tokens):
        return True
    if any(token in _BASH_DANGER_TOKENS for token in tokens):
        return False
    if tokens[0] in _SAFE_BASH_COMMANDS:
        return True
    return any(len(tokens) >= len(prefix) and tuple(tokens[: len(prefix)]) == prefix for prefix in _SAFE_BASH_PREFIXES)


def _is_workspace_write_segment(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if _is_workspace_cd(tokens):
        return True
    base_cmd = tokens[0]
    if base_cmd not in _WORKSPACE_WRITE_COMMANDS:
        return False
    path_args = [t for t in tokens[1:] if not t.startswith("-")]
    return bool(path_args) and all(_workspace_path_ok(path_arg) for path_arg in path_args)


def _is_bash_allowed(tool_input: dict[str, Any] | None, *, mode: str) -> bool:
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
        if mode == "read" and not _is_safe_read_segment(tokens):
            return False
        if mode == "workspace" and not _is_workspace_write_segment(tokens):
            return False
    return True


def is_safe_bash_command(tool_input: dict[str, Any] | None) -> bool:
    return _is_bash_allowed(tool_input, mode="read")


def is_bash_workspace_safe(tool_input: dict[str, Any] | None) -> bool:
    return _is_bash_allowed(tool_input, mode="workspace")


def get_action_tier(tool_name: str, tool_input: dict[str, Any] | None = None) -> str:
    operation = _get_operation(tool_input)
    alias_tier = _APPLE_MCP_TOOL_ALIAS_TIERS.get(tool_name)
    if alias_tier:
        return alias_tier

    if tool_name in _APPLE_MCP_READ_OPS:
        if operation in _APPLE_MCP_READ_OPS[tool_name]:
            return "AUTO"
        if operation in _APPLE_MCP_WRITE_OPS.get(tool_name, set()):
            return "CONFIRM"
        return "CONFIRM"

    if tool_name == "Bash" and (is_safe_bash_command(tool_input) or is_bash_workspace_safe(tool_input)):
        return "AUTO"

    for tier in ("AUTO", "CONFIRM", "BLOCKED"):
        if tool_name in config.ACTION_TIERS.get(tier, set()):
            return tier
    return "BLOCKED"


def is_auto_approved_path(tool_name: str, tool_input: dict) -> bool:
    if tool_name not in ("Write", "Edit"):
        return False
    file_path = tool_input.get("file_path", "")
    workspace = str(config.WORKSPACE)
    return any(file_path.startswith(f"{workspace}/{suffix}") for suffix in config.AUTO_APPROVE_PATHS)


def _log_approval_decision(tool_name: str, decision: str, response_time_s: float = 0.0) -> None:
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
    lines = ["Approval needed\n", f"Action: {tool_name}"]

    if tool_name == "Write":
        content = str(tool_input.get("content", ""))
        lines += [f"File: {tool_input.get('file_path', 'unknown')}", f"Preview: {content[:200] + '...' if len(content) > 200 else content}"]
    elif tool_name == "Edit":
        old = str(tool_input.get("old_string", ""))[:100]
        new = str(tool_input.get("new_string", ""))[:100]
        lines += [f"File: {tool_input.get('file_path', 'unknown')}", f"Replacing: {old}", f"With: {new}"]
    elif tool_name in {"gmail_send", "gmail_reply", "gmail_draft"}:
        lines += [f"To: {tool_input.get('to', 'unknown')}", f"Subject: \"{tool_input.get('subject', '')}\""]
        if tool_name != "gmail_draft":
            body = str(tool_input.get("body", ""))
            lines.append(f"Preview: \"{body[:200] + '...' if len(body) > 200 else body}\"")
    elif tool_name in {"calendar_create", "calendar_update"}:
        lines += [f"Event: {tool_input.get('title', tool_input.get('summary', 'unknown'))}", f"When: {tool_input.get('start', 'unknown')}"]
        attendees = tool_input.get("attendees")
        if attendees:
            lines.append(f"Attendees: {', '.join(attendees)}")
    elif tool_name == "calendar_delete":
        lines.append(f"Event: {tool_input.get('event_id', 'unknown')}")
    elif tool_name in {"reminders", "create_reminder"}:
        operation = str(tool_input.get("operation", "create")).strip().lower() if tool_name == "reminders" else "create"
        lines += [
            f"Operation: {operation or 'create'}",
            f"List: {tool_input.get('list', tool_input.get('list_name', 'Molly'))}",
            f"Title: {tool_input.get('title', 'unknown')}",
        ]
        if tool_input.get("due_at"):
            lines.append(f"Due: {tool_input.get('due_at')}")
        if tool_input.get("notes"):
            notes = str(tool_input.get("notes", ""))
            lines.append(f"Notes: {notes[:200] + '...' if len(notes) > 200 else notes}")
    else:
        for key, value in list(tool_input.items())[:5]:
            text = str(value)
            lines.append(f"{key}: {text[:200] + '...' if len(text) > 200 else text}")

    lines.append("\nReply YES to approve this action, ALL to approve all actions for this request, NO to cancel, or EDIT: [changes]")
    return "\n".join(lines)


@dataclass
class RequestApprovalState:
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
    return " ".join(normalized_text.strip().split()) in APPROVE_ALL_WORDS


class ApprovalManager:
    def __init__(self):
        self._pending: dict[str, PendingApproval] = {}
        self._pending_by_request_chat: dict[str, set[str]] = {}
        self._pending_by_response_chat: dict[str, set[str]] = {}

    @staticmethod
    def _is_whatsapp_jid(chat_jid: str) -> bool:
        if not isinstance(chat_jid, str) or "@" not in chat_jid:
            return False
        user, server = chat_jid.split("@", 1)
        return bool(user and server and (server.endswith(WHATSAPP_JID_SERVER_SUFFIX) or server in WHATSAPP_JID_ALLOWED_SERVERS))

    def _iter_owner_approval_jids(self, molly) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(value: str | None):
            jid = (value or "").strip()
            if jid and jid not in seen and self._is_whatsapp_jid(jid):
                seen.add(jid)
                candidates.append(jid)

        _add(OWNER_PRIMARY_WHATSAPP_JID)
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

        for owner_id in sorted(config.OWNER_IDS):
            _add(f"{owner_id}@s.whatsapp.net")
            _add(f"{owner_id}@lid")

        for fallback in OWNER_APPROVAL_JID_FALLBACKS:
            _add(fallback)

        return candidates

    def _resolve_response_chat_jid(self, request_chat_jid: str, molly) -> str:
        if self._is_whatsapp_jid(request_chat_jid):
            return request_chat_jid

        for candidate in self._iter_owner_approval_jids(molly):
            if candidate != request_chat_jid:
                log.info(
                    "[approval INFO] - Rerouting approval to WhatsApp (original chat: %s) -> %s",
                    request_chat_jid,
                    candidate,
                )
            return candidate

        log.warning("No owner WhatsApp JID available for non-WhatsApp approval source %s", request_chat_jid)
        return request_chat_jid

    def _track_send(self, molly, message_result):
        if hasattr(molly, "_track_send"):
            try:
                molly._track_send(message_result)
            except Exception:
                log.debug("Failed to track outbound approval message", exc_info=True)

    @staticmethod
    def _add_pending_index(index: dict[str, set[str]], chat_jid: str, approval_id: str) -> None:
        index.setdefault(chat_jid, set()).add(approval_id)

    @staticmethod
    def _discard_pending_index(index: dict[str, set[str]], chat_jid: str, approval_id: str) -> None:
        ids = index.get(chat_jid)
        if not ids:
            return
        ids.discard(approval_id)
        if not ids:
            index.pop(chat_jid, None)

    def _pending_ids_for_chat(self, chat_jid: str) -> set[str]:
        return set(self._pending_by_response_chat.get(chat_jid, set())) | set(self._pending_by_request_chat.get(chat_jid, set()))

    @staticmethod
    def _extract_request_id(text: str) -> str | None:
        match = re.search(r"\b(?:id|request)\s*[:#]?\s*([a-f0-9]{8})\b", text.lower())
        if match:
            return match.group(1)
        bare = re.fullmatch(r"\s*([a-f0-9]{8})\s*", text.lower())
        return bare.group(1) if bare else None

    def _lookup_pending(self, chat_jid: str, request_id: str | None = None) -> PendingApproval | None:
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
        return max(candidates, key=lambda item: item.created_at)

    def _register_pending(self, approval: PendingApproval):
        self._pending[approval.id] = approval
        self._add_pending_index(self._pending_by_request_chat, approval.chat_jid, approval.id)
        self._add_pending_index(self._pending_by_response_chat, approval.response_chat_jid, approval.id)

    def _remove_pending(self, approval: PendingApproval):
        existing = self._pending.pop(approval.id, None)
        if not existing:
            return
        self._discard_pending_index(self._pending_by_request_chat, existing.chat_jid, existing.id)
        self._discard_pending_index(self._pending_by_response_chat, existing.response_chat_jid, existing.id)

    def _pop_pending_for_chat(self, chat_jid: str, request_id: str | None = None) -> PendingApproval | None:
        pending = self._lookup_pending(chat_jid, request_id=request_id)
        if pending:
            self._remove_pending(pending)
        return pending

    def _send_approval_message(self, molly, request_chat_jid: str, preferred_chat_jid: str, text: str) -> str | None:
        wa = getattr(molly, "wa", None)
        if not wa:
            return None

        targets: list[str] = []
        seen: set[str] = set()

        def _add_target(candidate: str | None):
            target = (candidate or "").strip()
            if target and target not in seen:
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

        log.error("Approval message delivery failed (request_chat=%s preferred_chat=%s)", request_chat_jid, preferred_chat_jid)
        return None

    @staticmethod
    def find_approval_tag(text: str) -> tuple[str, str] | None:
        match = APPROVAL_TAG_RE.search(text)
        return (match.group(1).strip(), match.group(2).strip()) if match else None

    @staticmethod
    def strip_approval_tag(text: str) -> str:
        return APPROVAL_TAG_RE.sub("", text).strip()

    async def _request_impl(
        self,
        *,
        category: str,
        description: str,
        chat_jid: str,
        molly,
        session_id: str | None = None,
        request_id: str | None = None,
        required_keyword: str = "",
        allow_edit: bool = True,
        allow_approve_all: bool = False,
        timeout_s: float | None = None,
    ) -> tuple[bool, Any]:
        response_chat_jid = self._resolve_response_chat_jid(chat_jid, molly)
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        approval = PendingApproval(
            id=str(uuid.uuid4()),
            request_id=(request_id or uuid.uuid4().hex[:8]),
            category=category,
            description=description,
            chat_jid=chat_jid,
            response_chat_jid=response_chat_jid,
            session_id=session_id,
            future=future,
            required_keyword=(required_keyword or "").strip().upper(),
            allow_edit=allow_edit,
            allow_approve_all=allow_approve_all,
        )
        self._register_pending(approval)

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
            return False, False

        if sent_chat_jid != approval.response_chat_jid:
            self._discard_pending_index(self._pending_by_response_chat, approval.response_chat_jid, approval.id)
            approval.response_chat_jid = sent_chat_jid
            self._add_pending_index(self._pending_by_response_chat, sent_chat_jid, approval.id)

        try:
            result = await asyncio.wait_for(future, timeout=timeout_s if timeout_s is not None else config.APPROVAL_TIMEOUT)
            return True, result
        except asyncio.TimeoutError:
            self._remove_pending(approval)
            if not future.done():
                future.set_result(False)
            return True, False

    async def request_tool_approval(
        self,
        tool_name: str,
        tool_input: dict,
        chat_jid: str,
        molly,
        request_state: RequestApprovalState | None = None,
    ) -> bool | str:
        if request_state is not None:
            request_state.tool_asks += 1

        if request_state is not None and request_state.approved_all_confirm:
            request_state.auto_approved += 1
            return True

        approval_request_id = uuid.uuid4().hex[:8]
        description = (
            f"🔧 {tool_name} requested\n"
            f"Request ID: {approval_request_id}\n"
            "Reply YES <id> to approve this tool call.\n\n"
            f"{format_approval_message(tool_name, tool_input)}"
        )

        if request_state is not None:
            request_state.inflight_tool_approvals[approval_request_id] = asyncio.get_running_loop().create_future()

        response_chat_jid = self._resolve_response_chat_jid(chat_jid, molly)
        if response_chat_jid != chat_jid:
            description = f"{description}\n\nOrigin chat: {chat_jid}"

        t0 = asyncio.get_event_loop().time()
        delivered, result = await self._request_impl(
            category=tool_name,
            description=description,
            chat_jid=chat_jid,
            molly=molly,
            request_id=approval_request_id,
            required_keyword="",
            allow_edit=True,
            allow_approve_all=True,
        )

        if request_state is not None:
            request_state.inflight_tool_approvals.pop(approval_request_id, None)
            if delivered:
                request_state.prompts_sent += 1

        elapsed = asyncio.get_event_loop().time() - t0
        if isinstance(result, tuple):
            if result[0] == "edit":
                _log_approval_decision(tool_name, "edited", elapsed)
                return result[1]
            if result[0] == "approve_all":
                if request_state is not None:
                    request_state.approved_all_confirm = True
                _log_approval_decision(tool_name, "approved_all", elapsed)
                return True
        if result is True:
            _log_approval_decision(tool_name, "approved", elapsed)
            return True
        _log_approval_decision(tool_name, "denied", elapsed)
        return False

    async def request(self, category: str, description: str, chat_jid: str, session_id: str | None, molly) -> bool:
        self._cancel_pending(chat_jid)
        response_chat_jid = self._resolve_response_chat_jid(chat_jid, molly)
        if response_chat_jid != chat_jid:
            self._cancel_pending(response_chat_jid)

        msg = (
            "Approval needed\n\n"
            f"{description}\n\n"
            f"Category: {category}\n"
            "Reply YES to proceed, NO to cancel, or EDIT: [changes]"
        )
        if response_chat_jid != chat_jid:
            msg = f"{msg}\n\nOrigin chat: {chat_jid}"
        result = await self._request_impl(
            category=category,
            description=msg,
            chat_jid=chat_jid,
            molly=molly,
            session_id=session_id,
            allow_edit=True,
            allow_approve_all=False,
        )
        return bool(result[1] is True)

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
        self._cancel_pending(chat_jid)
        response_chat_jid = self._resolve_response_chat_jid(chat_jid, molly)
        if response_chat_jid != chat_jid:
            self._cancel_pending(response_chat_jid)

        key = (required_keyword or "YES").strip().upper()
        msg = f"{description}\n\nReply {key} to approve, NO to reject{', EDIT: [changes]' if allow_edit else ''}"
        if response_chat_jid != chat_jid:
            msg = f"{msg}\n\nOrigin chat: {chat_jid}"

        t0 = asyncio.get_event_loop().time()
        _delivered, result = await self._request_impl(
            category=category,
            description=msg,
            chat_jid=chat_jid,
            molly=molly,
            required_keyword=key,
            allow_edit=allow_edit,
            timeout_s=timeout_s,
        )
        elapsed = asyncio.get_event_loop().time() - t0

        if result is True:
            _log_approval_decision(category, "approved", elapsed)
            return True
        if isinstance(result, tuple) and result[0] == "edit":
            _log_approval_decision(category, "edited", elapsed)
            return result[1]
        if isinstance(result, tuple) and result[0] == "deny":
            _log_approval_decision(category, "denied", elapsed)
            return ("deny", str(result[1] if len(result) > 1 else "").strip()) if return_reasoned_denial else False

        _log_approval_decision(category, "denied", elapsed)
        return False

    def try_resolve(self, text: str, chat_jid: str) -> bool:
        request_id = self._extract_request_id(text)
        pending_ids = self._pending_ids_for_chat(chat_jid)
        if len(pending_ids) > 1 and not request_id:
            return False

        pending = self._lookup_pending(chat_jid, request_id=request_id)
        if not pending:
            return False

        normalized = text.strip().lower()
        if request_id:
            normalized = re.sub(rf"\b(?:id|request)\s*[:#]?\s*{re.escape(request_id)}\b", "", normalized)
            normalized = re.sub(rf"\b{re.escape(request_id)}\b", "", normalized)
            normalized = " ".join(normalized.split())
        keyword = pending.required_keyword.strip().lower()

        if _is_approve_all_reply(normalized):
            if pending.allow_approve_all and (not keyword or keyword == "yes"):
                approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
                if approval and not approval.future.done():
                    approval.future.set_result(("approve_all", True))
                    return True
            return False

        if keyword and normalized == keyword:
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if approval and not approval.future.done():
                approval.future.set_result(True)
                return True
            return False

        if normalized in YES_WORDS:
            if keyword and keyword != "yes":
                return False
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if approval and not approval.future.done():
                approval.future.set_result(True)
                return True
            return False

        if normalized in NO_WORDS or normalized.startswith("no:") or normalized.startswith("no "):
            reason = ""
            if normalized not in NO_WORDS:
                reason = normalized[2:].lstrip(": ").strip()
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if approval and not approval.future.done():
                approval.future.set_result(("deny", reason))
                return True
            return False

        if normalized.startswith("edit:") or normalized.startswith("edit "):
            if not pending.allow_edit:
                return False
            raw = text.strip()
            edit_instruction = raw[5:].strip() if normalized.startswith("edit:") else raw[4:].strip()
            approval = self._pop_pending_for_chat(chat_jid, request_id=request_id)
            if approval and not approval.future.done():
                approval.future.set_result(("edit", edit_instruction))
                return True

        return False

    def has_pending(self, chat_jid: str) -> bool:
        return bool(self._pending_ids_for_chat(chat_jid))

    def get_pending(self, chat_jid: str) -> PendingApproval | None:
        return self._lookup_pending(chat_jid)

    def get_all_pending(self) -> list[PendingApproval]:
        return list(self._pending.values())

    def cancel_pending(self, chat_jid: str):
        self._cancel_pending(chat_jid)

    def _cancel_pending(self, chat_jid: str):
        for pending_id in list(self._pending_ids_for_chat(chat_jid)):
            approval = self._pending.get(pending_id)
            if approval is None:
                continue
            self._remove_pending(approval)
            if not approval.future.done():
                approval.future.set_result(False)
