"""Channel-specific text rendering helpers."""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)

_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def render_for_whatsapp(text: str) -> str:
    """Convert markdown-heavy output into plain WhatsApp-friendly text."""
    if not text:
        return ""

    rendered = text.replace("\r\n", "\n").replace("\r", "\n")
    rendered = _convert_fenced_code(rendered)
    rendered = _convert_links(rendered)
    rendered = _convert_headings(rendered)
    rendered = _convert_tables(rendered)
    rendered = _INLINE_CODE_RE.sub(r"\1", rendered)
    rendered = _collapse_whitespace(rendered)
    return rendered.strip()


def split_for_whatsapp(text: str, max_chars: int = 1400) -> list[str]:
    """Split long text into paragraph-aware chunks."""
    value = (text or "").strip()
    if not value:
        return []

    limit = max(200, int(max_chars))
    if len(value) <= limit:
        return [value]

    paragraphs = [p.strip() for p in value.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        pieces = _split_piece(paragraph, limit)
        for piece in pieces:
            if not current:
                current = piece
                continue
            candidate = f"{current}\n\n{piece}"
            if len(candidate) <= limit:
                current = candidate
                continue
            chunks.append(current)
            current = piece

    if current:
        chunks.append(current)
    return _merge_orphan_heading_chunks(chunks, limit)


def _convert_fenced_code(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        body = match.group(1).strip("\n")
        return body

    return _FENCE_RE.sub(_repl, text)


def _convert_links(text: str) -> str:
    return _MD_LINK_RE.sub(r"\1: \2", text)


def _convert_headings(text: str) -> str:
    return _HEADING_RE.sub("", text)


def _convert_tables(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and _looks_like_table_row(line) and _looks_like_separator(lines[i + 1]):
            headers = _split_table_row(line)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and _looks_like_table_row(lines[i]):
                row = _split_table_row(lines[i])
                if row:
                    rows.append(row)
                i += 1
            out.extend(_table_rows_to_bullets(headers, rows))
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _looks_like_table_row(line: str) -> bool:
    value = line.strip()
    if not value or "|" not in value:
        return False
    cells = _split_table_row(value)
    return len(cells) >= 2 and any(cell for cell in cells)


def _looks_like_separator(line: str) -> bool:
    cells = _split_table_row(line)
    if not cells:
        return False
    return all(_TABLE_SEPARATOR_CELL_RE.match(cell) for cell in cells)


def _split_table_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return [cell.strip() for cell in value.split("|")]


def _table_rows_to_bullets(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not headers:
        return []

    if not rows:
        return ["- " + "; ".join(headers)]

    result: list[str] = []
    for row in rows:
        pairs = []
        for idx, header in enumerate(headers):
            value = row[idx].strip() if idx < len(row) else ""
            if value:
                pairs.append(f"{header}: {value}")
        if pairs:
            result.append("- " + "; ".join(pairs))
    return result


def _collapse_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    compact: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip():
            compact.append(line)
            blank_count = 0
            continue
        blank_count += 1
        if blank_count <= 1:
            compact.append("")
    return "\n".join(compact)


def _split_piece(piece: str, limit: int) -> list[str]:
    value = piece.strip()
    if len(value) <= limit:
        return [value]

    parts: list[str] = []
    start = 0
    while start < len(value):
        end = min(start + limit, len(value))
        if end < len(value):
            soft_break = max(value.rfind("\n", start + 1, end), value.rfind(" ", start + 1, end))
            if soft_break > start + int(limit * 0.6):
                end = soft_break

        chunk = value[start:end].strip()
        if chunk:
            parts.append(chunk)
        start = end
        while start < len(value) and value[start] in {" ", "\n"}:
            start += 1

    return parts


def _merge_orphan_heading_chunks(chunks: list[str], limit: int) -> list[str]:
    """Merge tiny heading-only chunks with the next chunk when feasible."""
    if len(chunks) < 2:
        return chunks

    out = list(chunks)
    i = 0
    while i < len(out) - 1:
        head = out[i].strip()
        if not _is_heading_only_chunk(head):
            i += 1
            continue

        available = limit - len(head) - 2
        if available < 60:
            i += 1
            continue

        next_chunk = out[i + 1]
        prefix, remainder = _split_prefix(next_chunk, available)
        if not prefix:
            i += 1
            continue

        out[i] = f"{head}\n\n{prefix}".strip()
        if remainder:
            out[i + 1] = remainder
            i += 1
        else:
            out.pop(i + 1)

    return out


def _is_heading_only_chunk(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if "\n" in value:
        return False
    return len(value) <= 40 and value.endswith(":")


def _split_prefix(text: str, max_len: int) -> tuple[str, str]:
    value = (text or "").strip()
    if not value:
        return "", ""

    if len(value) <= max_len:
        return value, ""

    end = min(max_len, len(value))
    soft_break = max(value.rfind("\n", 0, end), value.rfind(" ", 0, end))
    if soft_break > int(end * 0.6):
        end = soft_break

    prefix = value[:end].strip()
    remainder = value[end:].lstrip(" \n")
    return prefix, remainder
