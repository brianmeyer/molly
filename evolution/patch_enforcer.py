"""Patch enforcer — salvage malformed LLM output into clean diffs.

LLMs frequently produce broken diffs.  The 6-strategy salvage pipeline
tries each format in order, returning the first success.  In
primordiumevolv-min this increased patch success rate from ~55% to ~92%.
"""
from __future__ import annotations

import base64
import json
import logging
import re

from evolution._base import EnforcedPatch

log = logging.getLogger(__name__)

# <<<SEARCH / >>>REPLACE block pattern
_SEARCH_REPLACE_RE = re.compile(
    r"<<<\s*SEARCH\s*\n(.*?)\n>>>\s*REPLACE\s*\n(.*?)(?:\n<<<|$)",
    re.DOTALL,
)


def enforce(raw_output: str) -> EnforcedPatch | None:
    """Try all salvage strategies in order; return first success or None."""
    strategies = [
        ("clean_diff", _try_clean_diff),
        ("json_file_old_new", _try_json_file_old_new),
        ("json_diff_lines", _try_json_diff_lines),
        ("base64_diff", _try_base64_diff),
        ("loose_json", _try_loose_json),
        ("regex_search_replace", _try_regex_search_replace),
    ]
    for name, fn in strategies:
        try:
            result = fn(raw_output)
            if result:
                log.info("Patch enforced via strategy: %s", name)
                return EnforcedPatch(
                    target_file=result["file"],
                    diff_text=result["diff"],
                    strategy_used=name,
                    original_output=raw_output[:2000],
                )
        except Exception:
            log.debug("Strategy %s failed", name, exc_info=True)
    log.warning("All patch enforcement strategies failed")
    return None


def validate_patch(patch: EnforcedPatch) -> bool:
    """Verify a patch applies cleanly via ``git apply --check``.

    Sync helper — for async callers, use git_safety.validate_patch instead.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ["git", "apply", "--check"],
            input=patch.diff_text.encode(),
            capture_output=True,
            timeout=30,
        )
        return proc.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Salvage strategies
# ---------------------------------------------------------------------------

def _try_clean_diff(raw: str) -> dict | None:
    """Strategy 1: raw output is already a valid unified diff."""
    if not raw.strip().startswith("---") and "diff --git" not in raw:
        return None
    # Extract target file from diff header
    for line in raw.splitlines():
        if line.startswith("+++ b/"):
            return {"file": line[6:], "diff": raw.strip()}
        if line.startswith("+++ "):
            return {"file": line[4:], "diff": raw.strip()}
    return None


def _try_json_file_old_new(raw: str) -> dict | None:
    """Strategy 2: JSON with {file, old, new} keys."""
    data = json.loads(raw)
    if isinstance(data, dict) and "file" in data and "old" in data and "new" in data:
        diff = _synthesize_diff(data["file"], data["old"], data["new"])
        return {"file": data["file"], "diff": diff}
    return None


def _try_json_diff_lines(raw: str) -> dict | None:
    """Strategy 3: JSON with diff_lines array."""
    data = json.loads(raw)
    if isinstance(data, dict) and "diff_lines" in data:
        diff = "\n".join(data["diff_lines"])
        target = data.get("file", "unknown")
        return {"file": target, "diff": diff}
    return None


def _try_base64_diff(raw: str) -> dict | None:
    """Strategy 4: base64-encoded diff."""
    # Look for base64 block
    stripped = raw.strip()
    if stripped.startswith("```"):
        # Strip markdown fences
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    decoded = base64.b64decode(stripped).decode("utf-8")
    if decoded.strip().startswith("---") or "diff --git" in decoded:
        return _try_clean_diff(decoded)
    return None


def _try_loose_json(raw: str) -> dict | None:
    """Strategy 5: loose JSON (unescape newlines, strip markdown fences)."""
    cleaned = raw.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Unescape literal \\n
    cleaned = cleaned.replace("\\n", "\n").replace('\\"', '"')
    return _try_json_file_old_new(cleaned)


def _try_regex_search_replace(raw: str) -> dict | None:
    """Strategy 6: <<<SEARCH / >>>REPLACE block extraction."""
    matches = _SEARCH_REPLACE_RE.findall(raw)
    if not matches:
        return None
    # Use first match
    old_text, new_text = matches[0]
    # Try to find a filename reference
    file_match = re.search(r"(?:file|path):\s*(\S+)", raw)
    target = file_match.group(1) if file_match else "unknown"
    diff = _synthesize_diff(target, old_text, new_text)
    return {"file": target, "diff": diff}


def _synthesize_diff(filename: str, old: str, new: str) -> str:
    """Create a minimal unified diff from old/new content."""
    import difflib
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{filename}", tofile=f"b/{filename}")
    return "".join(diff)
