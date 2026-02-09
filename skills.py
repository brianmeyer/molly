"""Skill loading and trigger matching for Phase 3D.

Skills are markdown playbooks in ~/.molly/workspace/skills/.
They're loaded into Molly's context when an incoming message
matches a skill's trigger patterns.
"""

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import config

log = logging.getLogger(__name__)

# Skills that can be activated by heartbeat events (not message matching)
HEARTBEAT_SKILLS = {"daily-digest", "meeting-prep"}

# Cache for compiled trigger patterns (rebuilt when skills reload)
_trigger_patterns: list[tuple[str, re.Pattern]] | None = None


# ---------------------------------------------------------------------------
# Parsed skill data
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    path: Path
    content: str        # full markdown content
    trigger: str        # extracted ## Trigger section text
    tools: str          # extracted ## Required Tools section text
    steps: str          # extracted ## Steps section text
    guardrails: str     # extracted ## Guardrails section text


# ---------------------------------------------------------------------------
# Skill loading
# ---------------------------------------------------------------------------

_skills_cache: dict[str, Skill] | None = None
_skills_snapshot: tuple[tuple[str, int], ...] | None = None
_last_reload_status = "cold"
_state_lock = threading.RLock()
_PENDING_SUFFIXES = (".pending", ".pending-edit")


def _parse_section(content: str, heading: str) -> str:
    """Extract text under a ## heading until the next ## or end of file."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(content)
    return match.group(1).strip() if match else ""


def _is_pending_skill_name(name: str) -> bool:
    """Return True if the skill name is a pending/temp artifact."""
    lowered = name.lower()
    return lowered.endswith(_PENDING_SUFFIXES)


def _extract_front_matter_block(content: str) -> str:
    """Extract raw YAML front matter block text from a markdown document."""
    match = re.match(r"\A---\s*\n(.*?)\n---(?:\s*\n|\s*\Z)", content, re.DOTALL)
    return match.group(1) if match else ""


def _extract_front_matter_triggers(content: str) -> list[str]:
    """Extract trigger phrases from YAML front matter `triggers:` list."""
    front_matter = _extract_front_matter_block(content)
    if not front_matter:
        return []

    lines = front_matter.splitlines()
    triggers: list[str] = []
    in_triggers_block = False
    triggers_indent = 0

    for line in lines:
        if not in_triggers_block:
            match = re.match(r"^(\s*)triggers\s*:\s*$", line)
            if match:
                in_triggers_block = True
                triggers_indent = len(match.group(1))
            continue

        if not line.strip():
            continue

        line_indent = len(line) - len(line.lstrip())
        # Exit when another key at the same or shallower indentation starts.
        if line_indent <= triggers_indent and re.match(r"^[\w-]+\s*:", line.strip()):
            break

        item_match = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if not item_match:
            continue

        item = item_match.group(1).strip()
        if (
            len(item) >= 2
            and item[0] == item[-1]
            and item[0] in {'"', "'"}
        ):
            item = item[1:-1]

        if item:
            triggers.append(item)

    return triggers


def _is_pending_skill_path(path: Path) -> bool:
    """Return True if a markdown path points to a pending/temp skill file."""
    return _is_pending_skill_name(path.stem)


def _snapshot_skill_files(skills_dir: Path) -> tuple[tuple[str, int], ...]:
    """Snapshot all *.md files as (filename, mtime_ns) tuples."""
    if not skills_dir.exists():
        return tuple()

    snapshot: list[tuple[str, int]] = []
    for path in sorted(skills_dir.glob("*.md")):
        try:
            snapshot.append((path.name, path.stat().st_mtime_ns))
        except FileNotFoundError:
            # The file changed during snapshot; next cycle will reconcile.
            continue
    return tuple(snapshot)


def _build_skill_map(
    snapshot: tuple[tuple[str, int], ...],
    *,
    strict: bool,
) -> dict[str, Skill]:
    """Build skill map from a snapshot of markdown files."""
    skills: dict[str, Skill] = {}
    skills_dir = config.SKILLS_DIR

    for file_name, _mtime_ns in snapshot:
        path = skills_dir / file_name
        if _is_pending_skill_path(path):
            continue

        try:
            content = path.read_text()
            name = path.stem
            skills[name] = Skill(
                name=name,
                path=path,
                content=content,
                trigger=_parse_section(content, "Trigger"),
                tools=_parse_section(content, "Required Tools"),
                steps=_parse_section(content, "Steps"),
                guardrails=_parse_section(content, "Guardrails"),
            )
        except Exception:
            if strict:
                raise
            log.error("Failed to load skill: %s", path, exc_info=True)

    return skills


def _build_patterns_for_skills(
    skills: dict[str, Skill],
    *,
    strict: bool,
) -> list[tuple[str, re.Pattern]]:
    """Build trigger patterns from an in-memory skill map."""
    patterns: list[tuple[str, re.Pattern]] = []

    for name, skill in skills.items():
        if _is_pending_skill_name(name):
            continue

        trigger_values = _collect_skill_trigger_values(skill)
        if not trigger_values:
            continue

        regex_parts = [_phrase_to_regex(trigger_value) for trigger_value in trigger_values]
        if not regex_parts:
            continue

        combined = "|".join(regex_parts)
        try:
            compiled = re.compile(combined, re.IGNORECASE)
            patterns.append((name, compiled))
        except re.error:
            if strict:
                raise
            log.warning("Failed to compile trigger regex for skill %s: %s", name, combined)

    return patterns


def _build_skill_state(
    *,
    strict: bool,
    snapshot: tuple[tuple[str, int], ...] | None = None,
) -> tuple[dict[str, Skill], list[tuple[str, re.Pattern]], tuple[tuple[str, int], ...]]:
    """Build a full skills state bundle without mutating module globals."""
    resolved_snapshot = snapshot if snapshot is not None else _snapshot_skill_files(config.SKILLS_DIR)
    skills = _build_skill_map(resolved_snapshot, strict=strict)
    patterns = _build_patterns_for_skills(skills, strict=strict)
    return skills, patterns, resolved_snapshot


def _compute_snapshot_diff(
    old_snapshot: tuple[tuple[str, int], ...],
    new_snapshot: tuple[tuple[str, int], ...],
) -> tuple[list[str], list[str], list[str]]:
    """Return added/removed/modified filenames between snapshots."""
    old_map = dict(old_snapshot)
    new_map = dict(new_snapshot)
    old_names = set(old_map)
    new_names = set(new_map)

    added = sorted(new_names - old_names)
    removed = sorted(old_names - new_names)
    modified = sorted(name for name in (old_names & new_names) if old_map[name] != new_map[name])
    return added, removed, modified


def _load_all() -> dict[str, Skill]:
    """Read and parse all skill files from the skills directory."""
    global _skills_cache, _trigger_patterns, _skills_snapshot, _last_reload_status
    with _state_lock:
        if _skills_cache is not None and _trigger_patterns is not None:
            return _skills_cache

        skills, patterns, snapshot = _build_skill_state(strict=False)
        _skills_cache = skills
        _trigger_patterns = patterns
        _skills_snapshot = snapshot
        _last_reload_status = "loaded"

        if not config.SKILLS_DIR.exists():
            log.debug("Skills directory not found: %s", config.SKILLS_DIR)
        log.info("Loaded %d skills: %s", len(skills), ", ".join(skills.keys()))
        log.debug("Built %d trigger patterns from skill files", len(patterns))
        return _skills_cache


def reload_skills():
    """Force reload of skills from disk (e.g., after editing skill files)."""
    global _skills_cache, _trigger_patterns, _skills_snapshot, _last_reload_status
    with _state_lock:
        skills, patterns, snapshot = _build_skill_state(strict=False)
        _skills_cache = skills
        _trigger_patterns = patterns
        _skills_snapshot = snapshot
        _last_reload_status = "reloaded"

        log.info("Reloaded %d skills: %s", len(skills), ", ".join(skills.keys()))
        log.debug("Built %d trigger patterns from skill files", len(patterns))
        return _skills_cache


def get_reload_status() -> str:
    """Return the latest hot-reload status for heartbeat observability."""
    with _state_lock:
        return _last_reload_status


def check_for_changes() -> bool:
    """Hot-reload skills if *.md filenames/mtimes changed.

    Returns True only when a full rebuild succeeds and state is swapped.
    Returns False when unchanged or when rebuild fails (old state remains).
    """
    global _skills_cache, _trigger_patterns, _skills_snapshot, _last_reload_status

    _load_all()
    with _state_lock:
        baseline = _skills_snapshot if _skills_snapshot is not None else tuple()

    latest = _snapshot_skill_files(config.SKILLS_DIR)
    if latest == baseline:
        with _state_lock:
            _last_reload_status = "unchanged"
        return False

    added, removed, modified = _compute_snapshot_diff(baseline, latest)
    log.info(
        "Skill file change detected: added=%d removed=%d modified=%d",
        len(added),
        len(removed),
        len(modified),
    )

    try:
        new_skills, new_patterns, resolved_snapshot = _build_skill_state(
            strict=True,
            snapshot=latest,
        )
    except Exception:
        with _state_lock:
            _last_reload_status = "failed"
        log.error("Skill hot-reload failed; keeping prior cache intact", exc_info=True)
        return False

    with _state_lock:
        _skills_cache = new_skills
        _trigger_patterns = new_patterns
        _skills_snapshot = resolved_snapshot
        _last_reload_status = "reloaded"

    log.info(
        "Skill hot-reload applied: skills=%d patterns=%d",
        len(new_skills),
        len(new_patterns),
    )
    return True


# ---------------------------------------------------------------------------
# Trigger parsing from markdown
# ---------------------------------------------------------------------------

def _extract_trigger_phrases(trigger_text: str) -> list[str]:
    """Extract quoted trigger phrases from a ## Trigger section.

    Finds all double-quoted strings and returns them as raw phrases.
    E.g.: '"daily digest", "morning briefing"' -> ["daily digest", "morning briefing"]
    """
    return re.findall(r'"([^"]+)"', trigger_text)


def _normalize_trigger_value(value: str) -> str:
    """Normalize trigger values for stable, case-insensitive de-duplication."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _ordered_dedup(values: list[str]) -> list[str]:
    """De-duplicate values while preserving first-seen order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalize_trigger_value(value)
        if not key or key in seen:
            continue
        deduped.append(value)
        seen.add(key)
    return deduped


def _collect_skill_trigger_values(skill: Skill) -> list[str]:
    """Collect trigger values with deterministic precedence.

    Precedence:
    1. YAML front matter `triggers:` items
    2. Quoted phrases from legacy `## Trigger` section
    3. Legacy command patterns in backticks (e.g. `/digest`)
    """
    yaml_triggers = _extract_front_matter_triggers(skill.content)
    legacy_phrases = _extract_trigger_phrases(skill.trigger)
    legacy_commands = re.findall(r'`(/\w+)`', skill.trigger)
    return _ordered_dedup(yaml_triggers + legacy_phrases + legacy_commands)


def _phrase_to_regex(phrase: str) -> str:
    """Convert a trigger phrase to a regex pattern.

    Handles:
    - Bracketed placeholders: [person] -> \\w+(?:\\s+\\w+)*
    - Whitespace: collapsed to \\s+
    - Apostrophes: made optional with '?
    - Special regex chars: escaped
    """
    # Replace bracketed placeholders first
    phrase = re.sub(r'\[.*?\]', r'PLACEHOLDER_TOKEN', phrase)

    # Escape special regex chars (but not the placeholder token)
    parts = phrase.split('PLACEHOLDER_TOKEN')
    escaped_parts = [re.escape(p) for p in parts]
    # Re-join with flexible word matching for placeholders
    result = r'\w+(?:\s+\w+)*'.join(escaped_parts)

    # Make whitespace flexible
    result = re.sub(r'\\ ', r'\\s+', result)

    # Make apostrophes optional
    result = result.replace("\\'", "'?")

    return result


def _build_trigger_patterns() -> list[tuple[str, re.Pattern]]:
    """Build trigger patterns from all loaded skill files."""
    _load_all()
    with _state_lock:
        return _trigger_patterns if _trigger_patterns is not None else []


# ---------------------------------------------------------------------------
# Trigger matching
# ---------------------------------------------------------------------------

def match_skills(message: str) -> list[Skill]:
    """Match an incoming message against skill trigger patterns.

    Trigger patterns are parsed dynamically from each skill's ## Trigger
    section. Adding a new .md file with quoted trigger phrases will be
    picked up automatically after reload_skills() or restart.

    Returns a list of matching Skill objects (may be empty).
    Multiple skills can match simultaneously.
    """
    _load_all()
    with _state_lock:
        skills = _skills_cache if _skills_cache is not None else {}
        patterns = _trigger_patterns if _trigger_patterns is not None else []

    matched = []

    for skill_name, pattern in patterns:
        if _is_pending_skill_name(skill_name):
            continue
        if skill_name in skills and not _is_pending_skill_name(skills[skill_name].name) and pattern.search(message):
            matched.append(skills[skill_name])

    if matched:
        log.info(
            "Skills matched for message: %s",
            ", ".join(s.name for s in matched),
        )

    return matched


def get_skill_context(skills: list[Skill]) -> str:
    """Format matched skills into a system prompt section."""
    if not skills:
        return ""

    parts = ["<!-- Active Skills -->"]
    parts.append(
        "The following skill playbooks are active for this message. "
        "Follow their steps and guardrails.\n"
    )
    for skill in skills:
        parts.append(skill.content)
        parts.append("---")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def list_skills() -> list[dict[str, str]]:
    """Return a summary of all available skills."""
    skills = _load_all()
    return [
        {
            "name": s.name,
            "trigger": s.trigger,
        }
        for s in skills.values()
    ]


def get_skill(name: str) -> Skill | None:
    """Get a specific skill by name (stem, e.g. 'daily-digest')."""
    if _is_pending_skill_name(name):
        return None
    skills = _load_all()
    return skills.get(name)


def get_skill_by_name(name: str) -> Skill | None:
    """Fuzzy get — try exact match first, then partial match."""
    skills = _load_all()
    requested = name.lower().strip()
    if _is_pending_skill_name(requested):
        return None

    # Exact match
    if requested in skills and not _is_pending_skill_name(skills[requested].name):
        return skills[requested]

    # Partial match (e.g. "digest" matches "daily-digest")
    for key, skill in skills.items():
        if _is_pending_skill_name(key):
            continue
        if requested in key:
            return skill

    return None
