"""Skill loading and trigger matching for Phase 3D.

Skills are markdown playbooks in ~/.molly/workspace/skills/.
They're loaded into Molly's context when an incoming message
matches a skill's trigger patterns.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import config

log = logging.getLogger(__name__)

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


def _parse_section(content: str, heading: str) -> str:
    """Extract text under a ## heading until the next ## or end of file."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(content)
    return match.group(1).strip() if match else ""


def _load_all() -> dict[str, Skill]:
    """Read and parse all skill files from the skills directory."""
    global _skills_cache
    if _skills_cache is not None:
        return _skills_cache

    skills_dir = config.WORKSPACE / "skills"
    skills: dict[str, Skill] = {}

    if not skills_dir.exists():
        log.debug("Skills directory not found: %s", skills_dir)
        _skills_cache = skills
        return skills

    for path in sorted(skills_dir.glob("*.md")):
        try:
            content = path.read_text()
            name = path.stem  # e.g. "daily-digest"
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
            log.error("Failed to load skill: %s", path, exc_info=True)

    log.info("Loaded %d skills: %s", len(skills), ", ".join(skills.keys()))
    _skills_cache = skills
    return skills


def reload_skills():
    """Force reload of skills from disk (e.g., after editing skill files)."""
    global _skills_cache, _trigger_patterns
    _skills_cache = None
    _trigger_patterns = None
    return _load_all()


# ---------------------------------------------------------------------------
# Trigger parsing from markdown
# ---------------------------------------------------------------------------

def _extract_trigger_phrases(trigger_text: str) -> list[str]:
    """Extract quoted trigger phrases from a ## Trigger section.

    Finds all double-quoted strings and returns them as raw phrases.
    E.g.: '"daily digest", "morning briefing"' -> ["daily digest", "morning briefing"]
    """
    return re.findall(r'"([^"]+)"', trigger_text)


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
    global _trigger_patterns
    if _trigger_patterns is not None:
        return _trigger_patterns

    skills = _load_all()
    patterns: list[tuple[str, re.Pattern]] = []

    for name, skill in skills.items():
        if not skill.trigger:
            continue

        phrases = _extract_trigger_phrases(skill.trigger)
        if not phrases:
            continue

        # Also extract /command patterns (e.g., "`/digest` command")
        commands = re.findall(r'`(/\w+)`', skill.trigger)
        regex_parts = [_phrase_to_regex(p) for p in phrases] + [re.escape(c) for c in commands]

        if regex_parts:
            combined = "|".join(regex_parts)
            try:
                compiled = re.compile(combined, re.IGNORECASE)
                patterns.append((name, compiled))
            except re.error:
                log.warning("Failed to compile trigger regex for skill %s: %s", name, combined)

    _trigger_patterns = patterns
    log.debug("Built %d trigger patterns from skill files", len(patterns))
    return patterns


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
    skills = _load_all()
    patterns = _build_trigger_patterns()
    matched = []

    for skill_name, pattern in patterns:
        if skill_name in skills and pattern.search(message):
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
    skills = _load_all()
    return skills.get(name)


def get_skill_by_name(name: str) -> Skill | None:
    """Fuzzy get — try exact match first, then partial match."""
    skills = _load_all()

    # Exact match
    if name in skills:
        return skills[name]

    # Partial match (e.g. "digest" matches "daily-digest")
    name_lower = name.lower().strip()
    for key, skill in skills.items():
        if name_lower in key:
            return skill

    return None
