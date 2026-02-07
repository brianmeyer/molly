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

# ---------------------------------------------------------------------------
# Trigger definitions — maps skill filename stems to match patterns
# ---------------------------------------------------------------------------

# Each entry: (skill_name, compiled_regex, is_command)
# is_command=True means the pattern matches slash commands (handled separately)
_TRIGGER_PATTERNS: list[tuple[str, re.Pattern, bool]] = [
    # Daily Digest
    ("daily-digest", re.compile(
        r"/digest|daily\s+digest|morning\s+briefing|what'?s\s+on\s+today",
        re.IGNORECASE,
    ), False),

    # Meeting Prep
    ("meeting-prep", re.compile(
        r"prep\s+me\s+for|meeting\s+prep|brief\s+me\s+for\s+my\s+meeting|who\s+am\s+i\s+meeting",
        re.IGNORECASE,
    ), False),

    # Email Drafting
    ("email-draft", re.compile(
        r"draft\s+an?\s+email|reply\s+to\s+.+\s+saying|"
        r"email\s+\w+|write\s+an?\s+email|send\s+an?\s+email",
        re.IGNORECASE,
    ), False),

    # Follow-Up Tracking — commitment language detection
    # Two patterns: standalone commitment phrases OR commitment + deadline
    ("follow-up-tracking", re.compile(
        r"(?:"
        # Standalone commitment phrases (no deadline needed)
        r"(?:let\s+me\s+get\s+back\s+to|i\s+owe\s+\w+\s+a\s+|remind\s+me\s+to|"
        r"don'?t\s+let\s+me\s+forget\s+to)"
        r"|"
        # Commitment prefix + action/deadline
        r"(?:i'?ll|let\s+me|i\s+need\s+to|i\s+have\s+to|i\s+promised)\s+.+?"
        r"(?:by\s+\w|before\s+|tomorrow|tonight|this\s+week|next\s+week|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"send\s+|finish\s+|call\s+|follow\s+up)"
        r")",
        re.IGNORECASE,
    ), False),

    # Research & Brief
    ("research-brief", re.compile(
        r"research\s+\w|brief\s+me\s+on|what\s+should\s+i\s+know\s+about|"
        r"look\s+into\s+\w|dig\s+into\s+\w|what'?s\s+the\s+deal\s+with",
        re.IGNORECASE,
    ), False),
]

# Skills that can be activated by heartbeat events (not message matching)
HEARTBEAT_SKILLS = {"daily-digest", "meeting-prep"}


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
    global _skills_cache
    _skills_cache = None
    return _load_all()


# ---------------------------------------------------------------------------
# Trigger matching
# ---------------------------------------------------------------------------

def match_skills(message: str) -> list[Skill]:
    """Match an incoming message against skill trigger patterns.

    Returns a list of matching Skill objects (may be empty).
    Multiple skills can match simultaneously.
    """
    skills = _load_all()
    matched = []

    for skill_name, pattern, _is_command in _TRIGGER_PATTERNS:
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
