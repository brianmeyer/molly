"""Shared deduplication engine used by maintenance and health checks.

This module intentionally has no runtime wiring yet; callers can migrate to it
when ready. The API is deterministic and pure so it is straightforward to test.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Iterable, Mapping

AliasMap = Mapping[str, Iterable[str]]

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DedupConfig:
    """Configuration for dedup scoring and near-duplicate decisions."""

    near_duplicate_threshold: float = 0.82
    max_length_delta: int = 6
    sequence_weight: float = 0.55
    edit_weight: float = 0.4
    token_weight: float = 0.05
    acronym_floor: float = 0.9
    alias_floor: float = 0.95
    enable_acronym_helper: bool = True
    enable_alias_helper: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in [0, 1]")
        if self.max_length_delta < 0:
            raise ValueError("max_length_delta must be >= 0")
        if self.sequence_weight < 0 or self.edit_weight < 0 or self.token_weight < 0:
            raise ValueError("weights must be >= 0")
        if (self.sequence_weight + self.edit_weight + self.token_weight) <= 0:
            raise ValueError("at least one weight must be > 0")
        if not 0 <= self.acronym_floor <= 1:
            raise ValueError("acronym_floor must be in [0, 1]")
        if not 0 <= self.alias_floor <= 1:
            raise ValueError("alias_floor must be in [0, 1]")


@dataclass(frozen=True)
class DedupScore:
    """Detailed score for a single name pair."""

    left: str
    right: str
    canonical_left: str
    canonical_right: str
    compact_left: str
    compact_right: str
    length_delta: int
    sequence_ratio: float
    edit_similarity: float
    token_overlap: float
    compact_match: bool
    acronym_match: bool
    alias_match: bool
    score: float


def canonical_normalize(value: str) -> str:
    """Canonical normalization used for all dedup comparisons."""
    if not value:
        return ""
    folded = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in folded if not unicodedata.combining(ch))
    lowered = without_marks.lower().replace("&", " and ")
    cleaned = _NON_ALNUM_RE.sub(" ", lowered)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def acronym_key(value: str) -> str:
    """Build a compact acronym key from a name."""
    tokens = [tok for tok in canonical_normalize(value).split(" ") if tok]
    if not tokens:
        return ""
    if len(tokens) == 1:
        token = tokens[0]
        return token if len(token) <= 4 else token[0]
    return "".join(tok[0] for tok in tokens if tok and tok[0].isalnum())


def aliases_equivalent(left: str, right: str, aliases: AliasMap | None) -> bool:
    """Return True when both names resolve to the same alias group."""
    if not aliases:
        return False
    lookup = _prepare_alias_lookup(aliases)
    left_key = _compact_key(left)
    right_key = _compact_key(right)
    if not left_key or not right_key:
        return False
    return right_key in lookup.get(left_key, set())


def score_pair(
    left: str,
    right: str,
    *,
    config: DedupConfig | None = None,
    aliases: AliasMap | None = None,
) -> DedupScore:
    """Score one pair with consistent, deterministic signals."""
    cfg = config or DedupConfig()
    alias_lookup = _prepare_alias_lookup(aliases) if aliases else {}
    return _score_pair_prepared(
        left,
        right,
        config=cfg,
        alias_lookup=alias_lookup,
    )


def is_near_duplicate(
    left: str,
    right: str,
    *,
    config: DedupConfig | None = None,
    aliases: AliasMap | None = None,
) -> bool:
    """Decision helper based on a single, shared score function."""
    cfg = config or DedupConfig()
    scored = score_pair(left, right, config=cfg, aliases=aliases)

    if scored.compact_match or scored.alias_match:
        return True
    if scored.length_delta > cfg.max_length_delta and not scored.acronym_match:
        return False
    return scored.score >= cfg.near_duplicate_threshold


def find_near_duplicates(
    names: Iterable[str],
    *,
    config: DedupConfig | None = None,
    aliases: AliasMap | None = None,
) -> list[DedupScore]:
    """Find near-duplicate pairs with deterministic ordering."""
    cfg = config or DedupConfig()
    cleaned = [name for name in names if isinstance(name, str) and name.strip()]
    alias_lookup = _prepare_alias_lookup(aliases) if aliases else {}

    matches: list[DedupScore] = []
    for left, right in combinations(cleaned, 2):
        ordered_left, ordered_right = sorted((left, right))
        scored = _score_pair_prepared(
            ordered_left,
            ordered_right,
            config=cfg,
            alias_lookup=alias_lookup,
        )
        if scored.compact_match or scored.alias_match:
            matches.append(scored)
            continue
        if scored.length_delta > cfg.max_length_delta and not scored.acronym_match:
            continue
        if scored.score >= cfg.near_duplicate_threshold:
            matches.append(scored)

    return sorted(
        matches,
        key=lambda item: (
            -item.score,
            item.canonical_left,
            item.canonical_right,
            item.left,
            item.right,
        ),
    )


def _score_pair_prepared(
    left: str,
    right: str,
    *,
    config: DedupConfig,
    alias_lookup: Mapping[str, set[str]],
) -> DedupScore:
    canonical_left = canonical_normalize(left)
    canonical_right = canonical_normalize(right)
    compact_left = _compact_from_canonical(canonical_left)
    compact_right = _compact_from_canonical(canonical_right)
    length_delta = abs(len(compact_left) - len(compact_right))

    sequence_ratio = SequenceMatcher(None, canonical_left, canonical_right).ratio()
    max_len = max(len(compact_left), len(compact_right), 1)
    edit_distance = _levenshtein(compact_left, compact_right)
    edit_similarity = 1.0 - (edit_distance / max_len)
    token_overlap = _token_jaccard(canonical_left, canonical_right)

    compact_match = bool(compact_left) and compact_left == compact_right
    acronym_match = (
        config.enable_acronym_helper
        and _acronym_equivalent(left, right, compact_left, compact_right)
    )
    alias_match = config.enable_alias_helper and _alias_equivalent_from_lookup(
        compact_left, compact_right, alias_lookup
    )

    score = _blended_score(
        sequence_ratio=sequence_ratio,
        edit_similarity=edit_similarity,
        token_overlap=token_overlap,
        config=config,
    )
    if compact_match:
        score = 1.0
    elif acronym_match:
        score = max(score, config.acronym_floor)
    if alias_match:
        score = max(score, config.alias_floor)
    score = max(0.0, min(1.0, score))

    return DedupScore(
        left=left,
        right=right,
        canonical_left=canonical_left,
        canonical_right=canonical_right,
        compact_left=compact_left,
        compact_right=compact_right,
        length_delta=length_delta,
        sequence_ratio=sequence_ratio,
        edit_similarity=edit_similarity,
        token_overlap=token_overlap,
        compact_match=compact_match,
        acronym_match=acronym_match,
        alias_match=alias_match,
        score=score,
    )


def _blended_score(
    *,
    sequence_ratio: float,
    edit_similarity: float,
    token_overlap: float,
    config: DedupConfig,
) -> float:
    total_weight = config.sequence_weight + config.edit_weight + config.token_weight
    return (
        (sequence_ratio * config.sequence_weight)
        + (edit_similarity * config.edit_weight)
        + (token_overlap * config.token_weight)
    ) / total_weight


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = {token for token in left.split(" ") if token}
    right_tokens = {token for token in right.split(" ") if token}
    if not left_tokens or not right_tokens:
        return 0.0
    if len(left_tokens) == 1 and len(right_tokens) == 1:
        # One-token comparisons can be near-duplicates without exact token overlap.
        return 1.0 if left_tokens == right_tokens else 0.5
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _acronym_equivalent(
    left: str,
    right: str,
    compact_left: str,
    compact_right: str,
) -> bool:
    left_acronym = acronym_key(left)
    right_acronym = acronym_key(right)
    if not left_acronym or not right_acronym:
        return False
    return (
        left_acronym == compact_right
        or right_acronym == compact_left
        or left_acronym == right_acronym
    )


def _prepare_alias_lookup(aliases: AliasMap | None) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {}
    if not aliases:
        return lookup
    for canonical, alt_names in aliases.items():
        group = {_compact_key(canonical)}
        group.update(_compact_key(alias) for alias in alt_names)
        group.discard("")
        if len(group) < 2:
            continue
        for key in group:
            lookup.setdefault(key, set()).update(group)
    return lookup


def _alias_equivalent_from_lookup(
    left_compact: str,
    right_compact: str,
    lookup: Mapping[str, set[str]],
) -> bool:
    if not left_compact or not right_compact:
        return False
    return right_compact in lookup.get(left_compact, set())


def _compact_key(value: str) -> str:
    return _compact_from_canonical(canonical_normalize(value))


def _compact_from_canonical(value: str) -> str:
    return value.replace(" ", "")


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_idx, left_ch in enumerate(left, start=1):
        current = [left_idx]
        for right_idx, right_ch in enumerate(right, start=1):
            insert_cost = current[right_idx - 1] + 1
            delete_cost = previous[right_idx] + 1
            replace_cost = previous[right_idx - 1] + (left_ch != right_ch)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


__all__ = [
    "AliasMap",
    "DedupConfig",
    "DedupScore",
    "acronym_key",
    "aliases_equivalent",
    "canonical_normalize",
    "find_near_duplicates",
    "is_near_duplicate",
    "score_pair",
]
