from __future__ import annotations

from difflib import SequenceMatcher

import pytest

from memory.dedup import (
    DedupConfig,
    acronym_key,
    aliases_equivalent,
    canonical_normalize,
    find_near_duplicates,
    is_near_duplicate,
    score_pair,
)


HEALTH_FLAGS_MAINTENANCE_MISSES = [
    ("U.S.", "US"),
    ("A.J.", "AJ"),
    ("I.B.M", "IBM"),
    ("S.F.", "SF"),
]


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
            current.append(
                min(
                    current[right_idx - 1] + 1,
                    previous[right_idx] + 1,
                    previous[right_idx - 1] + (left_ch != right_ch),
                )
            )
        previous = current
    return previous[-1]


def _health_duplicate_signal(left: str, right: str, threshold: int = 3) -> bool:
    if abs(len(left) - len(right)) > threshold:
        return False
    return _levenshtein(left.lower(), right.lower()) < threshold


def _maintenance_duplicate_signal(left: str, right: str) -> bool:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio() >= 0.85 and left != right


@pytest.mark.parametrize(("left", "right"), HEALTH_FLAGS_MAINTENANCE_MISSES)
def test_fixture_health_flags_but_maintenance_misses(left: str, right: str) -> None:
    assert _health_duplicate_signal(left, right)
    assert not _maintenance_duplicate_signal(left, right)


@pytest.mark.parametrize(("left", "right"), HEALTH_FLAGS_MAINTENANCE_MISSES)
def test_shared_engine_catches_known_mismatch_pairs(left: str, right: str) -> None:
    scored = score_pair(left, right)
    assert scored.compact_match
    assert scored.score == pytest.approx(1.0)
    assert is_near_duplicate(left, right)


def test_canonical_normalization_is_stable() -> None:
    assert canonical_normalize("  Crème-Brûlée, Inc.  ") == "creme brulee inc"
    assert canonical_normalize("AT&T") == "at and t"


def test_acronym_helper_can_promote_near_duplicate_detection() -> None:
    scored = score_pair("NYU", "New York University")
    assert acronym_key("New York University") == "nyu"
    assert scored.acronym_match
    assert is_near_duplicate("NYU", "New York University")


def test_alias_helper_is_optional_and_lightweight() -> None:
    aliases = {"International Business Machines": ["IBM", "I.B.M."]}
    strict = DedupConfig(
        near_duplicate_threshold=0.99,
        enable_acronym_helper=False,
        max_length_delta=2,
    )
    assert aliases_equivalent("IBM", "International Business Machines", aliases)
    assert not is_near_duplicate(
        "IBM",
        "International Business Machines",
        config=strict,
    )
    assert is_near_duplicate(
        "IBM",
        "International Business Machines",
        config=strict,
        aliases=aliases,
    )


def test_threshold_is_configurable() -> None:
    left = "Jon"
    right = "John"
    loose = DedupConfig(near_duplicate_threshold=0.75)
    strict = DedupConfig(near_duplicate_threshold=0.95)

    assert is_near_duplicate(left, right, config=loose)
    assert not is_near_duplicate(left, right, config=strict)


def test_find_near_duplicates_is_deterministic() -> None:
    names = [
        "IBM",
        "I.B.M",
        "US",
        "U.S.",
        "OpenAI",
        "Open AI",
        "Completely Different",
    ]

    first = find_near_duplicates(names)
    second = find_near_duplicates(list(reversed(names)))

    first_pairs = [(item.left, item.right, round(item.score, 6)) for item in first]
    second_pairs = [(item.left, item.right, round(item.score, 6)) for item in second]

    assert first_pairs == second_pairs
    assert ("I.B.M", "IBM", 1.0) in first_pairs
    assert ("U.S.", "US", 1.0) in first_pairs
