"""Relationship quality audit for nightly maintenance (Step 4b).

Two-tier system:
  Tier 1 — Deterministic checks (always runs): entity-type/rel-type mismatches,
           self-refs, zombies, contradictions, low-confidence, RELATED_TO accumulation.
  Tier 2 — Model-based review via Kimi K2.5 (opt-in): semantic review of flagged edges.
"""

import logging
from collections import defaultdict

import config
from memory.graph import (
    VALID_REL_TYPES,
    delete_self_referencing_rels,
    delete_specific_relationship,
    get_relationship_type_distribution,
    get_relationships_for_audit,
    reclassify_relationship,
    set_relationship_audit_status,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTITY_REL_COMPATIBILITY: dict[tuple[str, str], set[str]] = {
    ("Person", "Person"): {
        "KNOWS", "CLASSMATE_OF", "MENTORS", "MENTORED_BY", "REPORTS_TO",
        "COLLABORATES_WITH", "DISCUSSED_WITH", "CONTACT_OF", "PARENT_OF",
        "CHILD_OF", "RELATED_TO",
    },
    ("Person", "Organization"): {
        "WORKS_AT", "STUDIED_AT", "ALUMNI_OF", "CUSTOMER_OF",
        "ATTENDS", "RECEIVED_FROM", "RELATED_TO",
    },
    ("Person", "Technology"): {
        "USES", "CREATED", "INTERESTED_IN", "WORKS_ON", "RELATED_TO",
    },
    ("Person", "Project"): {
        "WORKS_ON", "CREATED", "MANAGES", "INTERESTED_IN", "RELATED_TO",
    },
    ("Person", "Place"): {"LOCATED_IN", "RELATED_TO"},
    ("Person", "Concept"): {"INTERESTED_IN", "DISCUSSED_WITH", "RELATED_TO"},
    ("Organization", "Organization"): {
        "COLLABORATES_WITH", "CUSTOMER_OF", "DEPENDS_ON", "RELATED_TO",
    },
    ("Organization", "Place"): {"LOCATED_IN", "RELATED_TO"},
    ("Organization", "Technology"): {"USES", "DEPENDS_ON", "RELATED_TO"},
    ("Organization", "Project"): {"WORKS_ON", "CREATED", "MANAGES", "RELATED_TO"},
    ("Organization", "Concept"): {"INTERESTED_IN", "RELATED_TO"},
    ("Technology", "Technology"): {"DEPENDS_ON", "RELATED_TO"},
    ("Project", "Technology"): {"USES", "DEPENDS_ON", "RELATED_TO"},
    ("Project", "Project"): {"DEPENDS_ON", "RELATED_TO"},
    ("Project", "Place"): {"LOCATED_IN", "RELATED_TO"},
    ("Project", "Concept"): {"INTERESTED_IN", "RELATED_TO"},
    ("Place", "Place"): {"LOCATED_IN", "RELATED_TO"},
    ("Concept", "Concept"): {"DEPENDS_ON", "RELATED_TO"},
}

DETERMINISTIC_RECLASSIFY: dict[tuple[str, str, str], str] = {
    ("Person", "Place", "WORKS_AT"): "LOCATED_IN",
    ("Person", "Person", "WORKS_AT"): "COLLABORATES_WITH",
    ("Person", "Technology", "WORKS_AT"): "USES",
    ("Person", "Project", "WORKS_AT"): "WORKS_ON",
}

CONFLICTING_SYMMETRIC: list[tuple[str, str]] = [
    ("MENTORS", "MENTORED_BY"),
    ("PARENT_OF", "CHILD_OF"),
    ("REPORTS_TO", "MANAGES"),
]


def _compatible(head_type: str, tail_type: str, rel_type: str) -> bool:
    pair = (head_type, tail_type)
    reverse = (tail_type, head_type)
    allowed = ENTITY_REL_COMPATIBILITY.get(pair) or ENTITY_REL_COMPATIBILITY.get(reverse)
    if allowed is None:
        return rel_type == "RELATED_TO"
    return rel_type in allowed


# ---------------------------------------------------------------------------
# Tier 1: deterministic audit
# ---------------------------------------------------------------------------

def run_deterministic_audit() -> dict:
    checks: list[dict] = []
    flagged: list[dict] = []
    auto_fixes = 0
    quarantined = 0

    # Fetch up to 2000 edges per run, ordered by strength ASC so the weakest
    # (most suspect) edges are audited first.  If the graph exceeds 2000 edges
    # the strongest edges won't be reached; they are also the least likely to
    # have problems.  As edges are deleted/reclassified the window shifts.
    rels = get_relationships_for_audit(limit=2000)
    if len(rels) >= 2000:
        log.warning("Relationship audit hit 2000-edge limit — some edges may be deferred to next run")
    dist = get_relationship_type_distribution()

    # Track edges mutated by earlier checks so later checks skip them
    _mutated: set[tuple[str, str, str]] = set()

    def _rel_key(r: dict) -> tuple[str, str, str]:
        return (r["head"], r["tail"], r.get("rel_type", ""))

    # Check 1: self-referencing relationships
    deleted_self = delete_self_referencing_rels()
    # Mark self-ref edges in the snapshot so later checks skip them (they're
    # already deleted from Neo4j by the bulk delete above).
    if deleted_self > 0:
        for r in rels:
            if r["head"] == r["tail"]:
                _mutated.add(_rel_key(r))
    checks.append({
        "name": "self_refs",
        "status": "warn" if deleted_self > 0 else "pass",
        "detail": f"Deleted {deleted_self} self-referencing relationships",
    })

    # Check 2: zero-strength zombies
    zombie_deleted = 0
    zombie_quarantined = 0
    for r in rels:
        if _rel_key(r) in _mutated:
            continue
        strength = r.get("strength") or 0
        mentions = r.get("mention_count") or 0
        if strength <= 0.01:
            if mentions <= 1:
                if delete_specific_relationship(r["head"], r["tail"], r["rel_type"]):
                    zombie_deleted += 1
                    _mutated.add(_rel_key(r))
            else:
                set_relationship_audit_status(
                    r["head"], r["tail"], r["rel_type"], "quarantined",
                )
                zombie_quarantined += 1
                quarantined += 1
                _mutated.add(_rel_key(r))
    checks.append({
        "name": "zero_strength_zombies",
        "status": "warn" if (zombie_deleted + zombie_quarantined) > 0 else "pass",
        "detail": f"Deleted {zombie_deleted}, quarantined {zombie_quarantined} zero-strength rels",
    })

    # Check 3: entity-type vs rel-type mismatch
    mismatch_fixed = 0
    mismatch_flagged = 0
    for r in rels:
        if _rel_key(r) in _mutated:
            continue
        head_type = r.get("head_type") or ""
        tail_type = r.get("tail_type") or ""
        rel_type = r.get("rel_type") or ""
        if not head_type or not tail_type:
            continue
        if _compatible(head_type, tail_type, rel_type):
            continue

        reclass_key = (head_type, tail_type, rel_type)
        new_type = DETERMINISTIC_RECLASSIFY.get(reclass_key)
        if (new_type and config.REL_AUDIT_AUTO_FIX_ENABLED
                and r.get("audit_status") != "quarantined"):
            try:
                reclassify_relationship(
                    r["head"], r["tail"], rel_type, new_type,
                    r.get("strength") or 0.5,
                    r.get("mention_count") or 1,
                    r.get("context_snippets"),
                    r.get("first_mentioned"),
                )
                _log_auto_fix(r, rel_type, new_type)
                mismatch_fixed += 1
                auto_fixes += 1
                _mutated.add(_rel_key(r))
                continue
            except Exception:
                log.warning("reclassify_relationship failed for %s→%s, quarantining instead",
                            r["head"], r["tail"], exc_info=True)
        # Quarantine: no reclassify rule, auto-fix disabled, already quarantined, or reclassify failed
        already_q = r.get("audit_status") == "quarantined"
        if not already_q:
            set_relationship_audit_status(
                r["head"], r["tail"], rel_type, "quarantined",
            )
            quarantined += 1
        flagged.append({**r, "reason": "type_mismatch"})
        mismatch_flagged += 1
        _mutated.add(_rel_key(r))
    checks.append({
        "name": "type_mismatch",
        "status": "warn" if (mismatch_fixed + mismatch_flagged) > 0 else "pass",
        "detail": f"Auto-fixed {mismatch_fixed}, flagged {mismatch_flagged} type mismatches",
    })

    # Check 4: expanded contradictions
    contradiction_q = 0

    # 4a: multi-WORKS_AT
    works_at: dict[str, list[dict]] = defaultdict(list)
    for r in rels:
        if _rel_key(r) in _mutated:
            continue
        if r.get("rel_type") == "WORKS_AT":
            works_at[r["head"]].append(r)
    for _person, work_rels in works_at.items():
        if len(work_rels) > 1:
            sorted_rels = sorted(work_rels, key=lambda x: x.get("strength") or 0, reverse=True)
            for weak in sorted_rels[1:]:
                if weak.get("audit_status") != "quarantined" and _rel_key(weak) not in _mutated:
                    set_relationship_audit_status(
                        weak["head"], weak["tail"], "WORKS_AT", "quarantined",
                    )
                    flagged.append({**weak, "reason": "multi_works_at"})
                    contradiction_q += 1
                    quarantined += 1
                    _mutated.add(_rel_key(weak))

    # 4b/4c: symmetric conflicts
    for type_a, type_b in CONFLICTING_SYMMETRIC:
        edges_a: dict[tuple[str, str], dict] = {}
        edges_b: dict[tuple[str, str], dict] = {}
        for r in rels:
            if _rel_key(r) in _mutated:
                continue
            rt = r.get("rel_type")
            key = (r["head"], r["tail"])
            if rt == type_a:
                edges_a[key] = r
            elif rt == type_b:
                edges_b[key] = r
        # 4b: same-type bidirectional (check both edges_a and edges_b)
        for edges_map in (edges_a, edges_b):
            for key, ra in edges_map.items():
                reverse_key = (key[1], key[0])
                rb = edges_map.get(reverse_key)
                if rb and _rel_key(rb) not in _mutated:
                    sa, sb = ra.get("strength") or 0, rb.get("strength") or 0
                    if sa == sb:
                        # Deterministic tiebreaker: quarantine lexicographically-later head
                        weaker = rb if rb["head"] >= ra["head"] else ra
                    else:
                        weaker = rb if sb <= sa else ra
                    if weaker.get("audit_status") != "quarantined" and _rel_key(weaker) not in _mutated:
                        set_relationship_audit_status(
                            weaker["head"], weaker["tail"], weaker["rel_type"], "quarantined",
                        )
                        flagged.append({**weaker, "reason": "symmetric_conflict"})
                        contradiction_q += 1
                        quarantined += 1
                        _mutated.add(_rel_key(weaker))
        # 4c: conflicting pair (type_a + type_b on same edge)
        for key, ra in edges_a.items():
            rb = edges_b.get(key)
            if rb and _rel_key(rb) not in _mutated and _rel_key(ra) not in _mutated:
                sa, sb = ra.get("strength") or 0, rb.get("strength") or 0
                if sa == sb:
                    weaker = rb if rb.get("rel_type", "") >= ra.get("rel_type", "") else ra
                else:
                    weaker = rb if sb <= sa else ra
                if weaker.get("audit_status") != "quarantined" and _rel_key(weaker) not in _mutated:
                    set_relationship_audit_status(
                        weaker["head"], weaker["tail"], weaker["rel_type"], "quarantined",
                    )
                    flagged.append({**weaker, "reason": "conflicting_pair"})
                    contradiction_q += 1
                    quarantined += 1
                    _mutated.add(_rel_key(weaker))
    checks.append({
        "name": "contradictions",
        "status": "warn" if contradiction_q > 0 else "pass",
        "detail": f"Quarantined {contradiction_q} contradictory relationships",
    })

    # Check 5: low-confidence single-mention
    low_conf_flagged = 0
    threshold = config.REL_AUDIT_LOW_CONFIDENCE_THRESHOLD
    for r in rels:
        if _rel_key(r) in _mutated:
            continue
        mentions = r.get("mention_count") or 0
        strength = r.get("strength") or 0
        if mentions == 1 and strength < threshold and r.get("audit_status") not in ("quarantined", "verified"):
            flagged.append({**r, "reason": "low_confidence_single"})
            low_conf_flagged += 1
    checks.append({
        "name": "low_confidence_single",
        "status": "warn" if low_conf_flagged > 0 else "pass",
        "detail": f"Flagged {low_conf_flagged} low-confidence single-mention rels for review",
    })

    # Check 6: RELATED_TO accumulation + graph_suggestions enrichment
    warn_threshold = config.REL_AUDIT_RELATED_TO_WARN_THRESHOLD
    related_to_per_entity: dict[str, int] = defaultdict(int)
    for r in rels:
        if _rel_key(r) in _mutated:
            continue
        if r.get("rel_type") == "RELATED_TO":
            related_to_per_entity[r["head"]] += 1
            related_to_per_entity[r["tail"]] += 1
    entities_over = {e: c for e, c in related_to_per_entity.items() if c >= warn_threshold}

    # Pull today's graph_suggestions to enrich flagged RELATED_TO edges.
    # NOTE: Hotspots come from the graph_suggestions table, NOT the rels snapshot.
    # They are intentionally not filtered by _mutated — they represent persistent
    # patterns across days, not current edge state.
    suggestions_enriched = 0
    try:
        from memory.graph_suggestions import get_suggestions, get_related_to_hotspots
    except ImportError:
        get_suggestions = lambda: []
        get_related_to_hotspots = lambda **kw: []
        log.debug("graph_suggestions not available")

    try:
        # Build lookup of original intended types from today's fallbacks
        fallback_hints: dict[tuple[str, str], str] = {}
        for s in get_suggestions():
            if s.get("type") == "relationship_fallback":
                key = (s.get("head", "").strip().lower(), s.get("tail", "").strip().lower())
                fallback_hints[key] = s.get("original_type", "")

        # Flag RELATED_TO hotspots for Tier 2 review (with original_type hints)
        hotspots = get_related_to_hotspots(min_mentions=warn_threshold)
        already_flagged = {
            (f.get("head", "").lower(), f.get("tail", "").lower()) for f in flagged
        }
        for h in hotspots:
            hname = (h.get("head", "").strip().lower(), h.get("tail", "").strip().lower())
            if hname in already_flagged:
                continue
            entry = {
                "head": h.get("head", ""),
                "tail": h.get("tail", ""),
                "head_type": "",
                "tail_type": "",
                "rel_type": "RELATED_TO",
                "strength": 0.5,
                "mention_count": h.get("mentions", 0),
                "context_snippets": h.get("contexts") or [],
                "reason": "related_to_hotspot",
            }
            hint = fallback_hints.get(hname)
            if hint:
                entry["original_type_hint"] = hint
                suggestions_enriched += 1
            flagged.append(entry)
            already_flagged.add(hname)
    except Exception:
        log.debug("graph_suggestions enrichment failed", exc_info=True)

    checks.append({
        "name": "related_to_accumulation",
        "status": "warn" if entities_over or suggestions_enriched > 0 else "pass",
        "detail": f"{len(entities_over)} entities with >= {warn_threshold} RELATED_TO edges"
                  + (f" (top: {list(entities_over.keys())[:3]})" if entities_over else "")
                  + (f", {suggestions_enriched} enriched with suggestion hints" if suggestions_enriched else ""),
    })

    # Check 7: new type monitoring
    missing_types = [t for t in VALID_REL_TYPES if t not in dist and t != "RELATED_TO"]
    checks.append({
        "name": "new_type_monitoring",
        "status": "warn" if missing_types else "pass",
        "detail": f"{len(missing_types)} valid rel types with 0 extractions"
                  + (f": {missing_types[:5]}" if missing_types else ""),
    })

    statuses = {c["status"] for c in checks}
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")

    return {
        "status": overall,
        "summary": f"{auto_fixes} auto-fixed, {quarantined} quarantined, "
                   f"{len(flagged)} flagged for review",
        "checks": checks,
        "flagged": flagged,
        "auto_fixes": auto_fixes,
        "quarantined": quarantined,
        "stats": {"total_rels_scanned": len(rels), "type_distribution": dist},
    }


def _log_auto_fix(rel: dict, old_type: str, new_type: str) -> None:
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        vs.log_correction(
            context=f"{rel['head']} ({rel.get('head_type', '?')}) → "
                    f"{rel['tail']} ({rel.get('tail_type', '?')})",
            molly_output=f"Extracted as {old_type}",
            user_correction=f"Auto-reclassified to {new_type} (entity-type mismatch)",
            pattern="relationship_audit_auto_fix",
        )
    except Exception:
        log.debug("Could not log auto-fix correction", exc_info=True)


# ---------------------------------------------------------------------------
# Tier 2: model-based audit (Kimi K2.5)
# ---------------------------------------------------------------------------

async def run_model_audit(flagged: list[dict]) -> dict:
    if not flagged:
        return {"status": "pass", "verdicts": [], "auto_fixes": 0, "quarantined": 0}

    if not config.MOONSHOT_API_KEY:
        log.warning("MOONSHOT_API_KEY not set — skipping Tier 2 model audit")
        return {"status": "skipped", "verdicts": [], "auto_fixes": 0, "quarantined": 0}

    try:
        import httpx
    except ImportError:
        log.warning("httpx not available — skipping Tier 2 model audit")
        return {"status": "skipped", "verdicts": [], "auto_fixes": 0, "quarantined": 0}

    priority = {"multi_works_at": 0, "symmetric_conflict": 0, "conflicting_pair": 0,
                "type_mismatch": 1, "low_confidence_single": 2}
    batch = sorted(flagged, key=lambda r: priority.get(r.get("reason", ""), 9))
    batch = batch[:config.REL_AUDIT_MAX_MODEL_BATCH]

    rel_descriptions = []
    for i, r in enumerate(batch):
        snippets = r.get("context_snippets") or []
        snippet_text = "; ".join(str(s) for s in snippets[:3]) if snippets else "none"
        hint = r.get("original_type_hint", "")
        hint_line = f"\n   original_intended_type: {hint}" if hint else ""
        rel_descriptions.append(
            f"{i+1}. {r['head']} ({r.get('head_type', '?')}) "
            f"--[{r.get('rel_type', '?')}]--> "
            f"{r['tail']} ({r.get('tail_type', '?')})\n"
            f"   strength={r.get('strength', '?')}, "
            f"mentions={r.get('mention_count', '?')}, "
            f"reason_flagged={r.get('reason', '?')}\n"
            f"   context: {snippet_text}"
            f"{hint_line}"
        )

    valid_types_str = ", ".join(sorted(VALID_REL_TYPES))
    prompt = (
        "Review these flagged knowledge-graph relationships extracted from WhatsApp "
        "conversations. For each, decide whether the relationship type is correct.\n\n"
        f"Valid relationship types: {valid_types_str}\n\n"
        "Relationships to review:\n" + "\n".join(rel_descriptions) + "\n\n"
        "For each numbered relationship, return a JSON array of objects with:\n"
        '- "index": the relationship number\n'
        '- "verdict": "correct" | "reclassify" | "delete"\n'
        '- "suggested_type": (only if verdict is "reclassify", must be from valid types)\n'
        '- "confidence": "high" | "medium" | "low"\n\n'
        "Return ONLY the JSON array, no other text."
    )

    system_prompt = (
        "You audit a personal knowledge graph. Relationships are extracted from "
        "WhatsApp messages between a user and their contacts. Your job is to verify "
        "that relationship types match the entity types and conversational context. "
        "Be conservative: only suggest reclassification when you are confident the "
        "current type is wrong."
    )

    body = {
        "model": config.REL_AUDIT_KIMI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {config.MOONSHOT_API_KEY}",
        "Content-Type": "application/json",
    }

    auto_fixes = 0
    quarantined_count = 0
    verdicts = []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{config.MOONSHOT_BASE_URL}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(c) for c in content)
        content = str(content).strip()

        import json
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            verdict_list = json.loads(content[start:end])
        else:
            log.warning("Kimi response did not contain JSON array: %s", content[:200])
            return {"status": "error", "verdicts": [], "auto_fixes": 0, "quarantined": 0}

        for v in verdict_list:
            idx = v.get("index", 0) - 1
            if idx < 0 or idx >= len(batch):
                continue
            r = batch[idx]
            verdict = v.get("verdict", "").lower()
            confidence = v.get("confidence", "low").lower()
            suggested = v.get("suggested_type", "")
            verdicts.append(v)

            if verdict == "correct" and confidence == "high":
                set_relationship_audit_status(r["head"], r["tail"], r["rel_type"], "verified")
            elif verdict == "reclassify" and confidence == "high" and suggested in VALID_REL_TYPES:
                if config.REL_AUDIT_AUTO_FIX_ENABLED:
                    try:
                        reclassify_relationship(
                            r["head"], r["tail"], r["rel_type"], suggested,
                            r.get("strength") or 0.5,
                            r.get("mention_count") or 1,
                            r.get("context_snippets"),
                            r.get("first_mentioned"),
                        )
                        _log_auto_fix(r, r["rel_type"], suggested)
                        auto_fixes += 1
                        continue
                    except Exception:
                        log.warning("Tier 2 reclassify failed for %s→%s, quarantining",
                                    r["head"], r["tail"], exc_info=True)
                # Fall through to quarantine on failure or auto-fix disabled
                set_relationship_audit_status(r["head"], r["tail"], r["rel_type"], "quarantined")
                quarantined_count += 1
            elif verdict == "delete" and confidence == "high":
                if config.REL_AUDIT_AUTO_FIX_ENABLED:
                    try:
                        if delete_specific_relationship(r["head"], r["tail"], r["rel_type"]):
                            auto_fixes += 1
                            continue
                    except Exception:
                        log.warning("Tier 2 delete failed for %s→%s, quarantining",
                                    r["head"], r["tail"], exc_info=True)
                # Fall through to quarantine on failure or auto-fix disabled
                set_relationship_audit_status(r["head"], r["tail"], r["rel_type"], "quarantined")
                quarantined_count += 1
            else:
                set_relationship_audit_status(r["head"], r["tail"], r["rel_type"], "quarantined")
                quarantined_count += 1

    except Exception:
        log.error("Tier 2 model audit failed", exc_info=True)
        return {"status": "error", "verdicts": verdicts, "auto_fixes": auto_fixes,
                "quarantined": quarantined_count}

    return {"status": "pass", "verdicts": verdicts, "auto_fixes": auto_fixes,
            "quarantined": quarantined_count}


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

async def run_relationship_audit(
    model_enabled: bool = False,
    molly: object | None = None,
) -> dict:
    det_result = run_deterministic_audit()

    model_result = None
    if model_enabled and det_result.get("flagged"):
        model_result = await run_model_audit(det_result["flagged"])

    total_auto_fixes = det_result.get("auto_fixes", 0)
    total_quarantined = det_result.get("quarantined", 0)
    if model_result:
        total_auto_fixes += model_result.get("auto_fixes", 0)
        total_quarantined += model_result.get("quarantined", 0)

    det_status = det_result.get("status", "pass")
    model_status = (model_result or {}).get("status", "skipped")

    return {
        "status": det_status,
        "summary": f"{total_auto_fixes} auto-fixed, {total_quarantined} quarantined"
                   + (f" (model: {model_status})" if model_result else ""),
        "deterministic_result": det_result,
        "model_result": model_result,
        "auto_fixes_applied": total_auto_fixes,
        "quarantined_count": total_quarantined,
        "stats": det_result.get("stats", {}),
    }
