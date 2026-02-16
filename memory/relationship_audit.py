"""Nightly relationship quality audit (deterministic + model tiers)."""

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

ENTITY_REL_COMPATIBILITY: dict[tuple[str, str], set[str]] = {
    ("Person", "Person"): {
        "KNOWS", "CLASSMATE_OF", "MENTORS", "MENTORED_BY", "REPORTS_TO", "COLLABORATES_WITH", "DISCUSSED_WITH", "CONTACT_OF", "PARENT_OF", "CHILD_OF",
    },
    ("Person", "Organization"): {"WORKS_AT", "STUDIED_AT", "ALUMNI_OF", "CUSTOMER_OF", "ATTENDS", "RECEIVED_FROM"},
    ("Person", "Technology"): {"USES", "CREATED", "INTERESTED_IN", "WORKS_ON"},
    ("Person", "Project"): {"WORKS_ON", "CREATED", "MANAGES", "INTERESTED_IN"},
    ("Person", "Place"): {"LOCATED_IN"},
    ("Person", "Concept"): {"INTERESTED_IN", "DISCUSSED_WITH"},
    ("Organization", "Organization"): {"COLLABORATES_WITH", "CUSTOMER_OF", "DEPENDS_ON"},
    ("Organization", "Place"): {"LOCATED_IN"},
    ("Organization", "Technology"): {"USES", "DEPENDS_ON"},
    ("Organization", "Project"): {"WORKS_ON", "CREATED", "MANAGES"},
    ("Organization", "Concept"): {"INTERESTED_IN"},
    ("Technology", "Technology"): {"DEPENDS_ON"},
    ("Project", "Technology"): {"USES", "DEPENDS_ON"},
    ("Project", "Project"): {"DEPENDS_ON"},
    ("Project", "Place"): {"LOCATED_IN"},
    ("Project", "Concept"): {"INTERESTED_IN"},
    ("Place", "Place"): {"LOCATED_IN"},
    ("Concept", "Concept"): {"DEPENDS_ON"},
}
for _allowed in ENTITY_REL_COMPATIBILITY.values():
    _allowed.add("RELATED_TO")

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
    if rel_type == "RELATED_TO":
        return True
    pair = (head_type, tail_type)
    reverse = (tail_type, head_type)
    allowed = ENTITY_REL_COMPATIBILITY.get(pair) or ENTITY_REL_COMPATIBILITY.get(reverse)
    return rel_type in allowed if allowed is not None else False


def _rel_key(r: dict) -> tuple[str, str, str]:
    return (r["head"], r["tail"], r.get("rel_type", ""))


def _append_check(checks: list[dict], name: str, detail: str, warn_if: bool = False) -> None:
    checks.append({"name": name, "status": "warn" if warn_if else "pass", "detail": detail})


def _quarantine(r: dict, rel_type: str | None = None) -> bool:
    if r.get("audit_status") == "quarantined":
        return False
    set_relationship_audit_status(r["head"], r["tail"], rel_type or r["rel_type"], "quarantined")
    return True


def _quarantine_weaker(a: dict, b: dict, *, by_rel_type: bool = False) -> dict:
    sa, sb = a.get("strength") or 0, b.get("strength") or 0
    if sa == sb:
        if by_rel_type:
            return b if b.get("rel_type", "") >= a.get("rel_type", "") else a
        return b if b["head"] >= a["head"] else a
    return b if sb <= sa else a


def run_deterministic_audit() -> dict:
    checks: list[dict] = []
    flagged: list[dict] = []
    auto_fixes = 0
    quarantined = 0

    rels = get_relationships_for_audit(limit=2000)
    if len(rels) >= 2000:
        log.warning("Relationship audit hit 2000-edge limit — some edges may be deferred to next run")
    dist = get_relationship_type_distribution()

    mutated: set[tuple[str, str, str]] = set()

    deleted_self = delete_self_referencing_rels()
    if deleted_self > 0:
        for r in rels:
            if r["head"] == r["tail"]:
                mutated.add(_rel_key(r))
    _append_check(checks, "self_refs", f"Deleted {deleted_self} self-referencing relationships", warn_if=deleted_self > 0)

    zombie_deleted = 0
    zombie_quarantined = 0
    for r in rels:
        if _rel_key(r) in mutated:
            continue
        if (r.get("strength") or 0) > 0.01:
            continue
        if (r.get("mention_count") or 0) <= 1:
            if delete_specific_relationship(r["head"], r["tail"], r["rel_type"]):
                zombie_deleted += 1
                mutated.add(_rel_key(r))
            continue
        if _quarantine(r):
            zombie_quarantined += 1
            quarantined += 1
            mutated.add(_rel_key(r))
    _append_check(
        checks,
        "zero_strength_zombies",
        f"Deleted {zombie_deleted}, quarantined {zombie_quarantined} zero-strength rels",
        warn_if=(zombie_deleted + zombie_quarantined) > 0,
    )

    mismatch_fixed = 0
    mismatch_flagged = 0
    for r in rels:
        if _rel_key(r) in mutated:
            continue
        head_type = r.get("head_type") or ""
        tail_type = r.get("tail_type") or ""
        rel_type = r.get("rel_type") or ""
        if not head_type or not tail_type or _compatible(head_type, tail_type, rel_type):
            continue

        new_type = DETERMINISTIC_RECLASSIFY.get((head_type, tail_type, rel_type))
        if new_type and config.REL_AUDIT_AUTO_FIX_ENABLED and r.get("audit_status") != "quarantined":
            try:
                reclassify_relationship(
                    r["head"],
                    r["tail"],
                    rel_type,
                    new_type,
                    r.get("strength") or 0.5,
                    r.get("mention_count") or 1,
                    r.get("context_snippets"),
                    r.get("first_mentioned"),
                )
                _log_auto_fix(r, rel_type, new_type)
                mismatch_fixed += 1
                auto_fixes += 1
                mutated.add(_rel_key(r))
                continue
            except Exception:
                log.warning("reclassify_relationship failed for %s→%s, quarantining instead", r["head"], r["tail"], exc_info=True)

        if _quarantine(r, rel_type):
            quarantined += 1
        flagged.append({**r, "reason": "type_mismatch"})
        mismatch_flagged += 1
        mutated.add(_rel_key(r))
    _append_check(
        checks,
        "type_mismatch",
        f"Auto-fixed {mismatch_fixed}, flagged {mismatch_flagged} type mismatches",
        warn_if=(mismatch_fixed + mismatch_flagged) > 0,
    )

    contradiction_q = 0
    works_at: dict[str, list[dict]] = defaultdict(list)
    for r in rels:
        if _rel_key(r) not in mutated and r.get("rel_type") == "WORKS_AT":
            works_at[r["head"]].append(r)
    for work_rels in works_at.values():
        if len(work_rels) <= 1:
            continue
        for weak in sorted(work_rels, key=lambda x: x.get("strength") or 0, reverse=True)[1:]:
            if _rel_key(weak) in mutated:
                continue
            if _quarantine(weak, "WORKS_AT"):
                flagged.append({**weak, "reason": "multi_works_at"})
                contradiction_q += 1
                quarantined += 1
                mutated.add(_rel_key(weak))

    for type_a, type_b in CONFLICTING_SYMMETRIC:
        edges_a: dict[tuple[str, str], dict] = {}
        edges_b: dict[tuple[str, str], dict] = {}
        for r in rels:
            if _rel_key(r) in mutated:
                continue
            key = (r["head"], r["tail"])
            rt = r.get("rel_type")
            if rt == type_a:
                edges_a[key] = r
            elif rt == type_b:
                edges_b[key] = r

        for edges_map in (edges_a, edges_b):
            for key, ra in edges_map.items():
                rb = edges_map.get((key[1], key[0]))
                if not rb or _rel_key(rb) in mutated:
                    continue
                weaker = _quarantine_weaker(ra, rb)
                if _rel_key(weaker) in mutated:
                    continue
                if _quarantine(weaker):
                    flagged.append({**weaker, "reason": "symmetric_conflict"})
                    contradiction_q += 1
                    quarantined += 1
                    mutated.add(_rel_key(weaker))

        for key, ra in edges_a.items():
            rb = edges_b.get(key)
            if not rb or _rel_key(ra) in mutated or _rel_key(rb) in mutated:
                continue
            weaker = _quarantine_weaker(ra, rb, by_rel_type=True)
            if _rel_key(weaker) in mutated:
                continue
            if _quarantine(weaker):
                flagged.append({**weaker, "reason": "conflicting_pair"})
                contradiction_q += 1
                quarantined += 1
                mutated.add(_rel_key(weaker))

    _append_check(checks, "contradictions", f"Quarantined {contradiction_q} contradictory relationships", warn_if=contradiction_q > 0)

    low_conf_flagged = 0
    threshold = config.REL_AUDIT_LOW_CONFIDENCE_THRESHOLD
    for r in rels:
        if _rel_key(r) in mutated:
            continue
        mentions = r.get("mention_count") or 0
        strength = r.get("strength") or 0
        if mentions == 1 and strength < threshold and r.get("audit_status") not in {"quarantined", "verified"}:
            flagged.append({**r, "reason": "low_confidence_single"})
            low_conf_flagged += 1
    _append_check(
        checks,
        "low_confidence_single",
        f"Flagged {low_conf_flagged} low-confidence single-mention rels for review",
        warn_if=low_conf_flagged > 0,
    )

    warn_threshold = config.REL_AUDIT_RELATED_TO_WARN_THRESHOLD
    related_to_per_entity: dict[str, int] = defaultdict(int)
    for r in rels:
        if _rel_key(r) not in mutated and r.get("rel_type") == "RELATED_TO":
            related_to_per_entity[r["head"]] += 1
            related_to_per_entity[r["tail"]] += 1
    entities_over = {e: c for e, c in related_to_per_entity.items() if c >= warn_threshold}

    suggestions_enriched = 0
    try:
        from memory.graph_suggestions import get_related_to_hotspots, get_suggestions
    except ImportError:
        get_suggestions = lambda: []
        get_related_to_hotspots = lambda **kw: []
        log.debug("graph_suggestions not available")

    try:
        fallback_hints: dict[tuple[str, str], str] = {}
        for s in get_suggestions():
            if s.get("type") == "relationship_fallback":
                fallback_hints[(s.get("head", "").strip().lower(), s.get("tail", "").strip().lower())] = s.get("original_type", "")

        already_flagged = {(f.get("head", "").lower(), f.get("tail", "").lower()) for f in flagged}
        for h in get_related_to_hotspots(min_mentions=warn_threshold):
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

    detail = f"{len(entities_over)} entities with >= {warn_threshold} RELATED_TO edges"
    if entities_over:
        detail += f" (top: {list(entities_over.keys())[:3]})"
    if suggestions_enriched:
        detail += f", {suggestions_enriched} enriched with suggestion hints"
    _append_check(checks, "related_to_accumulation", detail, warn_if=bool(entities_over or suggestions_enriched > 0))

    missing_types = [t for t in VALID_REL_TYPES if t not in dist and t != "RELATED_TO"]
    _append_check(
        checks,
        "new_type_monitoring",
        f"{len(missing_types)} valid rel types with 0 extractions" + (f": {missing_types[:5]}" if missing_types else ""),
        warn_if=bool(missing_types),
    )

    statuses = {c["status"] for c in checks}
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")

    return {
        "status": overall,
        "summary": f"{auto_fixes} auto-fixed, {quarantined} quarantined, {len(flagged)} flagged for review",
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
            context=f"{rel['head']} ({rel.get('head_type', '?')}) → {rel['tail']} ({rel.get('tail_type', '?')})",
            molly_output=f"Extracted as {old_type}",
            user_correction=f"Auto-reclassified to {new_type} (entity-type mismatch)",
            pattern="relationship_audit_auto_fix",
        )
    except Exception:
        log.debug("Could not log auto-fix correction", exc_info=True)


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

    priority = {"multi_works_at": 0, "symmetric_conflict": 0, "conflicting_pair": 0, "type_mismatch": 1, "low_confidence_single": 2}
    batch = sorted(flagged, key=lambda r: priority.get(r.get("reason", ""), 9))[: config.REL_AUDIT_MAX_MODEL_BATCH]

    rel_descriptions = []
    for i, r in enumerate(batch, start=1):
        snippets = r.get("context_snippets") or []
        hint = r.get("original_type_hint", "")
        rel_descriptions.append(
            f"{i}. {r['head']} ({r.get('head_type', '?')}) --[{r.get('rel_type', '?')}]--> {r['tail']} ({r.get('tail_type', '?')})\n"
            f"   strength={r.get('strength', '?')}, mentions={r.get('mention_count', '?')}, reason_flagged={r.get('reason', '?')}\n"
            f"   context: {'; '.join(str(s) for s in snippets[:3]) if snippets else 'none'}"
            f"{f'\\n   original_intended_type: {hint}' if hint else ''}"
        )

    prompt = (
        "Review these flagged knowledge-graph relationships extracted from WhatsApp conversations.\n\n"
        f"Valid relationship types: {', '.join(sorted(VALID_REL_TYPES))}\n\n"
        "Relationships to review:\n"
        + "\n".join(rel_descriptions)
        + "\n\nFor each numbered relationship, return a JSON array of objects with:\n"
        '- "index": relationship number\n'
        '- "verdict": "correct" | "reclassify" | "delete"\n'
        '- "suggested_type": only for "reclassify"\n'
        '- "confidence": "high" | "medium" | "low"\n\n'
        "Return ONLY the JSON array."
    )

    body = {
        "model": config.REL_AUDIT_KIMI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You audit a personal knowledge graph extracted from WhatsApp messages. "
                    "Be conservative and only reclassify when highly confident."
                ),
            },
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
        import asyncio as _asyncio

        data = None
        last_exc = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(f"{config.MOONSHOT_BASE_URL}/chat/completions", headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as exc:
                last_exc = exc
                wait = 5 * (2**attempt)
                log.warning("Kimi Tier 2 timeout (attempt %d/3), retrying in %ds: %s", attempt + 1, wait, exc)
                await _asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 502, 503):
                    last_exc = exc
                    wait = 10 * (2**attempt)
                    log.warning("Kimi Tier 2 HTTP %d (attempt %d/3), retrying in %ds", exc.response.status_code, attempt + 1, wait)
                    await _asyncio.sleep(wait)
                else:
                    raise
        else:
            raise last_exc or RuntimeError("Kimi Tier 2 failed after 3 attempts")

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(c) for c in content)
        content = str(content).strip()

        import json

        start = content.find("[")
        end = content.rfind("]") + 1
        if start < 0 or end <= start:
            log.warning("Kimi response did not contain JSON array: %s", content[:200])
            return {"status": "error", "verdicts": [], "auto_fixes": 0, "quarantined": 0}

        verdict_list = json.loads(content[start:end])
        handlers = {
            "correct": _apply_correct_verdict,
            "reclassify": _apply_reclassify_verdict,
            "delete": _apply_delete_verdict,
        }

        for v in verdict_list:
            idx = v.get("index", 0) - 1
            if idx < 0 or idx >= len(batch):
                continue
            r = batch[idx]
            verdict = str(v.get("verdict", "")).lower()
            confidence = str(v.get("confidence", "low")).lower()
            suggested = v.get("suggested_type", "")
            verdicts.append(v)

            handler = handlers.get(verdict)
            if not handler:
                if _quarantine(r):
                    quarantined_count += 1
                continue
            changed, quarantined_delta = handler(r, confidence, suggested)
            auto_fixes += changed
            quarantined_count += quarantined_delta

    except Exception:
        log.error("Tier 2 model audit failed", exc_info=True)
        return {"status": "error", "verdicts": verdicts, "auto_fixes": auto_fixes, "quarantined": quarantined_count}

    return {"status": "pass", "verdicts": verdicts, "auto_fixes": auto_fixes, "quarantined": quarantined_count}


def _apply_correct_verdict(r: dict, confidence: str, suggested: str) -> tuple[int, int]:
    if confidence == "high":
        set_relationship_audit_status(r["head"], r["tail"], r["rel_type"], "verified")
        return 0, 0
    return (0, 1) if _quarantine(r) else (0, 0)


def _apply_reclassify_verdict(r: dict, confidence: str, suggested: str) -> tuple[int, int]:
    if confidence != "high" or suggested not in VALID_REL_TYPES:
        return (0, 1) if _quarantine(r) else (0, 0)
    if config.REL_AUDIT_AUTO_FIX_ENABLED:
        try:
            reclassify_relationship(
                r["head"],
                r["tail"],
                r["rel_type"],
                suggested,
                r.get("strength") or 0.5,
                r.get("mention_count") or 1,
                r.get("context_snippets"),
                r.get("first_mentioned"),
            )
            _log_auto_fix(r, r["rel_type"], suggested)
            return 1, 0
        except Exception:
            log.warning("Tier 2 reclassify failed for %s→%s, quarantining", r["head"], r["tail"], exc_info=True)
    return (0, 1) if _quarantine(r) else (0, 0)


def _apply_delete_verdict(r: dict, confidence: str, suggested: str) -> tuple[int, int]:
    if confidence == "high" and config.REL_AUDIT_AUTO_FIX_ENABLED:
        try:
            if delete_specific_relationship(r["head"], r["tail"], r["rel_type"]):
                return 1, 0
        except Exception:
            log.warning("Tier 2 delete failed for %s→%s, quarantining", r["head"], r["tail"], exc_info=True)
    return (0, 1) if _quarantine(r) else (0, 0)


async def run_relationship_audit(model_enabled: bool = False, molly: object | None = None) -> dict:
    det_result = run_deterministic_audit()
    model_result = await run_model_audit(det_result["flagged"]) if model_enabled and det_result.get("flagged") else None

    total_auto_fixes = det_result.get("auto_fixes", 0) + ((model_result or {}).get("auto_fixes", 0))
    total_quarantined = det_result.get("quarantined", 0) + ((model_result or {}).get("quarantined", 0))
    model_status = (model_result or {}).get("status", "skipped")

    return {
        "status": det_result.get("status", "pass"),
        "summary": f"{total_auto_fixes} auto-fixed, {total_quarantined} quarantined" + (f" (model: {model_status})" if model_result else ""),
        "deterministic_result": det_result,
        "model_result": model_result,
        "auto_fixes_applied": total_auto_fixes,
        "quarantined_count": total_quarantined,
        "stats": det_result.get("stats", {}),
    }
