"""Skill lifecycle service — proposal, approval, iteration, and rejection cooldown.

Extracted from ``evolution/skills.py`` (SelfImprovementEngine) to keep the
engine class focused on orchestration while this service owns the full skill
lifecycle: owner phrase triggers, proposal creation, multi-round approval,
edit parsing, rejection cooldown tracking, and skill reloading.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil

from utils import atomic_write
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config
import db_pool
from evolution.infra import (
    _SKILL_REJECTION_COOLDOWN_DAYS,
    _OWNER_SKILL_PHRASE_RE,
)

log = logging.getLogger(__name__)


def _extract_rejection_reason(decision: Any) -> str:
    """Extract rejection reason from a deny decision tuple."""
    if isinstance(decision, tuple) and decision:
        kind = str(decision[0]).strip().lower()
        if kind == "deny":
            return str(decision[1] if len(decision) > 1 else "").strip()
    return ""



class SkillLifecycleService:
    """Skill lifecycle service — propose, approve, iterate, reject, reload.

    Uses ``pattern_helpers`` for step extraction (breaks cycle with SkillGaps).
    """

    def __init__(self, ctx, infra, comms, auto_patterns):
        from evolution.context import EngineContext
        from evolution.infra import InfraService
        from evolution.owner_comms import OwnerCommsService
        from evolution.automation_patterns import AutomationPatternsService
        self.ctx: EngineContext = ctx
        self.infra: InfraService = infra
        self.comms: OwnerCommsService = comms
        self.auto_patterns: AutomationPatternsService = auto_patterns

    def should_trigger_owner_skill_phrase(self, text: str) -> bool:
        return bool(_OWNER_SKILL_PHRASE_RE.search(str(text or "")))

    async def propose_skill_from_owner_phrase(self, owner_text: str = "") -> dict[str, Any]:
        patterns = self.auto_patterns.detect_workflow_patterns(days=30, min_occurrences=2)
        if not patterns:
            await self.comms.notify_owner(
                "No repeated workflow pattern detected yet. Run the workflow a few times, then ask again."
            )
            return {"status": "skipped", "reason": "no patterns"}
        return await self.propose_skill_from_patterns(patterns, source="owner_phrase", source_text=owner_text)

    async def propose_skill_from_patterns(
        self,
        workflow_patterns: list[dict[str, Any]],
        source: str = "pattern",
        source_text: str = "",
    ) -> dict[str, Any]:
        from evolution.pattern_helpers import pattern_steps as _pattern_steps

        if not workflow_patterns:
            return {"status": "skipped", "reason": "no patterns"}

        best = sorted(workflow_patterns, key=lambda p: p.get("count", 0), reverse=True)[0]
        tools = _pattern_steps(best)
        if not tools:
            return {"status": "skipped", "reason": "pattern has no steps"}

        skill_name = f"{str(best.get('name', 'workflow')).strip()} skill".strip()
        trigger_text = str(best.get("trigger", "When this multi-step workflow repeats.")).strip()
        steps = [f"Run `{tool}` and capture output." for tool in tools]
        return await self.propose_skill_lifecycle(
            {
                "name": skill_name,
                "trigger": [trigger_text],
                "tools": tools,
                "steps": steps,
                "guardrails": [
                    "Keep outputs concise and factual.",
                    "Ask for confirmation before write/send actions.",
                ],
                "source": source,
                "source_text": source_text,
                "metadata": {"pattern": best},
            }
        )

    async def propose_skill_lifecycle(self, proposal: dict[str, Any]) -> dict[str, Any]:
        name = str(proposal.get("name", "workflow skill")).strip() or "workflow skill"
        slug = self.infra.slug(name) or "workflow-skill"
        skills_dir = config.SKILLS_DIR
        skills_dir.mkdir(parents=True, exist_ok=True)
        live_path = skills_dir / f"{slug}.md"
        pending_new_path = skills_dir / f"{slug}.md.pending"
        pending_edit_path = skills_dir / f"{slug}.md.pending-edit"
        source = str(proposal.get("source", "manual")).strip() or "manual"
        source_text = str(proposal.get("source_text", "") or "").strip()
        metadata = proposal.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        if pending_new_path.exists() or pending_edit_path.exists():
            return {"status": "skipped", "reason": "pending skill lifecycle already exists"}

        current_markdown = live_path.read_text() if live_path.exists() else ""
        current_sections = self._extract_skill_sections(current_markdown) if current_markdown else {}

        trigger_values = self._normalize_skill_section_items(
            proposal.get("trigger", current_sections.get("trigger", "")), section="trigger",
        )
        tools_values = self._normalize_skill_section_items(
            proposal.get("tools", proposal.get("required_tools", current_sections.get("tools", ""))), section="tools",
        )
        steps_values = self._normalize_skill_section_items(
            proposal.get("steps", current_sections.get("steps", "")), section="steps",
        )
        guardrail_values = self._normalize_skill_section_items(
            proposal.get("guardrails", current_sections.get("guardrails", "")), section="guardrails",
        )

        if not trigger_values:
            trigger_values = ["When this workflow repeats."]
        if not guardrail_values:
            guardrail_values = [
                "Keep outputs concise and factual.",
                "Ask for confirmation before write/send actions.",
            ]
        if not steps_values:
            return {"status": "skipped", "reason": "skill has no steps"}

        draft_payload = {
            "name": name,
            "trigger": trigger_values,
            "tools": tools_values,
            "steps": steps_values,
            "guardrails": guardrail_values,
        }
        draft_markdown = self.render_skill_markdown(draft_payload)
        fingerprint = self._skill_rejection_fingerprint(draft_payload)
        if self._is_skill_rejection_cooldown_active(fingerprint, days=_SKILL_REJECTION_COOLDOWN_DAYS):
            self.comms.log_improvement_event(
                event_type="lifecycle",
                category="skill",
                title=name,
                payload=json.dumps(
                    {"source": source, "fingerprint": fingerprint, "cooldown_days": _SKILL_REJECTION_COOLDOWN_DAYS},
                    ensure_ascii=True,
                ),
                status="cooldown_active",
            )
            return {"status": "skipped", "reason": "rejection cooldown active"}

        existing_signature = self._skill_signature_from_markdown(current_markdown) if current_markdown else None
        draft_signature = self._skill_signature_from_payload(draft_payload)
        if existing_signature is not None and existing_signature == draft_signature:
            return {"status": "skipped", "reason": "no skill changes detected"}

        return await self._run_skill_lifecycle_approval(
            live_path=live_path,
            pending_new_path=pending_new_path,
            pending_edit_path=pending_edit_path,
            draft_payload=draft_payload,
            draft_markdown=draft_markdown,
            source=source,
            source_text=source_text,
            metadata=metadata,
            fingerprint=fingerprint,
            current_markdown=current_markdown,
        )

    async def _run_skill_lifecycle_approval(
        self,
        live_path: Path,
        pending_new_path: Path,
        pending_edit_path: Path,
        draft_payload: dict[str, Any],
        draft_markdown: str,
        source: str,
        source_text: str,
        metadata: dict[str, Any],
        fingerprint: str,
        current_markdown: str,
    ) -> dict[str, Any]:
        name = str(draft_payload["name"]).strip()
        is_iteration = bool(current_markdown and live_path.exists())
        pending_path = pending_edit_path if is_iteration else pending_new_path
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        if is_iteration:
            shutil.copy2(live_path, pending_path)
        atomic_write(pending_path, draft_markdown)

        proposal_payload = {
            "pending_path": str(pending_path),
            "live_path": str(live_path),
            "source": source,
            "source_text": source_text[:500],
            "fingerprint": fingerprint,
            "iteration": is_iteration,
            "metadata": metadata,
        }
        proposal_event_id = self.comms.log_improvement_event(
            event_type="proposal",
            category="skill",
            title=name,
            payload=json.dumps(proposal_payload, ensure_ascii=True),
            status="proposed",
        )

        if is_iteration:
            section_diff = self._format_skill_section_diff(current_markdown, draft_markdown)
            summary = self._format_skill_iteration_summary(
                name=name, pending_path=pending_path, section_diff=section_diff,
            )
            decision = await self.comms.request_owner_decision(
                category="self-improve-skill-edit",
                description=summary,
                required_keyword="YES",
                allow_edit=False,
                include_rejection_reason=True,
            )
            if decision is True:
                os.replace(pending_path, live_path)
                self._activate_skill_reload()
                self._update_improvement_event_status(proposal_event_id, "completed")
                self.comms.log_improvement_event(
                    event_type="lifecycle", category="skill", title=name,
                    payload=json.dumps({"path": str(live_path), "source": source, "fingerprint": fingerprint}, ensure_ascii=True),
                    status="approved",
                )
                self.comms.log_improvement_event(
                    event_type="lifecycle", category="skill", title=name,
                    payload=json.dumps({"path": str(live_path), "source": source, "fingerprint": fingerprint}, ensure_ascii=True),
                    status="activated",
                )
                return {"status": "approved", "path": str(live_path), "mode": "pending-edit"}

            explicit_owner_reject = isinstance(decision, tuple) and str(decision[0]).strip().lower() == "deny"
            reason = _extract_rejection_reason(decision)
            pending_path.unlink(missing_ok=True)
            self._update_improvement_event_status(proposal_event_id, "rejected")
            self.comms.log_improvement_event(
                event_type="lifecycle", category="skill", title=name,
                payload=json.dumps({"path": str(live_path), "source": source, "fingerprint": fingerprint, "reason": reason}, ensure_ascii=True),
                status="rejected",
            )
            if explicit_owner_reject:
                self._record_skill_rejection_cooldown(
                    fingerprint=fingerprint, title=name, reason=reason, source=source, payload=draft_payload,
                )
            return {"status": "rejected", "reason": reason or "owner rejected", "mode": "pending-edit"}

        # New skill: iterate until YES/NO, with structured EDIT support.
        current_payload = dict(draft_payload)
        current_fingerprint = fingerprint
        for _ in range(8):
            atomic_write(pending_path, self.render_skill_markdown(current_payload))
            simulation = self.infra.dry_run_skill(pending_path.read_text())
            summary = self._format_skill_draft_summary(
                name=name,
                pending_path=pending_path,
                trigger_values=current_payload["trigger"],
                tools_values=current_payload["tools"],
                steps_values=current_payload["steps"],
                dry_run=simulation,
            )
            decision = await self.comms.request_owner_decision(
                category="self-improve-skill",
                description=summary,
                required_keyword="YES",
                allow_edit=True,
                include_rejection_reason=True,
            )

            if decision is True:
                os.replace(pending_path, live_path)
                self._activate_skill_reload()
                self._update_improvement_event_status(proposal_event_id, "completed")
                self.comms.log_improvement_event(
                    event_type="lifecycle", category="skill", title=name,
                    payload=json.dumps({"path": str(live_path), "source": source, "fingerprint": current_fingerprint}, ensure_ascii=True),
                    status="approved",
                )
                self.comms.log_improvement_event(
                    event_type="lifecycle", category="skill", title=name,
                    payload=json.dumps({"path": str(live_path), "source": source, "fingerprint": current_fingerprint}, ensure_ascii=True),
                    status="activated",
                )
                return {"status": "approved", "path": str(live_path), "mode": "pending"}

            if isinstance(decision, str):
                edits = self._parse_structured_skill_edits(decision)
                if not edits:
                    await self.comms.notify_owner(
                        "Unsupported edit format. Use: EDIT trigger|tools|steps|guardrails: ..."
                    )
                    continue
                current_payload = self._apply_structured_skill_edits(current_payload, edits)
                current_fingerprint = self._skill_rejection_fingerprint(current_payload)
                self.comms.log_improvement_event(
                    event_type="lifecycle", category="skill", title=name,
                    payload=json.dumps(
                        {"pending_path": str(pending_path), "source": source, "fingerprint": current_fingerprint, "edits": edits},
                        ensure_ascii=True,
                    ),
                    status="edited",
                )
                continue

            explicit_owner_reject = isinstance(decision, tuple) and str(decision[0]).strip().lower() == "deny"
            reason = _extract_rejection_reason(decision)
            pending_path.unlink(missing_ok=True)
            self._update_improvement_event_status(proposal_event_id, "rejected")
            self.comms.log_improvement_event(
                event_type="lifecycle", category="skill", title=name,
                payload=json.dumps(
                    {"path": str(live_path), "source": source, "fingerprint": current_fingerprint, "reason": reason},
                    ensure_ascii=True,
                ),
                status="rejected",
            )
            if explicit_owner_reject:
                self._record_skill_rejection_cooldown(
                    fingerprint=current_fingerprint, title=name, reason=reason, source=source, payload=current_payload,
                )
            return {"status": "rejected", "reason": reason or "owner rejected", "mode": "pending"}

        pending_path.unlink(missing_ok=True)
        self._update_improvement_event_status(proposal_event_id, "abandoned")
        return {"status": "failed", "reason": "edit loop exceeded limit"}

    # --- Formatting / parsing helpers (no dependencies, reused as-is) ---

    def _format_skill_draft_summary(self, name, pending_path, trigger_values, tools_values, steps_values, dry_run):
        return (
            f"📝 Skill draft pending approval: \"{name}\"\n\n"
            f"Pending file: {pending_path}\n"
            f"Triggers: {self._summarize_values(trigger_values)}\n"
            f"Tools: {self._summarize_values(tools_values)}\n"
            f"Steps: {self._summarize_values(steps_values)}\n"
            f"Dry-run: {'PASS' if dry_run.get('ok') else 'FAIL'} ({dry_run.get('reason', '')})\n\n"
            "Reply YES to activate, NO to reject, or EDIT trigger/tools/steps/guardrails: ..."
        )

    def _format_skill_iteration_summary(self, name, pending_path, section_diff):
        return (
            f"✏️ Skill update pending approval: \"{name}\"\n\n"
            f"Pending edit file: {pending_path}\n"
            f"{section_diff}\n\n"
            "Reply YES to apply update, or NO to discard."
        )

    def _summarize_values(self, values, limit=4):
        compact = [str(v).strip() for v in values if str(v).strip()]
        if not compact:
            return "(none)"
        if len(compact) <= limit:
            return " | ".join(compact)
        remaining = len(compact) - limit
        return f"{' | '.join(compact[:limit])} | ... (+{remaining} more)"

    def _extract_skill_sections(self, markdown):
        if not markdown.strip():
            return {"trigger": "", "tools": "", "steps": "", "guardrails": ""}

        def _section(heading):
            pattern = re.compile(
                rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)",
                re.MULTILINE | re.DOTALL,
            )
            match = pattern.search(markdown)
            return match.group(1).strip() if match else ""

        return {
            "trigger": _section("Trigger"),
            "tools": _section("Required Tools"),
            "steps": _section("Steps"),
            "guardrails": _section("Guardrails"),
        }

    def _normalize_skill_section_items(self, raw, section):
        if isinstance(raw, list):
            chunks = [str(item) for item in raw]
        elif isinstance(raw, tuple):
            chunks = [str(item) for item in raw]
        else:
            chunks = [str(raw or "")]

        values: list[str] = []
        for chunk in chunks:
            text = str(chunk or "").strip()
            if not text:
                continue
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            if section == "steps" and "->" in text and "\n" not in text:
                candidates = [part.strip() for part in text.split("->")]
            else:
                candidates = re.split(r"[\n;]+", text)
            for candidate in candidates:
                clean = candidate.strip()
                if not clean:
                    continue
                clean = re.sub(r"^\s*[-*]\s+", "", clean)
                clean = re.sub(r"^\s*\d+\.\s+", "", clean)
                clean = clean.strip()
                if section == "tools":
                    parts = [part.strip("` ").strip() for part in clean.split(",")]
                elif section == "trigger":
                    parts = [clean.strip('" ')]
                else:
                    parts = [clean]
                for item in parts:
                    normalized = re.sub(r"\s+", " ", str(item or "").strip())
                    if normalized:
                        values.append(normalized)

        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped

    def render_skill_markdown(self, payload):
        name = str(payload.get("name", "workflow skill")).strip() or "workflow skill"
        triggers = self._normalize_skill_section_items(payload.get("trigger", ""), "trigger")
        tools = self._normalize_skill_section_items(payload.get("tools", ""), "tools")
        steps = self._normalize_skill_section_items(payload.get("steps", ""), "steps")
        guardrails = self._normalize_skill_section_items(payload.get("guardrails", ""), "guardrails")

        if not triggers:
            triggers = ["When this workflow repeats."]
        if not guardrails:
            guardrails = [
                "Keep outputs concise and factual.",
                "Ask for confirmation before write/send actions.",
            ]

        lines = [f"# {name}", "", "## Trigger"]
        lines.extend(f"- {item}" for item in triggers)
        lines.extend(["", "## Required Tools"])
        lines.extend(f"- `{item}`" for item in tools)
        lines.extend(["", "## Steps"])
        lines.extend(f"{idx}. {item}" for idx, item in enumerate(steps, start=1))
        lines.extend(["", "## Guardrails"])
        lines.extend(f"- {item}" for item in guardrails)
        return "\n".join(lines).strip() + "\n"

    def _format_skill_section_diff(self, current_markdown, draft_markdown):
        current_sections = self._extract_skill_sections(current_markdown)
        draft_sections = self._extract_skill_sections(draft_markdown)
        sections = [
            ("Trigger", "trigger"), ("Required Tools", "tools"),
            ("Steps", "steps"), ("Guardrails", "guardrails"),
        ]
        lines = ["Section-level diff summary:"]
        for label, key in sections:
            before_items = self._normalize_skill_section_items(current_sections.get(key, ""), key)
            after_items = self._normalize_skill_section_items(draft_sections.get(key, ""), key)
            if before_items == after_items:
                continue
            lines.append(f"- {label}:")
            removed = [item for item in before_items if item.lower() not in {x.lower() for x in after_items}]
            added = [item for item in after_items if item.lower() not in {x.lower() for x in before_items}]
            if not removed and not added:
                lines.append("  ~ rewritten")
                continue
            for item in removed[:5]:
                lines.append(f"  - - {item}")
            for item in added[:5]:
                lines.append(f"  - + {item}")
            extra = max(0, len(removed) - 5) + max(0, len(added) - 5)
            if extra > 0:
                lines.append(f"  - ... ({extra} more changes)")
        if len(lines) == 1:
            lines.append("- No section changes detected.")
        return "\n".join(lines)

    def _parse_structured_skill_edits(self, instruction):
        text = str(instruction or "").strip()
        if not text:
            return {}
        text = re.sub(r"^\s*edit\s*:?\s*", "", text, flags=re.IGNORECASE)
        pattern = re.compile(
            r"(trigger|steps|tools|guardrails)\s*:\s*(.+?)(?=(?:\s*[;|,]?\s*(?:trigger|steps|tools|guardrails)\s*:)|$)",
            re.IGNORECASE | re.DOTALL,
        )
        edits: dict[str, str] = {}
        for match in pattern.finditer(text):
            key = str(match.group(1) or "").strip().lower()
            value = str(match.group(2) or "").strip()
            if key and value:
                edits[key] = value
        return edits

    def _apply_structured_skill_edits(self, payload, edits):
        updated = {
            "name": str(payload.get("name", "workflow skill")).strip() or "workflow skill",
            "trigger": list(self._normalize_skill_section_items(payload.get("trigger", ""), "trigger")),
            "tools": list(self._normalize_skill_section_items(payload.get("tools", ""), "tools")),
            "steps": list(self._normalize_skill_section_items(payload.get("steps", ""), "steps")),
            "guardrails": list(self._normalize_skill_section_items(payload.get("guardrails", ""), "guardrails")),
        }
        for section, value in edits.items():
            if section in {"trigger", "tools", "steps", "guardrails"}:
                updated[section] = self._normalize_skill_section_items(value, section)
        if not updated["trigger"]:
            updated["trigger"] = ["When this workflow repeats."]
        if not updated["guardrails"]:
            updated["guardrails"] = [
                "Keep outputs concise and factual.",
                "Ask for confirmation before write/send actions.",
            ]
        return updated

    def _skill_signature_from_payload(self, payload):
        normalized = {
            "name": str(payload.get("name", "")).strip().lower(),
            "trigger": self._normalize_skill_section_items(payload.get("trigger", ""), "trigger"),
            "tools": self._normalize_skill_section_items(payload.get("tools", ""), "tools"),
            "steps": self._normalize_skill_section_items(payload.get("steps", ""), "steps"),
            "guardrails": self._normalize_skill_section_items(payload.get("guardrails", ""), "guardrails"),
        }
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True)

    def _skill_signature_from_markdown(self, markdown):
        text = str(markdown or "").strip()
        if not text:
            return None
        title = ""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break
        sections = self._extract_skill_sections(text)
        return self._skill_signature_from_payload(
            {"name": title, "trigger": sections.get("trigger", ""), "tools": sections.get("tools", ""),
             "steps": sections.get("steps", ""), "guardrails": sections.get("guardrails", "")}
        )

    def _skill_rejection_fingerprint(self, payload):
        signature = self._skill_signature_from_payload(payload)
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]

    def _is_skill_rejection_cooldown_active(self, fingerprint, days=30):
        if not fingerprint:
            return False
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.infra.rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT COUNT(*) AS c
            FROM self_improvement_events
            WHERE category = 'skill-cooldown'
              AND lower(title) = ?
              AND created_at > ?
            """,
            (fingerprint.strip().lower(), cutoff),
        )
        if not rows:
            return False
        try:
            return int(rows[0].get("c", 0) or 0) > 0
        except Exception:
            return False

    def _record_skill_rejection_cooldown(self, fingerprint, title, reason, source, payload):
        self.comms.log_improvement_event(
            event_type="lifecycle",
            category="skill-cooldown",
            title=fingerprint,
            payload=json.dumps(
                {"title": title, "source": source, "reason": reason[:500],
                 "cooldown_days": _SKILL_REJECTION_COOLDOWN_DAYS,
                 "signature": self._skill_signature_from_payload(payload)},
                ensure_ascii=True,
            ),
            status="rejected",
        )

    def _update_improvement_event_status(self, event_id, status):
        if not event_id:
            return
        try:
            from memory.retriever import get_vectorstore
            vs = get_vectorstore()
            vs.update_self_improvement_event_status(event_id=event_id, status=status)
        except Exception:
            log.debug("Failed to update self improvement event status", exc_info=True)

    def _activate_skill_reload(self):
        try:
            from skills import reload_skills
            reload_skills()
        except Exception:
            log.debug("Failed to reload skills after lifecycle update", exc_info=True)
