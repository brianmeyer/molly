"""Automation-pattern detection and proposal service.

Extracted from ``evolution/skills.py`` (SelfImprovementEngine).  Owns
workflow-pattern detection from tool-call history (with Foundry signal
fusion), automation proposal generation via the gateway, and discovery
of existing automation identifiers.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import config

log = logging.getLogger(__name__)



class AutomationPatternsService:
    """Workflow pattern detection and automation proposals.

    Uses ``pattern_helpers`` (pure functions) to break circular imports.
    """

    def __init__(self, ctx, infra, comms):
        from evolution.context import EngineContext
        from evolution.infra import InfraService
        from evolution.owner_comms import OwnerCommsService
        self.ctx: EngineContext = ctx
        self.infra: InfraService = infra
        self.comms: OwnerCommsService = comms

    def detect_workflow_patterns(self, days: int = 30, min_occurrences: int = 3) -> list[dict[str, Any]]:
        from evolution.pattern_helpers import is_low_value_workflow_tool, load_foundry_signals

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.infra.rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT tool_name, created_at
            FROM tool_calls
            WHERE created_at > ?
            ORDER BY created_at ASC
            """,
            (cutoff,),
        )
        tools = []
        for row in rows:
            tool_name = str(row.get("tool_name", "")).strip()
            if is_low_value_workflow_tool(tool_name):
                continue
            tools.append(tool_name)

        tool_call_counts: dict[str, int] = {}
        for i in range(0, len(tools) - 2):
            seq = tools[i:i + 3]
            if len(set(seq)) == 1:
                continue
            key = " -> ".join(seq)
            tool_call_counts[key] = tool_call_counts.get(key, 0) + 1

        foundry_signals = self.load_foundry_sequence_signals(days=days)
        if not tool_call_counts and not foundry_signals:
            return []

        existing_ids = self.existing_automation_ids()
        patterns = []
        all_sequences = set(tool_call_counts) | set(foundry_signals)
        for seq in sorted(all_sequences):
            sequence_tools = [step.strip() for step in seq.split(" -> ") if step.strip()]
            if len(sequence_tools) != 3:
                continue
            if any(is_low_value_workflow_tool(step) for step in sequence_tools):
                continue
            tool_count = int(tool_call_counts.get(seq, 0) or 0)
            foundry_signal = foundry_signals.get(seq)
            foundry_count = int(foundry_signal.count if foundry_signal else 0)
            total_count = tool_count + foundry_count
            if total_count < min_occurrences:
                continue
            overlap = sum(1 for tid in existing_ids if any(t in tid for t in sequence_tools))
            foundry_success_rate = foundry_signal.success_rate if foundry_signal else 0.0
            foundry_boost = min(0.35, (0.08 * foundry_count) + (0.10 * foundry_success_rate))
            confidence = min(0.99, 0.4 + (0.15 * tool_count) + (0.05 * overlap) + foundry_boost)

            source = "tool_calls"
            if tool_count > 0 and foundry_count > 0:
                source = "tool_calls+foundry"
            elif foundry_count > 0:
                source = "foundry"

            pattern = {
                "name": f"Workflow {sequence_tools[0]}",
                "steps": sequence_tools,
                "steps_text": seq,
                "count": total_count,
                "tool_call_count": tool_count,
                "foundry_count": foundry_count,
                "foundry_success_rate": round(foundry_success_rate, 2),
                "confidence": confidence,
                "trigger": "Detected repeated workflow chain",
                "example": seq,
                "source": source,
            }
            if foundry_signal and foundry_signal.latest_at:
                pattern["foundry_latest_at"] = foundry_signal.latest_at
            patterns.append(pattern)
        return sorted(patterns, key=lambda p: (p["count"], p["confidence"]), reverse=True)

    async def propose_automation_updates_from_patterns(self, patterns: list[dict[str, Any]]):
        if not patterns:
            return
        from evolution.pattern_helpers import pattern_steps
        from gateway import propose_automation

        selected = []
        for pattern in patterns:
            steps = pattern_steps(pattern)
            if len(steps) < 3:
                continue
            normalized = dict(pattern)
            normalized["steps"] = " -> ".join(steps)
            selected.append(normalized)
            if len(selected) >= 3:
                break
        if not selected:
            return
        skeleton = propose_automation(selected)
        if not skeleton:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = self.ctx.automations_dir / f"proposal-{ts}.yaml"
        path.write_text(skeleton)
        msg = (
            "🤖 Automation proposal from workflow patterns\n\n"
            f"Patterns considered: {len(selected)}\n"
            f"Draft file: {path}\n"
            "Reply YES to keep draft, NO to discard."
        )
        await self.comms.notify_owner(msg)
        decision = await self.comms.request_owner_decision(
            category="self-improve-automation",
            description=msg,
            required_keyword="YES",
            allow_edit=False,
        )
        status = "approved" if decision is True else "rejected"
        self.comms.log_improvement_event(
            event_type="proposal",
            category="automation",
            title="Workflow-derived automation proposal",
            payload=str(path),
            status=status,
        )
        if decision is not True:
            path.unlink(missing_ok=True)

    def existing_automation_ids(self) -> set[str]:
        ids = set()
        for path in config.AUTOMATIONS_DIR.glob("*.yaml"):
            ids.add(path.stem.lower())
        return ids

    def load_foundry_sequence_signals(self, days: int = 30) -> dict:
        from evolution.pattern_helpers import load_foundry_signals
        return load_foundry_signals(days=days)
