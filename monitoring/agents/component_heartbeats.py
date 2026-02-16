"""Component Heartbeats — 10 component availability checks."""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timezone

from monitoring._base import HealthCheck

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_component_heartbeats(molly=None) -> list[HealthCheck]:
    """Run all 10 component heartbeat checks and return results."""
    checks: list[HealthCheck] = []

    wa_status, wa_detail = _whatsapp_status(molly)
    checks.append(
        HealthCheck(
            check_id="component.whatsapp",
            layer="Component Heartbeats",
            label="WhatsApp",
            status=wa_status,
            detail=wa_detail,
            action_required=(wa_status == "red"),
        )
    )

    neo_status, neo_detail = _neo4j_heartbeat()
    checks.append(
        HealthCheck(
            check_id="component.neo4j",
            layer="Component Heartbeats",
            label="Neo4j",
            status=neo_status,
            detail=neo_detail,
            action_required=(neo_status == "red"),
        )
    )

    triage_loaded = _module_attr_loaded("memory.triage", "_TRIAGE_MODEL")
    checks.append(
        HealthCheck(
            check_id="component.triage_model",
            layer="Component Heartbeats",
            label="Triage model",
            status="green" if triage_loaded else "red",
            detail="loaded" if triage_loaded else "not loaded",
            action_required=not triage_loaded,
        )
    )

    embedding_status, embedding_detail = _embedding_model_status()
    checks.append(
        HealthCheck(
            check_id="component.embedding_model",
            layer="Component Heartbeats",
            label="EmbeddingGemma",
            status=embedding_status,
            detail=embedding_detail,
            action_required=(embedding_status == "red"),
        )
    )

    extractor_status, extractor_detail = _gliner_model_status()
    checks.append(
        HealthCheck(
            check_id="component.gliner_model",
            layer="Component Heartbeats",
            label="GLiNER2",
            status=extractor_status,
            detail=extractor_detail,
            action_required=(extractor_status == "red"),
        )
    )

    oauth_status, oauth_detail = _google_oauth_status()
    checks.append(
        HealthCheck(
            check_id="component.google_oauth",
            layer="Component Heartbeats",
            label="Google OAuth",
            status=oauth_status,
            detail=oauth_detail,
            action_required=(oauth_status == "red"),
        )
    )

    mcp_status, mcp_detail = _mcp_servers_status()
    checks.append(
        HealthCheck(
            check_id="component.mcp_servers",
            layer="Component Heartbeats",
            label="MCP tools",
            status=mcp_status,
            detail=mcp_detail,
            action_required=(mcp_status == "red"),
        )
    )

    auto_status, auto_detail = _automation_engine_status(molly)
    checks.append(
        HealthCheck(
            check_id="component.automation_engine",
            layer="Component Heartbeats",
            label="Automation engine",
            status=auto_status,
            detail=auto_detail,
            action_required=(auto_status == "red"),
        )
    )

    disk_status, disk_detail = _disk_status()
    checks.append(
        HealthCheck(
            check_id="component.disk_space",
            layer="Component Heartbeats",
            label="Disk",
            status=disk_status,
            detail=disk_detail,
            action_required=(disk_status == "red"),
        )
    )

    ram_status, ram_detail = _ram_status()
    checks.append(
        HealthCheck(
            check_id="component.ram_usage",
            layer="Component Heartbeats",
            label="RAM",
            status=ram_status,
            detail=ram_detail,
            action_required=(ram_status == "red"),
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _module_attr_loaded(module_name: str, attr_name: str) -> bool:
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name, None) is not None
    except Exception:
        return False


def _whatsapp_status(molly) -> tuple[str, str]:
    wa = getattr(molly, "wa", None) if molly else None
    if wa is None:
        return "red", "client not initialized"
    connected = bool(getattr(wa, "connected", False))
    bridge_client = getattr(wa, "client", None)
    bridge_identity = getattr(bridge_client, "me", None) if bridge_client else None
    if connected:
        return "green", "connected"
    if bridge_identity:
        return "green", "authenticated (bridge)"
    return "red", "disconnected"


def _embedding_model_status() -> tuple[str, str]:
    try:
        module = __import__("memory.embeddings", fromlist=["_model"])
        model = getattr(module, "_model", None)
        if model is None:
            return "red", "not loaded"
        try:
            vec = model.encode("health", normalize_embeddings=True)
        except TypeError:
            vec = model.encode("health")
        dim = 0
        shape = getattr(vec, "shape", None)
        if shape and len(shape) > 0:
            dim = int(shape[-1])
        elif isinstance(vec, (list, tuple)):
            if vec and isinstance(vec[0], (list, tuple)):
                dim = len(vec[0])
            else:
                dim = len(vec)
        if dim <= 0:
            return "red", "loaded, encode sanity failed"
        return "green", f"loaded ({dim}d sanity ok)"
    except Exception as exc:
        return "red", f"probe failed ({exc})"


def _gliner_model_status() -> tuple[str, str]:
    try:
        module = __import__("memory.extractor", fromlist=["_model"])
        model = getattr(module, "_model", None)
        if model is None:
            return "red", "not loaded"
        if not callable(getattr(model, "extract", None)):
            return "red", "loaded, extract unavailable"
        return "green", "loaded"
    except Exception as exc:
        return "red", f"probe failed ({exc})"


def _neo4j_heartbeat() -> tuple[str, str]:
    try:
        from memory.graph import get_driver

        t0 = time.monotonic()
        driver = get_driver()
        with driver.session() as session:
            session.run("RETURN 1").single()
        ms = int((time.monotonic() - t0) * 1000)
        if ms < 100:
            return "green", f"responding ({ms}ms)"
        if ms <= 500:
            return "yellow", f"slow ({ms}ms)"
        return "red", f"very slow ({ms}ms)"
    except Exception as exc:
        return "red", f"unreachable ({exc})"


def _google_oauth_status() -> tuple[str, str]:
    try:
        from tools.google_auth import get_credentials

        creds = get_credentials()
        if not creds or not creds.valid:
            return "red", "invalid or expired"
        expiry = getattr(creds, "expiry", None)
        if expiry is None:
            return "green", "valid"
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        mins = int((expiry - datetime.now(timezone.utc)).total_seconds() // 60)
        if mins < 0:
            return "red", "expired"
        if mins < 60:
            return "yellow", f"valid (<1h remaining, {mins}m)"
        return "green", f"valid ({mins}m remaining)"
    except Exception as exc:
        return "red", f"refresh failed ({exc})"


def _mcp_servers_status() -> tuple[str, str]:
    try:
        from agent import _MCP_SERVER_SPECS, _MCP_SERVER_TOOL_NAMES

        failures: list[str] = []
        total_servers = 0
        total_tools = 0
        healthy_tools = 0
        for server_name, spec in _MCP_SERVER_SPECS.items():
            total_servers += 1
            tool_count = max(1, len(_MCP_SERVER_TOOL_NAMES.get(server_name, set())))
            total_tools += tool_count
            import config as _cfg

            if server_name in getattr(_cfg, "DISABLED_MCP_SERVERS", set()):
                failures.append(server_name)
                continue
            try:
                if isinstance(spec, tuple):
                    module_name, attr_name = spec
                    mod = __import__(module_name, fromlist=[attr_name])
                    getattr(mod, attr_name)
                elif isinstance(spec, dict):
                    command = str(spec.get("command", "")).strip()
                    if not command:
                        raise RuntimeError("missing command")
                    if shutil.which(command) is None:
                        raise RuntimeError(f"command not found: {command}")
                else:
                    raise RuntimeError(f"unsupported MCP spec: {type(spec)!r}")
                healthy_tools += tool_count
            except Exception:
                failures.append(server_name)
        if not failures:
            return (
                "green",
                f"{healthy_tools}/{total_tools} tools responding "
                f"({total_servers}/{total_servers} servers)",
            )
        if len(failures) <= 2:
            return (
                "yellow",
                f"{healthy_tools}/{total_tools} tools responding; "
                f"down={', '.join(failures)}",
            )
        return (
            "red",
            f"{healthy_tools}/{total_tools} tools responding; "
            f"down={', '.join(failures)}",
        )
    except Exception as exc:
        return "red", f"probe failed ({exc})"


def _automation_engine_status(molly) -> tuple[str, str]:
    engine = getattr(molly, "automations", None) if molly else None
    if not engine:
        return "red", "not initialized"
    loaded = len(getattr(engine, "_automations", {}) or {})
    initialized = bool(getattr(engine, "_initialized", False))
    if initialized and loaded > 0:
        return "green", f"initialized ({loaded} loaded)"
    if initialized:
        return "yellow", "initialized (0 loaded)"
    return "red", "not initialized"


def _disk_status() -> tuple[str, str]:
    import config as _cfg

    usage = shutil.disk_usage(str(_cfg.WORKSPACE))
    free_gb = usage.free / (1024**3)
    if free_gb > 5:
        status = "green"
    elif free_gb > 1:
        status = "yellow"
    else:
        status = "red"
    return status, f"{free_gb:.1f}GB free"


def _ram_status() -> tuple[str, str]:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if os.uname().sysname == "Darwin" else rss * 1024
        rss_gb = rss_bytes / (1024**3)
        if rss_gb < 10:
            return "green", f"{rss_gb:.1f}GB used"
        if rss_gb < 14:
            return "yellow", f"{rss_gb:.1f}GB used"
        return "red", f"{rss_gb:.1f}GB used"
    except Exception as exc:
        return "yellow", f"unavailable ({exc})"
