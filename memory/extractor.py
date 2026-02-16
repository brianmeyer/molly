import logging
import threading
import time
from typing import Any

import yaml

import config

log = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()

# Description-enriched entity types for conversational text
ENTITY_SCHEMA = {
    "Person": {
        "description": "Full name or nickname of a person mentioned in conversation",
        "threshold": 0.4,
    },
    "Technology": {
        "description": "Programming language, framework, tool, platform, or technical system",
        "threshold": 0.4,
    },
    "Organization": {
        "description": "Company, institution, team, or named group",
        "threshold": 0.5,
    },
    "Project": {
        "description": "Named project, product, app, or initiative being worked on",
        "threshold": 0.45,
    },
    "Place": {
        "description": "City, country, region, office, or named location",
        "threshold": 0.5,
    },
    "Concept": {
        "description": "Abstract idea, field of study, methodology, or domain topic discussed",
        "threshold": 0.5,
    },
}

# Description-enriched relationship types
_BASE_RELATION_SCHEMA = {
    "works on": {
        "description": "Person actively works on a project or task",
        "threshold": 0.45,
    },
    "works at": {
        "description": "Person is employed at or affiliated with an organization",
        "threshold": 0.5,
    },
    "knows": {
        "description": "Person knows or has a personal connection with another person",
        "threshold": 0.5,
    },
    "uses": {
        "description": "Person or project uses a technology, tool, or platform",
        "threshold": 0.45,
    },
    "located in": {
        "description": "Person, organization, or project is located in or based at a place",
        "threshold": 0.5,
    },
    "discussed with": {
        "description": "Person discussed a topic or entity with another person",
        "threshold": 0.5,
    },
    "interested in": {
        "description": "Person expressed interest in a topic, technology, or project",
        "threshold": 0.45,
    },
    "created": {
        "description": "Person or organization created or built a project or technology",
        "threshold": 0.5,
    },
    "manages": {
        "description": "Person manages or leads a project, team, or organization",
        "threshold": 0.5,
    },
    "depends on": {
        "description": "Project or technology depends on or requires another technology",
        "threshold": 0.5,
    },
    "related to": {
        "description": "General association between two entities discussed together",
        "threshold": 0.4,
    },
    "classmate of": {
        "description": "Person attended the same program, cohort, or school as another person",
        "threshold": 0.45,
    },
    "studied at": {
        "description": "Person attended or was enrolled at an educational institution",
        "threshold": 0.45,
    },
    "alumni of": {
        "description": "Person graduated from an educational institution or program",
        "threshold": 0.45,
    },
    "mentors": {
        "description": "Person mentors or advises another person",
        "threshold": 0.5,
    },
    "mentored by": {
        "description": "Person is mentored or advised by another person",
        "threshold": 0.5,
    },
    "reports to": {
        "description": "Person directly reports to another person in a management hierarchy",
        "threshold": 0.5,
    },
    "collaborates with": {
        "description": "Person works together with another person but not at the same organization",
        "threshold": 0.45,
    },
    "customer of": {
        "description": "Person is a customer, subscriber, or account holder at a company or service",
        "threshold": 0.5,
    },
    "attends": {
        "description": "Person attends or is enrolled at a school, program, or recurring event",
        "threshold": 0.45,
    },
    "parent of": {
        "description": "Person is the parent or guardian of another person (typically a child)",
        "threshold": 0.5,
    },
    "child of": {
        "description": "Person is the child of another person",
        "threshold": 0.5,
    },
    "received from": {
        "description": "Person received a delivery, package, email, or communication from an organization or person",
        "threshold": 0.5,
    },
}


def _load_user_relation_schema() -> dict[str, dict[str, Any]]:
    """Load user-extended relation schema from ~/.molly/workspace/config/relation_schema.yaml."""
    path = getattr(config, "RELATION_SCHEMA_FILE", None)
    if path is None:
        return {}
    schema_path = path.expanduser()
    if not schema_path.exists():
        return {}

    try:
        payload = yaml.safe_load(schema_path.read_text())
    except Exception:
        log.warning("Failed to parse relation schema file: %s", schema_path, exc_info=True)
        return {}

    merged: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = [(str(item), {}) for item in payload]
    else:
        log.warning("Unsupported relation schema format at %s", schema_path)
        return {}

    for raw_name, raw_cfg in items:
        name = str(raw_name or "").strip().lower()
        if not name:
            continue
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
        description = str(cfg.get("description") or f"User-defined relation: {name}").strip()
        threshold_raw = cfg.get("threshold", 0.45)
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            threshold = 0.45
        merged[name] = {"description": description, "threshold": max(0.0, min(1.0, threshold))}
    return merged


RELATION_SCHEMA = dict(_BASE_RELATION_SCHEMA)
RELATION_SCHEMA.update(_load_user_relation_schema())

# Message type classification
MESSAGE_LABELS = {
    "actionable": "Message contains a task, request, reminder, or something requiring follow-up action",
    "informational": "Message shares facts, updates, news, or knowledge without requiring action",
    "personal": "Message about personal life, feelings, relationships, or casual conversation",
    "question": "Message asks a question or seeks information, advice, or clarification",
}


def _get_model():
    """Lazy-load the GLiNER2 model (unified NER + relations + classification).

    Checks for a fine-tuned active model first at ~/.molly/workspace/models/gliner_active/.
    Falls back to the base HuggingFace model if no active model exists.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from pathlib import Path
            from gliner2 import GLiNER2

            active_dir = Path.home() / ".molly" / "workspace" / "models" / "gliner_active"
            if active_dir.exists() and (active_dir / "model.safetensors").exists():
                log.info("Loading fine-tuned GLiNER2 model from %s ...", active_dir)
                _model = GLiNER2.from_pretrained(str(active_dir))
                log.info("Fine-tuned GLiNER2 model loaded.")
            else:
                log.info("Loading GLiNER2 base model (fastino/gliner2-large-v1)...")
                _model = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
                log.info("GLiNER2 base model loaded.")
    return _model


def _build_full_schema():
    """Build a combined schema for entities + relations + message classification."""
    model = _get_model()
    return (
        model.create_schema()
        .entities(ENTITY_SCHEMA)
        .relations(RELATION_SCHEMA)
        .classification("message_type", MESSAGE_LABELS, multi_label=True, cls_threshold=0.35)
    )


def _build_entity_schema():
    """Build an entity-only schema (for lightweight retrieval-time extraction)."""
    model = _get_model()
    return model.create_schema().entities(ENTITY_SCHEMA)


def extract_entities(text: str, threshold: float = 0.4) -> list[dict[str, Any]]:
    """Extract entities only (lightweight, for retrieval path).

    Returns list of {"text": str, "label": str, "score": float}.
    """
    model = _get_model()
    schema = _build_entity_schema()
    result = model.extract(text, schema, threshold=threshold, include_confidence=True)

    entity_dict = result.get("entities", result)

    entities = []
    for etype, items in entity_dict.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                entities.append({
                    "text": item.get("text", ""),
                    "label": etype,
                    "score": item.get("confidence", 0.5),
                })

    return entities


def extract(text: str, threshold: float = 0.4) -> dict[str, Any]:
    """Full extraction: entities + relationships + message classification.

    Returns {
        "entities": [{"text", "label", "score"}, ...],
        "relations": [{"head", "tail", "label", "score"}, ...],
        "message_type": ["actionable", ...],
        "latency_ms": int,
    }.
    """
    model = _get_model()
    schema = _build_full_schema()

    t0 = time.monotonic()

    try:
        result = model.extract(
            text, schema,
            threshold=threshold,
            include_confidence=True,
        )
    except Exception:
        log.error("Extraction failed", exc_info=True)
        return {"entities": [], "relations": [], "message_type": [], "latency_ms": 0}

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Parse entities
    entity_dict = result.get("entities", {})
    entities = []
    for etype, items in entity_dict.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                entities.append({
                    "text": item.get("text", ""),
                    "label": etype,
                    "score": item.get("confidence", 0.5),
                })

    # Parse relations
    rel_dict = result.get("relation_extraction", {})
    relations = []
    for rtype, items in rel_dict.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            head = item.get("head", {})
            tail = item.get("tail", {})
            head_text = head.get("text", "") if isinstance(head, dict) else str(head)
            tail_text = tail.get("text", "") if isinstance(tail, dict) else str(tail)
            head_conf = head.get("confidence", 0.5) if isinstance(head, dict) else 0.5
            tail_conf = tail.get("confidence", 0.5) if isinstance(tail, dict) else 0.5

            if head_text and tail_text:
                relations.append({
                    "head": head_text,
                    "tail": tail_text,
                    "label": rtype,
                    "score": min(head_conf, tail_conf),
                })

    # Parse message classification
    msg_type_raw = result.get("message_type", [])
    if isinstance(msg_type_raw, list):
        message_type = [
            item.get("label", item) if isinstance(item, dict) else str(item)
            for item in msg_type_raw
        ]
    elif isinstance(msg_type_raw, dict):
        message_type = [msg_type_raw.get("label", "")]
    elif isinstance(msg_type_raw, str):
        message_type = [msg_type_raw]
    else:
        message_type = []

    log.debug(
        "Extracted %d entities, %d relations, types=%s in %dms",
        len(entities), len(relations), message_type, latency_ms,
    )

    return {
        "entities": entities,
        "relations": relations,
        "message_type": message_type,
        "latency_ms": latency_ms,
    }
