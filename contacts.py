"""Contact resolution from Google Contacts JSON export.

Loads store/contacts.json (produced by scripts/import_contacts.py) and
provides phone→name lookup for WhatsApp, iMessage, and /groups.
"""

import logging
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config

log = logging.getLogger(__name__)

_NON_DIGITS = re.compile(r"\D+")
STORE_DIR = Path(__file__).resolve().parent / "store"
CONTACTS_FILE = STORE_DIR / "contacts.json"

_resolver: "ContactResolver | None" = None
_resolver_lock = threading.Lock()

# Bounded thread pool for fire-and-forget graph enrichment.
_enrichment_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="contact-enrich")


def normalize_phone(raw: str) -> str:
    """Strip non-digits, drop leading US country code, return last 10 digits.

    This is the canonical phone normalization used everywhere:
    import script, runtime resolver, iMessage, WhatsApp.
    """
    digits = _NON_DIGITS.sub("", str(raw))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits[-10:] if len(digits) >= 10 else digits


class ContactResolver:
    """In-memory contact lookup backed by store/contacts.json."""

    def __init__(self):
        self._contacts: dict[str, dict] = {}
        self._enriched: set[str] = set()  # Dedup: phones already sent to graph
        self._load_contacts()

    def _load_contacts(self):
        if not CONTACTS_FILE.exists():
            log.warning("No contacts file at %s — phone resolution disabled", CONTACTS_FILE)
            return
        try:
            data = json.loads(CONTACTS_FILE.read_text())
            if not isinstance(data, dict):
                log.error("contacts.json is not a JSON object — ignoring")
                return
            # Validate entries have a "name" key
            valid = {}
            for k, v in data.items():
                if isinstance(v, dict) and "name" in v:
                    valid[k] = v
                else:
                    log.warning("Skipping malformed contact entry: %s", k)
            self._contacts = valid
            log.info("Loaded %d contact entries from %s", len(valid), CONTACTS_FILE)
        except (json.JSONDecodeError, OSError) as e:
            log.error("Failed to load contacts: %s", e, exc_info=True)

    def resolve_phone(self, phone: str) -> str | None:
        """Look up a phone number and return the contact name, or None."""
        norm = normalize_phone(phone)
        entry = self._contacts.get(norm)
        return entry["name"] if entry else None

    def resolve_phone_entry(self, phone: str) -> dict | None:
        """Look up a phone number and return the full entry {name, email, phone_raw}."""
        norm = normalize_phone(phone)
        return self._contacts.get(norm)

    def cache_pushname(self, phone: str, pushname: str):
        """Seed the in-memory dict when WhatsApp provides a pushname."""
        norm = normalize_phone(phone)
        if norm and pushname and norm not in self._contacts:
            self._contacts[norm] = {"name": pushname, "email": "", "phone_raw": phone}

    def enrich_graph(self, name: str, phone: str, source: str, email: str = ""):
        """Fire-and-forget: upsert a Person entity with phone/email and CONTACT_OF → owner."""
        try:
            from memory.graph import (
                set_entity_properties,
                upsert_entity_sync,
                upsert_relationship_sync,
            )

            canonical = upsert_entity_sync(name, "Person", 0.9)

            # Set phone and email properties via graph API
            props: dict = {}
            if phone:
                props["phone"] = normalize_phone(phone)
            if email:
                props["email"] = email
            if props:
                set_entity_properties(canonical, props)

            # Create CONTACT_OF → owner relationship
            owner = upsert_entity_sync(config.OWNER_NAME, "Person", 1.0)
            upsert_relationship_sync(canonical, owner, "CONTACT_OF", 0.9, f"from {source}")
        except Exception:
            log.warning("Contact graph enrichment failed for %s", name, exc_info=True)

    def submit_enrichment(self, name: str, phone: str, source: str, email: str = ""):
        """Submit graph enrichment to the bounded thread pool, deduped by phone."""
        norm = normalize_phone(phone)
        if norm in self._enriched:
            return
        self._enriched.add(norm)
        _enrichment_pool.submit(self.enrich_graph, name, phone, source, email)


def get_resolver() -> ContactResolver:
    """Return the singleton ContactResolver (lazy-init, thread-safe)."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = ContactResolver()
    return _resolver
