"""Contact resolution from Google Contacts JSON export.

Loads store/contacts.json (produced by scripts/import_contacts.py) and
provides phone→name lookup for WhatsApp, iMessage, and /groups.
"""

import json
import logging
import re
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_NON_DIGITS = re.compile(r"\D+")
STORE_DIR = Path(__file__).resolve().parent / "store"
CONTACTS_FILE = STORE_DIR / "contacts.json"

_resolver: "ContactResolver | None" = None
_resolver_lock = threading.Lock()


def _normalize_phone(raw: str) -> str:
    """Strip non-digits, drop leading country code, return last 10 digits."""
    digits = _NON_DIGITS.sub("", str(raw))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits[-10:] if len(digits) >= 10 else digits


class ContactResolver:
    """In-memory contact lookup backed by store/contacts.json."""

    def __init__(self):
        self._contacts: dict[str, dict] = {}
        self._load_contacts()

    def _load_contacts(self):
        if not CONTACTS_FILE.exists():
            log.warning("No contacts file at %s — phone resolution disabled", CONTACTS_FILE)
            return
        try:
            data = json.loads(CONTACTS_FILE.read_text())
            self._contacts = data
            log.info("Loaded %d contact entries from %s", len(data), CONTACTS_FILE)
        except (json.JSONDecodeError, OSError) as e:
            log.error("Failed to load contacts: %s", e)

    def resolve_phone(self, phone: str) -> str | None:
        """Look up a phone number and return the contact name, or None."""
        norm = _normalize_phone(phone)
        entry = self._contacts.get(norm)
        return entry["name"] if entry else None

    def resolve_phone_entry(self, phone: str) -> dict | None:
        """Look up a phone number and return the full entry {name, email, phone_raw}."""
        norm = _normalize_phone(phone)
        return self._contacts.get(norm)

    def cache_pushname(self, phone: str, pushname: str):
        """Seed the in-memory dict when WhatsApp provides a pushname."""
        norm = _normalize_phone(phone)
        if norm and pushname and norm not in self._contacts:
            self._contacts[norm] = {"name": pushname, "email": "", "phone_raw": phone}

    def enrich_graph(self, name: str, phone: str, source: str, email: str = ""):
        """Fire-and-forget: upsert a Person entity with phone/email and CONTACT_OF → Brian."""
        try:
            from memory.graph import upsert_entity, upsert_relationship, get_driver

            canonical = upsert_entity(name, "Person", 0.9)

            driver = get_driver()
            with driver.session() as session:
                # Set phone and email properties
                props: dict = {}
                if phone:
                    props["phone"] = _normalize_phone(phone)
                if email:
                    props["email"] = email
                if props:
                    set_clauses = ", ".join(f"e.{k} = ${k}" for k in props)
                    session.run(
                        f"MATCH (e:Entity {{name: $name}}) SET {set_clauses}",
                        name=canonical,
                        **props,
                    )

            # Create CONTACT_OF → Brian relationship
            brian = upsert_entity("Brian", "Person", 1.0)
            upsert_relationship(canonical, brian, "CONTACT_OF", 0.9, f"from {source}")
        except Exception:
            log.debug("Contact graph enrichment failed for %s", name, exc_info=True)


def get_resolver() -> ContactResolver:
    """Return the singleton ContactResolver (lazy-init, thread-safe)."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = ContactResolver()
    return _resolver
