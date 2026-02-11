#!/usr/bin/env python3
"""Import Google Contacts vCard (.vcf) into store/contacts.json.

Usage:
    python scripts/import_contacts.py ~/Downloads/contacts.vcf

Parses BEGIN:VCARD/END:VCARD blocks, extracts FN, TEL, and EMAIL fields,
normalizes phone numbers to last-10-digits, and writes a JSON dict keyed
by normalized phone.  Multiple phone numbers per contact produce multiple
entries pointing to the same name.
"""

import json
import re
import sys
from pathlib import Path

STORE_DIR = Path(__file__).resolve().parent.parent / "store"
OUTPUT = STORE_DIR / "contacts.json"

_NON_DIGITS = re.compile(r"\D+")


def _normalize_phone(raw: str) -> str | None:
    """Strip non-digits, drop leading country code, return last 10 digits."""
    digits = _NON_DIGITS.sub("", raw)
    if len(digits) < 7:
        return None
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits[-10:]


def parse_vcf(path: Path) -> dict:
    """Parse a .vcf file into {normalized_phone: {name, email, phone_raw}}."""
    text = path.read_text(errors="replace")
    contacts: dict[str, dict] = {}

    for block in re.split(r"(?=BEGIN:VCARD)", text):
        if "END:VCARD" not in block:
            continue

        # Full name
        fn_match = re.search(r"^FN:(.+)$", block, re.MULTILINE)
        if not fn_match:
            continue
        name = fn_match.group(1).strip()

        # Email (first match)
        email_match = re.search(r"^EMAIL[^:]*:(.+)$", block, re.MULTILINE)
        email = email_match.group(1).strip() if email_match else ""

        # All phone numbers
        for tel_match in re.finditer(r"^TEL[^:]*:(.+)$", block, re.MULTILINE):
            raw = tel_match.group(1).strip()
            norm = _normalize_phone(raw)
            if norm:
                contacts[norm] = {
                    "name": name,
                    "email": email,
                    "phone_raw": raw,
                }

    return contacts


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_contacts.py <path-to-contacts.vcf>")
        sys.exit(1)

    vcf_path = Path(sys.argv[1]).expanduser()
    if not vcf_path.exists():
        print(f"File not found: {vcf_path}")
        sys.exit(1)

    contacts = parse_vcf(vcf_path)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(contacts, indent=2))
    print(f"Imported {len(contacts)} phone entries into {OUTPUT}")


if __name__ == "__main__":
    main()
