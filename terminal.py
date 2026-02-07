"""Terminal REPL for Molly (Phase 4).

Standalone CLI for debugging and local interaction.
Uses the same handle_message() pipeline as WhatsApp and Web UI.

Usage:
    python terminal.py
"""

import asyncio
import logging
import sys

import config
from agent import handle_message

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


async def main():
    """Interactive terminal session with Molly."""
    session_id = None
    session_key = "terminal"

    print("Molly Terminal (type /clear to reset session, /quit to exit)")
    print("-" * 50)

    while True:
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("\nyou> ")
            )
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        text = text.strip()
        if not text:
            continue

        if text == "/quit":
            print("Bye!")
            break

        if text == "/clear":
            session_id = None
            print("Session cleared.")
            continue

        try:
            response, new_session_id = await handle_message(
                text, session_key, session_id,
            )
            if new_session_id:
                session_id = new_session_id
            print(f"\nmolly> {response}")
        except Exception as e:
            print(f"\n[error] {e}")


if __name__ == "__main__":
    asyncio.run(main())
