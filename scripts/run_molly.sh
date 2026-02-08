#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MAIN_PATH="$ROOT_DIR/main.py"
RESTART_CODE="${MOLLY_RESTART_EXIT_CODE:-42}"
RESTART_DELAY_SECONDS="${MOLLY_RESTART_DELAY_SECONDS:-2}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN"
    exit 1
fi

if [[ ! -f "$MAIN_PATH" ]]; then
    echo "main.py not found: $MAIN_PATH"
    exit 1
fi

while true; do
    "$PYTHON_BIN" "$MAIN_PATH" "$@"
    EXIT_CODE=$?

    if [[ "$EXIT_CODE" -eq "$RESTART_CODE" ]]; then
        echo "Molly requested restart (exit $EXIT_CODE). Restarting in ${RESTART_DELAY_SECONDS}s..."
        sleep "$RESTART_DELAY_SECONDS"
        continue
    fi

    echo "Molly exited with code $EXIT_CODE. Stopping supervisor."
    exit "$EXIT_CODE"
done
