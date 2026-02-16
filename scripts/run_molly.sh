#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MAIN_PATH="$ROOT_DIR/main.py"
RESTART_CODE="${MOLLY_RESTART_EXIT_CODE:-42}"
RESTART_DELAY_SECONDS="${MOLLY_RESTART_DELAY_SECONDS:-2}"
RESTART_ON_CRASH="${MOLLY_RESTART_ON_CRASH:-1}"
STOP_REQUESTED=0
CHILD_PID=""

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN"
    exit 1
fi

if [[ ! -f "$MAIN_PATH" ]]; then
    echo "main.py not found: $MAIN_PATH"
    exit 1
fi

request_stop() {
    STOP_REQUESTED=1
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" >/dev/null 2>&1; then
        kill -TERM "$CHILD_PID" >/dev/null 2>&1 || true
    fi
}

trap 'request_stop' INT TERM

while true; do
    "$PYTHON_BIN" "$MAIN_PATH" "$@" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    EXIT_CODE=$?
    CHILD_PID=""

    if [[ "$STOP_REQUESTED" -eq 1 ]]; then
        echo "Shutdown requested. Stopping supervisor."
        exit 0
    fi

    if [[ "$EXIT_CODE" -eq "$RESTART_CODE" ]]; then
        echo "Molly requested restart (exit $EXIT_CODE). Restarting in ${RESTART_DELAY_SECONDS}s..."
        sleep "$RESTART_DELAY_SECONDS"
        continue
    fi

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        echo "Molly exited cleanly. Stopping supervisor."
        exit 0
    fi

    if [[ "$RESTART_ON_CRASH" == "1" ]]; then
        echo "Molly crashed (exit $EXIT_CODE). Restarting in ${RESTART_DELAY_SECONDS}s..."
        sleep "$RESTART_DELAY_SECONDS"
        continue
    fi

    echo "Molly exited with code $EXIT_CODE. Stopping supervisor."
    exit "$EXIT_CODE"
done
