#!/bin/zsh
set -euo pipefail

# Resolve PROJECT_DIR from this script's location (works on any machine)
PROJECT_DIR="${0:A:h}"
STATE_DIR="$PROJECT_DIR/state"
STATE_FILE="$STATE_DIR/last_successful_run_date.txt"
LOG_DIR="$PROJECT_DIR/logs"
PIPELINE_SCRIPT="$PROJECT_DIR/run_pipeline.sh"

mkdir -p "$STATE_DIR" "$LOG_DIR"

# Use the venv's Python for the gate check too (system Python may be too old)
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[launchd-entrypoint] ERROR: venv python not found at $VENV_PYTHON" >&2
    exit 1
fi

# --- Gate: only run once per weekday after 5 PM PT ---
GATE_RESULT=$("$VENV_PYTHON" - "$STATE_FILE" <<'PY'
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

state_file = Path(sys.argv[1])
tz = ZoneInfo("America/Los_Angeles")
now = datetime.now(tz)

is_weekday = now.weekday() < 5
after_cutoff = (now.hour, now.minute) >= (17, 0)
today_str = now.date().isoformat()

last_success = None
if state_file.exists():
    last_success = state_file.read_text().strip() or None

should_run = is_weekday and after_cutoff and last_success != today_str

print(f"[launchd-entrypoint] now={now.isoformat()}")
print(f"[launchd-entrypoint] is_weekday={is_weekday} after_cutoff={after_cutoff} last_success={last_success}")
print(f"[launchd-entrypoint] should_run={should_run}")

sys.exit(0 if should_run else 10)
PY
) && GATE_EXIT=0 || GATE_EXIT=$?

echo "$GATE_RESULT"

if [[ "$GATE_EXIT" -eq 10 ]]; then
    exit 0
elif [[ "$GATE_EXIT" -ne 0 ]]; then
    echo "[launchd-entrypoint] gate check failed with code $GATE_EXIT" >&2
    exit 1
fi

# --- Run the pipeline ---
/bin/zsh "$PIPELINE_SCRIPT"
PIPE_EXIT=$?

if [[ "$PIPE_EXIT" -eq 0 ]]; then
    TODAY=$("$VENV_PYTHON" -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat())")
    echo "$TODAY" > "$STATE_FILE"
    echo "[launchd-entrypoint] recorded success for $TODAY"
else
    echo "[launchd-entrypoint] pipeline failed with code $PIPE_EXIT" >&2
    exit "$PIPE_EXIT"
fi
