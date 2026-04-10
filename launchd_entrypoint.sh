#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/anish/Files/stock_temp/Stock-analysis"
STATE_DIR="$PROJECT_DIR/state"
STATE_FILE="$STATE_DIR/last_successful_run_date.txt"
LOG_DIR="$PROJECT_DIR/logs"
PIPELINE_SCRIPT="$PROJECT_DIR/run_pipeline.sh"

mkdir -p "$STATE_DIR" "$LOG_DIR"

export PROJECT_DIR STATE_FILE PIPELINE_SCRIPT

/usr/bin/python3 <<'PY'
import os
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

project_dir = Path(os.environ["PROJECT_DIR"])
state_file = Path(os.environ["STATE_FILE"])
pipeline_script = Path(os.environ["PIPELINE_SCRIPT"])

tz = ZoneInfo("America/Los_Angeles")
now = datetime.now(tz)

# Python weekday(): Monday=0 ... Sunday=6
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

if not should_run:
    sys.exit(0)

result = subprocess.run(
    ["/bin/zsh", str(pipeline_script)],
    cwd=str(project_dir),
)

if result.returncode == 0:
    state_file.write_text(today_str + "\n")
    print(f"[launchd-entrypoint] recorded success for {today_str}")
    sys.exit(0)
else:
    print(f"[launchd-entrypoint] pipeline failed with code {result.returncode}", file=sys.stderr)
    sys.exit(result.returncode)
PY
