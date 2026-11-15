#!/bin/bash
# Scheduled ProphitBet run: refresh data -> create missing leagues/models from the plan (if
# storage/configs/automation_plan.json exists) -> retrain models -> predict upcoming fixtures.
#
# Usage:   scripts/run_automation.sh [extra cli.py "run" arguments]
# Example: scripts/run_automation.sh --days 3 --tolerance 0.005
#
# Environment:
#   PROPHITBET_PYTHON  Python 3.11 interpreter (default: $HOME/prophitbet-venv/bin/python)
#
# Example crontab entry (daily at 06:30; see README "Automation"):
#   30 6 * * * /path/to/ProphitBet-Soccer-Bets-Predictor/scripts/run_automation.sh >/dev/null 2>&1
#
# Output: storage/predictions/<date>/  ·  Log: storage/logs/automation.log
# Do not run while the GUI is open (both write the model index).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROPHITBET_PYTHON:-$HOME/prophitbet-venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "Python interpreter not found: $PYTHON (set PROPHITBET_PYTHON)" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
if [ -f storage/configs/automation_plan.json ]; then
    exec "$PYTHON" cli.py run --plan --all "$@"
fi
exec "$PYTHON" cli.py run --all "$@"
