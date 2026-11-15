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
# The GUI must not be open during a run (both write the model index): the run waits for it to be
# closed (up to GUI_WAIT_MINUTES, default 60) and is skipped otherwise.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROPHITBET_PYTHON:-$HOME/prophitbet-venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "Python interpreter not found: $PYTHON (set PROPHITBET_PYTHON)" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p storage/logs

gui_is_open() {
    local pid
    for pid in $(pgrep -f 'python[0-9.]* .*main\.py$|python[0-9.]* main\.py' 2>/dev/null); do
        case "$(basename "$(readlink -f "/proc/$pid/exe" 2>/dev/null)")" in
            python*) ;;
            *) continue ;;
        esac
        if [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$PROJECT_ROOT" ]; then
            return 0
        fi
    done
    return 1
}

waited=0
while gui_is_open; do
    if [ "$waited" -ge "${GUI_WAIT_MINUTES:-60}" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') WARNING Run skipped: the ProphitBet GUI is still open." >> storage/logs/automation.log
        exit 3
    fi
    sleep 300
    waited=$((waited + 5))
done

if [ -f storage/configs/automation_plan.json ]; then
    exec "$PYTHON" cli.py run --plan --all "$@"
fi
exec "$PYTHON" cli.py run --all "$@"
