#!/bin/bash
# Scheduled ProphitBet run: refresh data -> create missing leagues/models from the plan (if
# storage/configs/automation_plan.json exists) -> retrain models -> predict upcoming fixtures.
#
# Usage:   scripts/run_automation.sh [extra cli.py "run" arguments]
# Example: scripts/run_automation.sh --days 3 --tolerance 0.005
#
# Environment:
#   PROPHITBET_PYTHON  Python 3.11 interpreter (default: $HOME/prophitbet-venv/bin/python)
#   PROPHITBET_BACKUP  1 (default) backs up data & models to MEGA after the run (scripts/backup_to_mega.sh)
#   GUI_WAIT_MINUTES   how long to wait for the GUI to be closed (default 60); GUI_POLL_SECONDS check interval (300)
#
# Scheduled by systemd user timers (scripts/systemd/, installed by scripts/setup.sh):
#   Tuesday 17:30 SAST (full run) and Friday 17:30 SAST (--skip-retrain); missed runs start when the computer is back on.
#   View:  systemctl --user list-timers 'prophitbet-*'      Run now:  systemctl --user start prophitbet-tuesday.service
#
# Output: storage/predictions/<date>/  ·  Log: storage/logs/automation.log
# The GUI must not be open during a run (both write the model index): the run waits for it to be
# closed (up to GUI_WAIT_MINUTES) and is skipped otherwise.
#
# The whole script is one function called on the last line: bash reads the complete file before running anything,
# so editing (or updating) this file while a run is waiting or running cannot break that run.

main() {
    set -euo pipefail

    local project_root python
    project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    python="${PROPHITBET_PYTHON:-$HOME/prophitbet-venv/bin/python}"

    if [ ! -x "$python" ]; then
        echo "Python interpreter not found: $python (set PROPHITBET_PYTHON)" >&2
        return 2
    fi

    cd "$project_root"
    mkdir -p storage/logs

    gui_is_open() {
        local pid
        for pid in $(pgrep -f 'python[0-9.]* .*main\.py$|python[0-9.]* main\.py' 2>/dev/null); do
            case "$(basename "$(readlink -f "/proc/$pid/exe" 2>/dev/null)")" in
                python*) ;;
                *) continue ;;
            esac
            if [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$project_root" ]; then
                return 0
            fi
        done
        return 1
    }

    local waited=0 poll="${GUI_POLL_SECONDS:-300}"
    while gui_is_open; do
        if [ "$waited" -ge "$(( ${GUI_WAIT_MINUTES:-60} * 60 ))" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') WARNING Run skipped: the ProphitBet GUI is still open." >> storage/logs/automation.log
            return 3
        fi
        sleep "$poll"
        waited=$((waited + poll))
    done

    local status=0
    if [ -f storage/configs/automation_plan.json ]; then
        "$python" cli.py run --plan --all "$@" || status=$?
    else
        "$python" cli.py run --all "$@" || status=$?
    fi

    # Backing up data & models to MEGA (skipped with a log entry if MEGA is not set up; never fails the run).
    if [ "${PROPHITBET_BACKUP:-1}" = "1" ]; then
        "$project_root/scripts/backup_to_mega.sh" || true
    fi
    return "$status"
}

main "$@"; exit $?
