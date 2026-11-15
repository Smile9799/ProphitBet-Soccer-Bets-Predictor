#!/bin/bash
# ProphitBet one-shot setup for Linux Mint / Ubuntu (fresh machine or re-run on an existing install).
#
#   1. Installs system packages (git, Graphviz, Python 3.11 + venv) when missing.
#   2. Clones (or updates) https://github.com/Smile9799/ProphitBet-Soccer-Bets-Predictor
#   3. Creates the Python 3.11 virtual environment and installs requirements.txt.
#   4. Installs MEGAcmd when missing and logs in to MEGA (prompts for e-mail/password).
#   5. Restores the latest MEGA backup (leagues, models, predictions) if you want it.
#   6. Creates a desktop / application-menu launcher with the ProphitBet icon.
#   7. Installs the Tuesday & Friday 17:30 automation schedule (cron), if you want it.
#
# Fresh machine (download first so the script can ask questions):
#   curl -fsSL https://raw.githubusercontent.com/Smile9799/ProphitBet-Soccer-Bets-Predictor/main/scripts/setup.sh -o setup.sh
#   bash setup.sh
#
# Options:
#   --dir PATH          install folder (default: ~/ProphitBet-Soccer-Bets-Predictor, or this clone if run from one)
#   --venv PATH         virtual environment (default: ~/prophitbet-venv)
#   --yes               accept all defaults without asking
#   --restore / --no-restore     restore the latest MEGA backup (default: ask; "yes" only if no local data exists)
#   --cron / --no-cron           install the automation schedule (default: ask, yes)
#   --no-desktop        do not create the launcher icon
#   --skip-deps         do not install Python requirements (pip)
#   --skip-mega         do not install / log in to MEGA

set -euo pipefail

REPO_URL="https://github.com/Smile9799/ProphitBet-Soccer-Bets-Predictor.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../main.py" ] && [ -d "$SCRIPT_DIR/../.git" ]; then
    INSTALL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    INSTALL_DIR="$HOME/ProphitBet-Soccer-Bets-Predictor"
fi
VENV_DIR="$HOME/prophitbet-venv"
ASSUME_YES=0
RESTORE=ask
CRON=ask
DESKTOP=1
SKIP_DEPS=0
SKIP_MEGA=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) INSTALL_DIR="$(realpath -m "$2")"; shift ;;
        --venv) VENV_DIR="$(realpath -m "$2")"; shift ;;
        --yes) ASSUME_YES=1 ;;
        --restore) RESTORE=yes ;;
        --no-restore) RESTORE=no ;;
        --cron) CRON=yes ;;
        --no-cron) CRON=no ;;
        --no-desktop) DESKTOP=0 ;;
        --skip-deps) SKIP_DEPS=1 ;;
        --skip-mega) SKIP_MEGA=1 ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m    WARNING: %s\033[0m\n' "$*"; }

HAS_TTY=0
if [ -r /dev/tty ] && [ -w /dev/tty ] && (exec </dev/tty) 2>/dev/null; then HAS_TTY=1; fi

# ask "question" default(y/n) -> returns 0 for yes
ask() {
    local question="$1" default="$2" answer
    if [ "$ASSUME_YES" = 1 ] || [ "$HAS_TTY" = 0 ]; then
        [ "$default" = y ]; return
    fi
    local hint="[Y/n]"; [ "$default" = n ] && hint="[y/N]"
    read -r -p "    $question $hint " answer </dev/tty || answer=""
    answer="${answer:-$default}"
    [[ "$answer" =~ ^[Yy] ]]
}

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This setup script supports Debian-based Linux (Linux Mint, Ubuntu). See README.md for other systems." >&2
    exit 1
fi

# ----------------------------------------------------------------------------------------------------------------
step "1/7 System packages"
MISSING=()
for pkg in git curl graphviz; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done
if ! command -v python3.11 >/dev/null 2>&1; then
    MISSING+=(python3.11 python3.11-venv)
elif ! python3.11 -c 'import ensurepip, venv' >/dev/null 2>&1; then
    MISSING+=(python3.11-venv)
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    info "Installing: ${MISSING[*]} (sudo password may be requested)"
    sudo apt-get update -qq
    if [[ " ${MISSING[*]} " == *" python3.11 "* ]] && ! apt-cache show python3.11 >/dev/null 2>&1; then
        info "Python 3.11 is not in the default repositories: adding the deadsnakes PPA."
        sudo apt-get install -y -qq software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update -qq
    fi
    sudo apt-get install -y -qq "${MISSING[@]}"
else
    info "git, curl, Graphviz and Python 3.11 are installed."
fi

# ----------------------------------------------------------------------------------------------------------------
step "2/7 ProphitBet code ($INSTALL_DIR)"
if [ -d "$INSTALL_DIR/.git" ]; then
    if [ -n "$(git -C "$INSTALL_DIR" status --porcelain --untracked-files=no)" ]; then
        warn "Local code changes found: skipping 'git pull' (commit or stash them to update)."
    else
        git -C "$INSTALL_DIR" pull --ff-only -q && info "Updated to $(git -C "$INSTALL_DIR" log --oneline -1)"
    fi
else
    git clone -q "$REPO_URL" "$INSTALL_DIR"
    info "Cloned $(git -C "$INSTALL_DIR" log --oneline -1)"
fi
cd "$INSTALL_DIR"
chmod +x prophitbet.sh scripts/*.sh

# ----------------------------------------------------------------------------------------------------------------
step "3/7 Python environment ($VENV_DIR)"
if [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
    python3.11 -m venv "$VENV_DIR"
    info "Created virtual environment."
fi
VENV_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$VENV_VERSION" != "3.11" ]; then
    echo "The virtual environment at $VENV_DIR uses Python $VENV_VERSION; ProphitBet requires 3.11." >&2
    exit 1
fi
if [ "$SKIP_DEPS" = 0 ]; then
    info "Installing Python libraries (first time: several minutes, ~3 GB)..."
    "$VENV_DIR/bin/python" -m pip install -q --upgrade pip
    "$VENV_DIR/bin/python" -m pip install -q -r requirements.txt
    "$VENV_DIR/bin/python" -m pip check >/dev/null 2>&1 || warn "pip check reported dependency conflicts."
    info "Python libraries installed."
else
    info "Skipped (--skip-deps)."
fi

# ----------------------------------------------------------------------------------------------------------------
step "4/7 MEGA"
MEGA_READY=0
if [ "$SKIP_MEGA" = 1 ]; then
    info "Skipped (--skip-mega)."
else
    if ! command -v mega-whoami >/dev/null 2>&1; then
        UBUNTU_CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}")"
        case "$UBUNTU_CODENAME" in
            noble) UBUNTU_VERSION=24.04 ;;
            jammy) UBUNTU_VERSION=22.04 ;;
            focal) UBUNTU_VERSION=20.04 ;;
            *) UBUNTU_VERSION="" ;;
        esac
        if [ -z "$UBUNTU_VERSION" ]; then
            warn "Unknown Ubuntu base '$UBUNTU_CODENAME': install MEGAcmd manually from https://mega.io/cmd"
        else
            DEB_DIR="$(mktemp -d)"
            DEB="megacmd-xUbuntu_${UBUNTU_VERSION}_amd64.deb"
            info "Installing MEGAcmd (Ubuntu $UBUNTU_VERSION package)..."
            curl -fsSL "https://mega.nz/linux/repo/xUbuntu_${UBUNTU_VERSION}/amd64/${DEB}" -o "$DEB_DIR/$DEB"
            chmod 755 "$DEB_DIR"; chmod 644 "$DEB_DIR/$DEB"
            sudo apt-get install -y -qq "$DEB_DIR/$DEB"
            rm -rf "$DEB_DIR"
        fi
    fi

    if command -v mega-whoami >/dev/null 2>&1; then
        if mega-whoami >/dev/null 2>&1; then
            MEGA_READY=1
            info "Logged in: $(mega-whoami 2>/dev/null | grep -i 'e-mail' | sed 's/.*: //')"
        elif [ "$HAS_TTY" = 1 ] && [ "$ASSUME_YES" = 0 ]; then
            info "Log in to MEGA (leave the e-mail empty to skip). The password is not shown or stored in history."
            read -r -p "    MEGA e-mail: " MEGA_EMAIL </dev/tty || MEGA_EMAIL=""
            if [ -n "$MEGA_EMAIL" ]; then
                read -r -s -p "    MEGA password: " MEGA_PASSWORD </dev/tty; echo
                if mega-login "$MEGA_EMAIL" "$MEGA_PASSWORD" >/dev/null 2>&1; then
                    MEGA_READY=1
                else
                    read -r -p "    Login failed. Two-factor code (empty to skip): " MEGA_CODE </dev/tty || MEGA_CODE=""
                    if [ -n "$MEGA_CODE" ] && mega-login --auth-code="$MEGA_CODE" "$MEGA_EMAIL" "$MEGA_PASSWORD" >/dev/null 2>&1; then
                        MEGA_READY=1
                    fi
                fi
                unset MEGA_PASSWORD MEGA_CODE
                if [ "$MEGA_READY" = 1 ]; then info "Logged in to MEGA."; else warn "MEGA login failed: run 'mega-cmd' then 'login <email>' later."; fi
            fi
        else
            warn "Not logged in to MEGA: run 'mega-cmd' then 'login <email>' (needed for backups)."
        fi
    fi
fi

# ----------------------------------------------------------------------------------------------------------------
step "5/7 Restore data from MEGA"
if [ "$MEGA_READY" = 1 ]; then
    LATEST="$(mega-ls "${MEGA_BACKUP_DIR:-/ProphitBet-Backups}" 2>/dev/null | grep -E '^prophitbet-[0-9]{8}-[0-9]{6}\.tar\.gz$' | sort | tail -n 1 || true)"
    if [ -z "$LATEST" ]; then
        info "No backups found in MEGA:${MEGA_BACKUP_DIR:-/ProphitBet-Backups}."
    else
        HAS_LOCAL_DATA=0; [ -f storage/leagues/league_index.pkl ] && HAS_LOCAL_DATA=1
        DEFAULT=y; [ "$HAS_LOCAL_DATA" = 1 ] && DEFAULT=n
        if [ "$RESTORE" = yes ] || { [ "$RESTORE" = ask ] && ask "Restore $LATEST? (local data is moved aside first)" "$DEFAULT"; }; then
            scripts/restore_from_mega.sh "$LATEST"
        else
            info "Not restored. Restore later with: scripts/restore_from_mega.sh"
        fi
    fi
else
    info "Skipped (MEGA not available)."
fi

# ----------------------------------------------------------------------------------------------------------------
step "6/7 Launcher icon"
if [ "$DESKTOP" = 1 ]; then
    ENV_PREFIX=""
    [ "$VENV_DIR" != "$HOME/prophitbet-venv" ] && ENV_PREFIX="env PROPHITBET_VENV=\"$VENV_DIR\" "
    APPS_DIR="$HOME/.local/share/applications"
    mkdir -p "$APPS_DIR"
    cat > "$APPS_DIR/prophitbet.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=ProphitBet
Comment=AI soccer predictions
Exec=${ENV_PREFIX}"$INSTALL_DIR/prophitbet.sh"
Path=$INSTALL_DIR
Icon=$INSTALL_DIR/storage/icons/menu/ball.png
Terminal=false
Categories=Utility;
StartupNotify=true
EOF
    chmod +x "$APPS_DIR/prophitbet.desktop"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$APPS_DIR" 2>/dev/null || true
    info "Application menu: ProphitBet"

    DESKTOP_DIR="$(command -v xdg-user-dir >/dev/null 2>&1 && xdg-user-dir DESKTOP || echo "$HOME/Desktop")"
    if [ -d "$DESKTOP_DIR" ]; then
        cp "$APPS_DIR/prophitbet.desktop" "$DESKTOP_DIR/prophitbet.desktop"
        chmod +x "$DESKTOP_DIR/prophitbet.desktop"
        command -v gio >/dev/null 2>&1 && gio set "$DESKTOP_DIR/prophitbet.desktop" metadata::trusted true 2>/dev/null || true
        info "Desktop icon: $DESKTOP_DIR/prophitbet.desktop"
    fi
else
    info "Skipped (--no-desktop)."
fi

# ----------------------------------------------------------------------------------------------------------------
step "7/7 Automation schedule (Tuesday & Friday 17:30, local time)"
if [ "$CRON" = yes ] || { [ "$CRON" = ask ] && ask "Install the automation schedule?" y; }; then
    PYTHON_ENV=""
    [ "$VENV_DIR" != "$HOME/prophitbet-venv" ] && PYTHON_ENV="PROPHITBET_PYTHON=$VENV_DIR/bin/python "
    CRON_LINES="# ProphitBet automation (local time). Output: $INSTALL_DIR/storage/predictions/<date>/
# Tuesday 17:30: refresh results -> create missing plan leagues/models -> weekly retrain -> predict -> MEGA backup
30 17 * * 2 ${PYTHON_ENV}$INSTALL_DIR/scripts/run_automation.sh >> $INSTALL_DIR/storage/logs/cron.log 2>&1
# Friday 17:30: refresh results -> create missing plan leagues/models -> predict (no retraining) -> MEGA backup
30 17 * * 5 ${PYTHON_ENV}$INSTALL_DIR/scripts/run_automation.sh --skip-retrain >> $INSTALL_DIR/storage/logs/cron.log 2>&1"
    mkdir -p storage/logs
    { crontab -l 2>/dev/null | grep -v -e 'ProphitBet automation' -e 'run_automation.sh' -e '^# Tuesday 17:30' -e '^# Friday 17:30' || true; echo "$CRON_LINES"; } | crontab -
    info "Installed (crontab -l to view)."
else
    info "Skipped."
fi

printf '\n\033[1;32mDone.\033[0m Launch ProphitBet from the desktop icon or the application menu (or %s/prophitbet.sh).\n' "$INSTALL_DIR"
