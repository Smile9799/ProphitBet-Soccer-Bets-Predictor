#!/bin/bash
# Back up ProphitBet data (leagues, trained models, tuning studies, predictions, configs) to MEGA.
#
# Usage:   scripts/backup_to_mega.sh
# Requires one of:
#   * MEGAcmd (https://mega.io/cmd), logged in once via the interactive shell: mega-cmd, then: login <email>
#   * rclone with a MEGA remote named "mega" (rclone config)
#
# Environment:
#   MEGA_BACKUP_DIR    remote folder (default: /ProphitBet-Backups)
#   MEGA_BACKUP_KEEP   number of backups kept remotely (default: 8)
#   RCLONE_REMOTE      rclone remote name (default: mega)
#
# Restore with scripts/restore_from_mega.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${MEGA_BACKUP_DIR:-/ProphitBet-Backups}"
KEEP="${MEGA_BACKUP_KEEP:-8}"
RCLONE_REMOTE="${RCLONE_REMOTE:-mega}"
LOG_FILE="$PROJECT_ROOT/storage/logs/backup.log"

mkdir -p "$PROJECT_ROOT/storage/logs"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

# Selecting the upload tool.
if command -v mega-put >/dev/null 2>&1 && mega-whoami >/dev/null 2>&1; then
    TOOL=megacmd
elif command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -qx "${RCLONE_REMOTE}:"; then
    TOOL=rclone
else
    log "ERROR Backup skipped: install MEGAcmd and log in once via 'mega-cmd' then 'login <email>' (or configure an rclone remote '${RCLONE_REMOTE}')."
    exit 2
fi

# Creating the archive.
cd "$PROJECT_ROOT"
ARCHIVE_NAME="prophitbet-$(date '+%Y%m%d-%H%M%S').tar.gz"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PATHS=()
for path in storage/leagues storage/predictions storage/configs; do
    [ -e "$path" ] && PATHS+=("$path")
done
tar --exclude='__pycache__' -czf "$WORK_DIR/$ARCHIVE_NAME" "${PATHS[@]}"
SIZE="$(du -h "$WORK_DIR/$ARCHIVE_NAME" | cut -f1)"

# Uploading and pruning old backups (archive names sort chronologically).
if [ "$TOOL" = megacmd ]; then
    # -c creates the remote folder if needed; the upload completes before mega-put returns.
    mega-put -c "$WORK_DIR/$ARCHIVE_NAME" "$REMOTE_DIR/"
    mapfile -t BACKUPS < <(mega-ls "$REMOTE_DIR" | grep -E '^prophitbet-[0-9]{8}-[0-9]{6}\.tar\.gz$' | sort)
else
    rclone copy "$WORK_DIR/$ARCHIVE_NAME" "${RCLONE_REMOTE}:${REMOTE_DIR#/}"
    mapfile -t BACKUPS < <(rclone lsf "${RCLONE_REMOTE}:${REMOTE_DIR#/}" | grep -E '^prophitbet-[0-9]{8}-[0-9]{6}\.tar\.gz$' | sort)
fi

REMOVE_COUNT=$(( ${#BACKUPS[@]} - KEEP ))
for (( i = 0; i < REMOVE_COUNT; i++ )); do
    if [ "$TOOL" = megacmd ]; then
        mega-rm "$REMOTE_DIR/${BACKUPS[$i]}"
    else
        rclone deletefile "${RCLONE_REMOTE}:${REMOTE_DIR#/}/${BACKUPS[$i]}"
    fi
    log "INFO Removed old backup ${BACKUPS[$i]}"
done

log "INFO Uploaded $ARCHIVE_NAME ($SIZE) to MEGA:$REMOTE_DIR via $TOOL; keeping the latest $KEEP backups."
