#!/bin/bash
# Restore ProphitBet data from a MEGA backup made by scripts/backup_to_mega.sh.
#
# Usage:   scripts/restore_from_mega.sh [backup-name.tar.gz]   (default: the latest backup)
# The current storage/leagues, storage/predictions and storage/configs are first moved to
# storage/pre-restore-<timestamp>/ so nothing is lost. Close the GUI before restoring.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${MEGA_BACKUP_DIR:-/ProphitBet-Backups}"
RCLONE_REMOTE="${RCLONE_REMOTE:-mega}"
cd "$PROJECT_ROOT"

if command -v mega-get >/dev/null 2>&1 && mega-whoami >/dev/null 2>&1; then
    TOOL=megacmd
    list() { mega-ls "$REMOTE_DIR"; }
elif command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -qx "${RCLONE_REMOTE}:"; then
    TOOL=rclone
    list() { rclone lsf "${RCLONE_REMOTE}:${REMOTE_DIR#/}"; }
else
    echo "Install MEGAcmd and run 'mega-login <email>' (or configure an rclone remote '${RCLONE_REMOTE}')." >&2
    exit 2
fi

BACKUP="${1:-$(list | grep -E '^prophitbet-[0-9]{8}-[0-9]{6}\.tar\.gz$' | sort | tail -n 1)}"
if [ -z "$BACKUP" ]; then
    echo "No backups found in MEGA:$REMOTE_DIR" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
echo "Downloading $BACKUP ..."
if [ "$TOOL" = megacmd ]; then
    mega-get "$REMOTE_DIR/$BACKUP" "$WORK_DIR/"
else
    rclone copy "${RCLONE_REMOTE}:${REMOTE_DIR#/}/$BACKUP" "$WORK_DIR"
fi

SAFE_DIR="storage/pre-restore-$(date '+%Y%m%d-%H%M%S')"
mkdir -p "$SAFE_DIR"
for path in storage/leagues storage/predictions storage/configs; do
    [ -e "$path" ] && mv "$path" "$SAFE_DIR/"
done

tar -xzf "$WORK_DIR/$BACKUP"
echo "Restored $BACKUP. Previous data was moved to $SAFE_DIR."
