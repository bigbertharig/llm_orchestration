#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/bryan/llm_orchestration"
SOURCE_BRANCH="${1:-feature/disaster-map-prep-suite}"
SOURCE_SUBDIR="disaster-map-prep-suite"
TARGET_DIR="/home/bryan/llm_orchestration/shared/plans/arms/disaster-map-prep-suite"

if [ ! -d "$REPO_ROOT/.git" ]; then
  echo "ERROR: repo not found at $REPO_ROOT" >&2
  exit 1
fi

echo "Fetching origin/$SOURCE_BRANCH ..."
git -C "$REPO_ROOT" fetch origin "$SOURCE_BRANCH" --prune

if ! git -C "$REPO_ROOT" rev-parse --verify "origin/$SOURCE_BRANCH" >/dev/null 2>&1; then
  echo "ERROR: remote branch origin/$SOURCE_BRANCH not found" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "Exporting $SOURCE_SUBDIR from origin/$SOURCE_BRANCH ..."
if ! git -C "$REPO_ROOT" archive "origin/$SOURCE_BRANCH" "$SOURCE_SUBDIR" | tar -x -C "$TMP_DIR"; then
  echo "ERROR: failed to export $SOURCE_SUBDIR from origin/$SOURCE_BRANCH" >&2
  exit 1
fi

if [ ! -f "$TMP_DIR/$SOURCE_SUBDIR/plan.md" ]; then
  echo "ERROR: exported plan is missing plan.md ($SOURCE_SUBDIR)" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

echo "Syncing into $TARGET_DIR ..."
rsync -a --delete "$TMP_DIR/$SOURCE_SUBDIR/" "$TARGET_DIR/"

echo "Done. Synced from origin/$SOURCE_BRANCH:$SOURCE_SUBDIR"
