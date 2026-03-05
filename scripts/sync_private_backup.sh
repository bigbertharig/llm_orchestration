#!/usr/bin/env bash
set -euo pipefail

PRIVATE_REPO="/home/bryan/llm_orchestration_private"

if [ ! -d "$PRIVATE_REPO/.git" ]; then
  echo "missing private repo: $PRIVATE_REPO" >&2
  exit 1
fi

"$PRIVATE_REPO/sync_from_public.sh"

if [ -n "$(git -C "$PRIVATE_REPO" status --porcelain)" ]; then
  git -C "$PRIVATE_REPO" add .
  git -C "$PRIVATE_REPO" commit -m "Sync private orchestration backup $(date +%Y-%m-%dT%H:%M:%S%z)"
  git -C "$PRIVATE_REPO" push
  echo "private backup updated and pushed"
else
  echo "private backup already up to date"
fi
