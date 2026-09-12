#!/usr/bin/env bash
# Mirror this checkout's CODE to the second Spark. Weights are scripts/replicate-checkpoint.sh's
# job and never travel through here.
#
# EP2 is not tolerant of version skew: both ranks run the same decode loop and must issue the
# same collectives in the same order, so a peer running last week's engine/model.py does not
# produce a wrong answer, it wedges the pair on a mismatched all-reduce. The peer is a plain
# file copy (not a git clone), so "git pull on both" is not available -- this is.
#
# Excluded, deliberately:
#   .env       box-specific (HOST/PORT/MASTER_ADDR) and holds a token; the peer's rank gets its
#              settings from the environment start.sh hands it over ssh, never from a file.
#   .venv      built per box, aarch64 wheels + CUDA; copying it is slower than it looks and
#              breaks the absolute paths baked into the venv's scripts.
#   models/logs  weights and run output: large, and per-box by nature.
#   results/     run output too -- EXCEPT results/trace-*/stats, which is not output at all but
#                an INPUT: coverage.json ranks the warm start's keep set, and PRUNE_KEEP masks the
#                router with it. Two ranks on different keep-sets mask different experts, route
#                differently, and desync on the first collective -- so the stats must be
#                byte-identical on both boxes and are synced deliberately (~1 MB).
#   .triton-cache  compiled Triton kernels, written by the container as root. Per-box by
#              nature (it keys on the local GPU and driver), and rsync cannot even read it
#              back as this user -- it failed with exit 23, partial transfer, before this
#              exclude existed.
#
# Usage: scripts/sync-peer.sh [--dry-run]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PEER="${PEER:-ryan@10.0.0.2}"
DEST="${DEST:-$ROOT}"
DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(--dry-run)

cd "$ROOT"
rsync -az --delete "${DRY[@]}" \
    --exclude '.git/' --exclude '.venv/' --exclude '.env' \
    --include '/results/' --include '/results/trace-*/' --include '/results/trace-*/stats/***' \
    --exclude '/results/**' \
    --exclude 'logs/' --exclude 'models/' --exclude '.triton-cache/' \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
    --itemize-changes \
    ./ "$PEER:$DEST/"
echo "synced $ROOT -> $PEER:$DEST"
