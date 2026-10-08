#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dual-dev.sh -- fast iteration boot.
#
# `dual-build.sh` bakes engine/tools/server into the image and ships ~11 GB; that costs minutes per
# code change.  This syncs the working tree to the peer (a few MB) and boots with source
# bind-mounts, so a change costs one `dual-down` + `dual-up` (~1 min, dominated by the 64 GB warm
# start and weight load).  Requires the repo at the same path on both boxes.  The image still
# provides the interpreter, CUDA, torch and Triton; only /app/{engine,tools,server} differ.
#
# Usage: scripts/dual-dev.sh          (same flags as dual-up.sh, e.g. --check)
# ---------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

info() { echo "--- $*"; }

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
PEER="${PEER:?set PEER in .env (user@host)}"

info "syncing engine/ tools/ server/ to $PEER"
for d in engine tools server; do
    rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
        "$ROOT/$d/" "$PEER:$ROOT/$d/"
done

info "booting with source mounts (DSV41_DEV_SOURCE=1)"
DSV41_DEV_SOURCE=1 exec "$ROOT/scripts/dual-up.sh" "$@"
