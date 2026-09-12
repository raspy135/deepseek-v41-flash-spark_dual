#!/usr/bin/env bash
# dual-down.sh -- stop the EP2 pair and wait for the unified pool to come back on BOTH boxes.
#
# Order matters a little: rank 0 first, so its shutdown broadcast (server/app.py) reaches the
# headless worker and that rank exits on its own instead of sitting in broadcast_request until
# the process-group timeout. The explicit rank-1 removal afterwards is the backstop for when
# the head died badly and never sent it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
PEER="${PEER:-}"
NAME0="${NAME0:-deepseek-v41-ep2-rank0}"
NAME1="${NAME1:-deepseek-v41-ep2-rank1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"

echo "--- stopping rank 0 ($NAME0)"
docker stop -t 180 "$NAME0" >/dev/null 2>&1 || true
docker rm -f "$NAME0" >/dev/null 2>&1 || true

if [[ -n "$PEER" ]]; then
    echo "--- stopping rank 1 on $PEER ($NAME1)"
    ssh -o BatchMode=yes "$PEER" "docker stop -t 180 '$NAME1' >/dev/null 2>&1 || true; docker rm -f '$NAME1' >/dev/null 2>&1 || true" || true
fi

wait_mem() {   # $1 = label, $2 = "" local or ssh target
    local label="$1" tgt="${2:-}" a
    echo -n "--- waiting for $label to return >= ${MIN_FREE_GIB} GiB: "
    for _ in $(seq 1 60); do
        if [[ -z "$tgt" ]]; then
            a=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
        else
            a=$(ssh -o BatchMode=yes "$tgt" 'awk "/^MemAvailable:/ {printf \"%d\", \$2/1048576}" /proc/meminfo' 2>/dev/null)
        fi
        if [[ -n "$a" ]] && (( a >= MIN_FREE_GIB )); then echo "OK, ${a} GiB"; return 0; fi
        sleep 3
    done
    echo "WARNING: still at ${a:-?} GiB -- do not start another server yet" >&2
    return 1
}
rc=0
wait_mem "this box" "" || rc=1
[[ -n "$PEER" ]] && { wait_mem "$PEER" "$PEER" || rc=1; }
exit $rc
