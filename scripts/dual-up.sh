#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# dual-up.sh -- bring up the EP2 pair in containers, one rank per Spark.
#
# Why a script and not `docker compose`: compose does not span hosts. The alternative is a
# compose file on each box plus an env file on each box, and EP2 is the worst possible place
# to have two copies of a config drift -- version or flag skew between the ranks does not
# produce a wrong answer, it wedges the pair on a mismatched collective. So the run flags are
# built ONCE here and applied to both ranks, and the only per-rank differences are the three
# that must differ: RANK, whether a port is bound, and each box's own RoCE GID index (which
# the entrypoint resolves locally, because the two boxes genuinely disagree -- see
# scripts/roce_gid.sh).
#
# What the container needs from the host for this to be fast:
#   --network host          the 10.0.0.0/24 rendezvous and NCCL_SOCKET_IFNAME are host links
#   --device /dev/infiniband + --cap-add IPC_LOCK + --ulimit memlock=-1
#                           verbs. Without these NCCL silently uses TCP and the per-collective
#                           cost goes ~60 us -> ~1.9 ms, which is 30x over the Gate G0 budget
#   --ipc host              one unified pool; see compose.yaml on why there is no mem_limit
#   /models bind mount      must be real local NVMe: the engine reads experts with O_DIRECT,
#                           which overlayfs and network filesystems will not give you
#
# Usage: scripts/dual-up.sh [--no-wait|--check]
#   --check   run every preflight on both boxes and exit without starting anything.
#             Worth having its own flag: the alternative way to find out that the peer is
#             short on memory or missing the image is to load ~100 GB on this box first and
#             discover it when the pair fails to rendezvous.
# Stop with: scripts/dual-down.sh
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

info() { echo "--- $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

# Caller-supplied env beats .env, the same way start.sh does it. Sourcing .env with `set -a`
# overwrites anything already exported, so `SPEC=0 scripts/dual-up.sh` would silently run with
# .env's SPEC=1 -- capture what the caller set, source, then put it back.
declare -A _CLI=()
for v in IMAGE PEER MASTER_ADDR MASTER_PORT PORT BIND_HOST MIN_FREE_GIB NAME0 NAME1 NCCL_SOCKET_IFNAME \
         MODELS_DIR MODEL_DIR MODEL_NAME SERVED_MODEL_NAME MAX_SEQ SPEC ARENA_GB TRACE_STATS \
         PRUNE_KEEP PRUNE_SELECT TRANSIENT_SLOTS KEEP_FREE_GB EXPERT_FORMAT EXTRA_FLAGS \
         DEFAULT_THINKING DEFAULT_EFFORT DSV41_DIST_TIMEOUT_S HEALTH_TIMEOUT_S; do
    [[ -n "${!v:-}" ]] && _CLI[$v]="${!v}"
done
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
for v in "${!_CLI[@]}"; do printf -v "$v" '%s' "${_CLI[$v]}"; export "${v?}"; done

IMAGE="${IMAGE:-deepseek-v41-flash-spark:local}"
PEER="${PEER:-}"
MASTER_PORT="${MASTER_PORT:-29611}"
PORT="${PORT:-8000}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"
NAME0="${NAME0:-deepseek-v41-ep2-rank0}"
NAME1="${NAME1:-deepseek-v41-ep2-rank1}"
IFACE="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-3600}"
# Where rank 0's API binds. Loopback by default: with --network host an open port is 510 GB of
# weights answering the whole LAN with no authentication of any kind. BIND_HOST=0.0.0.0 opts in to
# reaching it from another machine -- do that only on a network you trust, or put a reverse proxy
# with auth in front. BIND_HOST=192.168.11.206 (one interface) is the narrower middle ground.
# Falls back to HOST so .env stays the single place this is configured: HOST is what the native
# path (start.sh) and the entrypoint already call it, and having a container-only second name for
# the same thing meant editing .env had no effect on the pair. BIND_HOST still wins when set, for
# a one-off `BIND_HOST=0.0.0.0 scripts/dual-up.sh`.
BIND_HOST="${BIND_HOST:-${HOST:-127.0.0.1}}"
WAIT=true
CHECK_ONLY=false
case "${1:-}" in
    --no-wait) WAIT=false ;;
    --check)   CHECK_ONLY=true ;;
    "")        ;;
    *)         err "unknown argument '${1}' (expected --no-wait or --check)" ;;
esac

[[ -n "$PEER" ]] || err "PEER is required (e.g. PEER=ryan@10.0.0.2), set it in .env"
PEER_IP="${PEER#*@}"
MASTER_ADDR="${MASTER_ADDR:-$(ip route get "$PEER_IP" 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}' | head -1)}"
[[ -n "$MASTER_ADDR" ]] || err "cannot derive MASTER_ADDR toward $PEER_IP; set it in .env"

# The checkpoint directory on each HOST. MODEL_DIR in .env is a host path; inside every
# container the checkpoint is always /models/<name>, so only the directory name varies.
HOST_MODELS="${MODELS_DIR:-$(dirname "${MODEL_DIR:-/home/ryan/models/DeepSeek-V4.1-Flash}")}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_DIR:-DeepSeek-V4.1-Flash}")}"

ssh_peer() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" "$@"; }

# --- preflight, on BOTH boxes ----------------------------------------------------------
# A pair that boots half-short does not OOM, it wedges on the first collective, so every
# check here runs on both sides before either container starts.
check_box() {   # $1 = label, $2 = "" for local or the ssh target
    local label="$1" tgt="${2:-}" run avail
    run() { if [[ -z "$tgt" ]]; then bash -c "$1"; else ssh_peer "$1"; fi; }
    run "docker image inspect '$IMAGE' >/dev/null 2>&1" \
        || err "$label: image '$IMAGE' is missing. Build it there first: scripts/dual-build.sh"
    run "test -f '$HOST_MODELS/$MODEL_NAME/model.safetensors.index.json'" \
        || err "$label: no checkpoint at $HOST_MODELS/$MODEL_NAME (each box needs its OWN local copy -- O_DIRECT does not cross a network filesystem)"
    run "test -e /dev/infiniband/rdma_cm" || err "$label: /dev/infiniband missing -- is the RoCE driver loaded?"
    avail=$(run "awk '/^MemAvailable:/ {printf \"%d\", \$2/1048576}' /proc/meminfo")
    (( ${avail:-0} >= MIN_FREE_GIB )) \
        || err "$label: only ${avail:-?} GiB available (need >= $MIN_FREE_GIB)"
    info "$label: image ok, checkpoint ok, verbs ok, ${avail} GiB free"
}
check_box "rank 0 (local)" ""
check_box "rank 1 ($PEER)" "$PEER"

# Each box's own GID index, resolved here purely so a mismatch is a message rather than a
# QP failure 100 GB into the load. The containers resolve their own again at start.
g0=$("$ROOT/scripts/roce_gid.sh" "$MASTER_ADDR") || err "local RoCE GID unresolved for $MASTER_ADDR"
g1=$(ssh_peer "cd '$ROOT' && ./scripts/roce_gid.sh '$PEER_IP'") || err "peer RoCE GID unresolved for $PEER_IP"
info "RoCE: rank 0 = $g0   rank 1 = $g1   (rendezvous $MASTER_ADDR:$MASTER_PORT)"
if [[ "$BIND_HOST" != "127.0.0.1" && "$BIND_HOST" != "localhost" ]]; then
    info "WARNING: the API will bind $BIND_HOST:$PORT and this server has NO authentication."
fi

if [[ "$CHECK_ONLY" == true ]]; then
    info "--check: preflight passed on both boxes; nothing started"
    exit 0
fi

# Clear anything left from a previous run: an orphaned rank 1 would take the new head's
# broadcasts and both would wait forever.
docker rm -f "$NAME0" >/dev/null 2>&1 || true
ssh_peer "docker rm -f '$NAME1' >/dev/null 2>&1 || true" || true
mkdir -p "$ROOT/.triton-cache" "$ROOT/results"
ssh_peer "mkdir -p '$ROOT/.triton-cache' '$ROOT/results'" || true

# --- the flags both ranks share ---------------------------------------------------------
common_flags=(
    --network host
    --gpus all
    --device /dev/infiniband
    --cap-add IPC_LOCK
    --ipc host
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536
    --restart no
    --stop-timeout "${STOP_TIMEOUT:-30}"   # matches dual-down.sh; see the note there
    -v "$HOST_MODELS:/models"
    -v "$ROOT/results:/app/results"
    -v "$ROOT/.triton-cache:/app/.triton"
)
common_env=(
    -e WORLD_SIZE=2
    -e MASTER_ADDR="$MASTER_ADDR"
    -e MASTER_PORT="$MASTER_PORT"
    -e NCCL_SOCKET_IFNAME="$IFACE"
    -e GLOO_SOCKET_IFNAME="$IFACE"
    -e NCCL_IB_DISABLE=0
    -e DSV41_DIST_TIMEOUT_S="${DSV41_DIST_TIMEOUT_S:-600}"
    -e MODEL_DIR="/models/$MODEL_NAME"
    -e MAX_SEQ="${MAX_SEQ:-32768}"
    -e SPEC="${SPEC:-1}"
    -e MIN_FREE_GIB="$MIN_FREE_GIB"
    -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4.1-flash}"
    -e DEFAULT_THINKING="${DEFAULT_THINKING:-off}"
    -e DEFAULT_EFFORT="${DEFAULT_EFFORT:-75}"
)
# Tuning knobs: forwarded only when set, so an unset one keeps the image default.
for v in ARENA_GB TRACE_STATS PRUNE_KEEP PRUNE_SELECT TRANSIENT_SLOTS KEEP_FREE_GB EXPERT_FORMAT EXTRA_FLAGS; do
    [[ -n "${!v:-}" ]] && common_env+=(-e "$v=${!v}")
done
# Every DSV41_* knob in the caller's environment, forwarded verbatim to BOTH ranks. The engine's
# experiment switches all share that prefix (DSV41_ENGRAM_PINNED, DSV41_GRAPHS, DSV41_LUT,
# DSV41_GRAPH_SEGMENTS, ...) and there is no reason to grow this script a line per knob. Both
# ranks get the same value, which is the part that matters: a switch that changes numerics on one
# rank only would desync the pair rather than produce a slow answer.
while IFS='=' read -r k _; do
    [[ -n "$k" ]] && common_env+=(-e "$k=${!k}")
done < <(env | grep -E '^DSV41_[A-Z0-9_]+=' | sort)

# --- rank 0: the head, and the TCPStore server -------------------------------------------
# It goes first for the reason Gate G0 cost a bring-up to learn: rank 0 BINDS the rendezvous
# port and every other rank can only retry against it, so starting the peer first just burns
# its connect budget against a closed port and reports a "timeout" that looks like a dead
# fabric. HOST=127.0.0.1 keeps the API on loopback -- with --network host that is the host's
# loopback, so an open port here would be 510 GB of weights answering to the whole LAN.
info "starting rank 0 locally ($IMAGE)"
docker run -d --name "$NAME0" "${common_flags[@]}" "${common_env[@]}" \
    -e RANK=0 -e EP_LOCAL_IP="$MASTER_ADDR" \
    -e HOST="$BIND_HOST" -e PORT="$PORT" \
    "$IMAGE" >/dev/null

echo -n "--- waiting for the rendezvous port $MASTER_ADDR:$MASTER_PORT: "
for i in $(seq 1 120); do
    if ! docker ps --format '{{.Names}}' | grep -qx "$NAME0"; then
        echo "rank 0 exited"; docker logs --tail 40 "$NAME0"; exit 1
    fi
    if (exec 3<>"/dev/tcp/$MASTER_ADDR/$MASTER_PORT") 2>/dev/null; then echo "up after ${i}s"; break; fi
    sleep 1
    [[ $i -eq 120 ]] && { echo "TIMEOUT"; docker logs --tail 40 "$NAME0"; exit 1; }
done

# --- rank 1: the headless worker ---------------------------------------------------------
# --no-healthcheck because the image's HEALTHCHECK curls a socket this rank never binds; it
# would otherwise sit "unhealthy" forever while working perfectly.
info "starting rank 1 on $PEER"
ssh_peer "docker run -d --name '$NAME1' --no-healthcheck \
    $(printf '%q ' "${common_flags[@]}" "${common_env[@]}") \
    -e RANK=1 -e EP_LOCAL_IP='$PEER_IP' \
    '$IMAGE'" >/dev/null \
    || err "failed to start rank 1 on $PEER"

info "rank 0: docker logs -f $NAME0"
info "rank 1: ssh $PEER docker logs -f $NAME1"

if [[ "$WAIT" == false ]]; then
    echo "not waiting (--no-wait)."
    exit 0
fi

# Rank 0 binds its socket only after BOTH ranks have joined the process group and warmed
# their arenas, so this one wait covers the whole pair.
echo -n "--- waiting for /health (both ranks load ~18.5 GB of weights plus their arena half): "
for _ in $(seq 1 "$HEALTH_TIMEOUT_S"); do
    if ! docker ps --format '{{.Names}}' | grep -qx "$NAME0"; then
        echo "FAILED"; echo "--- rank 0 ---"; docker logs --tail 40 "$NAME0"
        echo "--- rank 1 ---"; ssh_peer "docker logs --tail 20 '$NAME1'" || true
        exit 1
    fi
    if curl -sf -o /dev/null "http://${BIND_HOST/0.0.0.0/127.0.0.1}:$PORT/health"; then
        echo "OK"
        curl -s "http://127.0.0.1:$PORT/health" | head -c 400; echo
        info "EP2 pair serving on http://$BIND_HOST:$PORT/v1"
        exit 0
    fi
    sleep 1
done
echo "TIMEOUT after ${HEALTH_TIMEOUT_S}s"; docker logs --tail 40 "$NAME0"; exit 1
