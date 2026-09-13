#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh -- serve DeepSeek-V4.1-Flash on ONE DGX Spark (the upstream launcher).
#
# THIS FORK RUNS TWO BOXES. If you cloned it to serve the pair, you want
#   scripts/dual-up.sh      (start both ranks)   /   scripts/dual-down.sh (stop them)
# and not this script, which starts a single-box server on this machine only and knows
# nothing about the peer. It is kept because the single-box path still works and every
# number in RESULTS.md came from it; it is guarded below so nobody runs it by accident.
#
#   ./start.sh                 # start, wait for /health, print the endpoint
#   ./start.sh --no-wait       # start and return immediately (tail logs yourself)
#   PORT=8001 ./start.sh       # environment beats .env
#   ARENA_GB=60 ./start.sh     # pin the resident expert arena instead of auto
#
# This is not SGLang and not a container: it launches `server/app.py --engine v41`
# (engine/v41_engine.py) with nohup, writes logs/server.pid and logs to
# logs/server.log. Stop it with ./stop.sh.
#
# Why the health wait is 20 minutes: the warm start fills the resident FP4
# expert arena from the checkpoint, which is ~80 GB of NVMe reads before the
# HTTP socket is even bound. A server that is "not up yet" at minute 6 is normal.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

err()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "--- $*"; }

WAIT=true
for arg in "$@"; do
    case "$arg" in
        --no-wait) WAIT=false ;;
        -h|--help) sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "unknown argument '$arg' (only --no-wait)" ;;
    esac
done

# --- guard: this fork's default is the two-box container path ----------------
# The confusing case is a fresh clone: start.sh sits in the repo root, looks like the way
# in, and silently gives you a single-box server with none of the pair's residency. Say so
# and require an explicit opt-in rather than leaving it to the README.
if [[ "${DSV41_SINGLE_BOX:-0}" != "1" ]]; then
    cat >&2 <<'GUARD'
ERROR: this is the UPSTREAM single-box launcher, and this fork serves two DGX Sparks.

  Two boxes (what this fork is for):
      scripts/dual-up.sh            start both ranks in containers
      scripts/dual-down.sh          stop them
      scripts/dual-build.sh         build the image and ship it to the peer

  One box, on purpose (upstream behaviour, ~25% of experts resident instead of 28.4%,
  no EP2, no cross-rank guard):
      DSV41_SINGLE_BOX=1 ./start.sh

See the top of README.md for the two-box quick start.
GUARD
    exit 2
fi

# Environment wins over .env, so `PORT=8001 ./start.sh` works.
declare -A _CLI=()
for v in MODEL_DIR PYTHON SERVED_MODEL_NAME HOST PORT MAX_SEQ ARENA_GB \
         TRACE_STATS DEFAULT_THINKING DEFAULT_EFFORT SPEC EXTRA_FLAGS PRUNE_KEEP PRUNE_SELECT TRANSIENT_SLOTS KEEP_FREE_GB \
         EXPERT_FORMAT WORLD_SIZE PEER MASTER_ADDR MASTER_PORT; do
    [[ -n "${!v:-}" ]] && _CLI[$v]="${!v}"
done
# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
for v in "${!_CLI[@]}"; do printf -v "$v" '%s' "${_CLI[$v]}"; done

MODEL_DIR="${MODEL_DIR:-./models/DeepSeek-V4.1-Flash}"
PYTHON="${PYTHON:-python3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4.1-flash}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_SEQ="${MAX_SEQ:-32768}"
ARENA_GB="${ARENA_GB:-}"                 # empty = size from free GPU memory
# Empty = auto: take the newest results/trace-*/stats/coverage.json (see below).
TRACE_STATS="${TRACE_STATS:-}"
DEFAULT_THINKING="${DEFAULT_THINKING:-off}"
DEFAULT_EFFORT="${DEFAULT_EFFORT:-75}"
SPEC="${SPEC:-1}"
# Whitespace-separated extra flags for server/app.py (A/B runs). Empty by default.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1200}"
# --- EP2 (docs/dual-spark-plan.md) -----------------------------------------
# WORLD_SIZE=2 splits the routed experts across two GB10 boxes over the
# ConnectX-7 link (rank 1 = a headless worker started on PEER over ssh).
# Everything else -- weights, KV, sampling -- is replicated, so the two ranks
# stay in lockstep on every collective. Unset/1 = the single-box recipe,
# unchanged. Requirements on BOTH boxes: same checkout at the same path, a
# .venv, the full checkpoint on LOCAL NVMe (O_DIRECT never crosses the
# network), passwordless ssh head -> peer, RoCE up. WORLD_SIZE/PEER reach the
# engine through the process environment (RANK/WORLD_SIZE/MASTER_*), which is
# exactly what engine/dist.py:EPDistributed reads.
WORLD_SIZE="${WORLD_SIZE:-1}"
PEER="${PEER:-}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29611}"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"

# --- sanity ---------------------------------------------------------------
[[ -d "$MODEL_DIR" ]] || err "model dir not found: $MODEL_DIR (set MODEL_DIR in .env)"
[[ -f "$MODEL_DIR/tokenizer.json" ]] || err "$MODEL_DIR has no tokenizer.json"
[[ -f "$MODEL_DIR/encoding/encoding.py" ]] || err "$MODEL_DIR has no encoding/encoding.py"
PYTHON_BIN="$(command -v -- "$PYTHON" 2>/dev/null || true)"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || err "interpreter not found or not executable: $PYTHON (set PYTHON in .env)"
PYTHON="$PYTHON_BIN"
[[ -f server/app.py ]] || err "server/app.py missing -- run this from a full checkout"
case "$DEFAULT_THINKING" in on|off) ;; *) err "DEFAULT_THINKING must be on|off (got '$DEFAULT_THINKING')" ;; esac
case "$SPEC" in 0|1) ;; *) err "SPEC must be 0|1 (got '$SPEC')" ;; esac
case "$WORLD_SIZE" in 1|2) ;; *) err "WORLD_SIZE must be 1 or 2 (the EP2 skeleton is world-2 only)" ;; esac
if [[ "$WORLD_SIZE" != "1" ]]; then
    [[ -n "$PEER" ]] || err "WORLD_SIZE=2 needs PEER=<ssh target of the second spark> (e.g. user@10.0.0.2)"
    command -v ssh >/dev/null 2>&1 || err "ssh not found but WORLD_SIZE=2"
fi

# --- guard: is the port already taken? ------------------------------------
port_busy() {
    if command -v ss >/dev/null 2>&1; then
        [[ -n "$(ss -ltnH "sport = :$PORT" 2>/dev/null)" ]]
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null
    fi
}
if port_busy; then
    holder=""
    command -v ss >/dev/null 2>&1 && holder="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | tr -s ' ')"
    err "port $PORT is already in use${holder:+ by: $holder}.
     If it is our own server, ./stop.sh. Otherwise pick another PORT."
fi

# --- guard: does something else own the box's memory? ---------------------
# MemAvailable is the "available" column of `free -g`. This model needs the
# unified pool essentially to itself: the resident expert arena is sized from
# what is free at load time, so starting next to another server does not OOM,
# it silently gives us a tiny arena and a NVMe-bound 1 tok/s server -- or wedges
# the driver with no OOM and no logs. Refuse instead.
if [[ -r /proc/meminfo ]]; then
    avail_gib=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
    if (( avail_gib < MIN_FREE_GIB )); then
        echo "ERROR: only ${avail_gib} GiB available (need >= ${MIN_FREE_GIB} GiB)." >&2
        echo "     Biggest resident processes:" >&2
        ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -6 |
            awk 'NR==1{print "       PID      RSS_GB  COMMAND"; next} {printf "       %-8s %-7.1f %s\n", $1, $2/1048576, $3}' >&2
        if command -v docker >/dev/null 2>&1 && [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
            echo "     Containers are running and may be holding the unified pool:" >&2
            docker ps --format '       {{.Names}}\t{{.Image}}' 2>/dev/null >&2
            echo "     Stop the one that owns the GPU:  docker stop <container>" >&2
        else
            echo "     Stop whatever holds the pool (another inference server, a container)" >&2
            echo "     and wait for MemAvailable to recover." >&2
        fi
        exit 1
    fi
else
    echo "WARNING: no /proc/meminfo -- skipping the memory guard (not a Linux box?)" >&2
fi

# --- guard: EP2 pair readiness ----------------------------------------------
# The peer's arena warms before ITS socket-level participation matters, and a
# pair that boots half-short wedges on the first collective instead of OOMing
# loudly -- so run the same memory guard there over ssh, derive MASTER_ADDR
# from the route to the peer if .env did not set it, and clear any orphaned
# rank-1 worker (a zombie peer would steal the new head's broadcasts).
peer_avail_gib() {
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
        'awk "/^MemAvailable:/ {printf \"%d\", \$2/1048576}" /proc/meminfo' 2>/dev/null
}
if [[ "$WORLD_SIZE" != "1" ]]; then
    MASTER_ADDR="${MASTER_ADDR:-$(ip route get "${PEER#*@}" 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}' | head -1)}"
    [[ -n "$MASTER_ADDR" ]] || err "cannot derive MASTER_ADDR toward ${PEER#*@}; set MASTER_ADDR in .env"
    pa=$(peer_avail_gib) || err "cannot ssh to PEER=$PEER (passwordless ssh is a dual-spark prerequisite)"
    if (( ${pa:-0} < MIN_FREE_GIB )); then
        err "peer $PEER has only ${pa:-?} GiB available (need >= ${MIN_FREE_GIB}); a half-booted EP2 pair hangs, it does not fail"
    fi
    ssh -o BatchMode=yes "$PEER" "pkill -f 'python.*server/app\.p[y]' 2>/dev/null; rm -f '$LOG_DIR/server_peer.pid'" || true
    # RoCE GID indices are NOT the same on the two boxes (5 here, 6 on the peer -- the peer's
    # index 5 is an empty slot), and a wrong index does not fall back to sockets: NCCL fails the
    # QP transition with "ibv_modify_qp failed with 61 ... local GID ::" and the pair never
    # forms. Gate G0 spent a bring-up on exactly this. Each rank resolves its own; see
    # scripts/roce_gid.sh, and set NCCL_IB_HCA/NCCL_IB_GID_INDEX by hand only to override.
    if [[ -z "${NCCL_IB_GID_INDEX:-}" ]]; then
        read -r HCA0 GID0 < <("$SCRIPT_DIR/scripts/roce_gid.sh" "$MASTER_ADDR") \
            || err "cannot resolve the local RoCE GID for $MASTER_ADDR (scripts/roce_gid.sh)"
        read -r HCA1 GID1 < <(ssh -o BatchMode=yes "$PEER" "cd '$SCRIPT_DIR' && ./scripts/roce_gid.sh '${PEER#*@}'") \
            || err "cannot resolve the peer's RoCE GID for ${PEER#*@}; is the checkout synced? (scripts/sync-peer.sh)"
    else
        HCA0="${NCCL_IB_HCA:-}"; GID0="$NCCL_IB_GID_INDEX"
        HCA1="${NCCL_IB_HCA:-}"; GID1="$NCCL_IB_GID_INDEX"
        info "EP2: using the NCCL_IB_GID_INDEX override ($GID0) for BOTH ranks -- verify it exists on each box"
    fi
    info "EP2: rank 0 here, rank 1 on $PEER (rendezvous $MASTER_ADDR:$MASTER_PORT, peer MemAvailable ${pa} GiB)"
    info "EP2: RoCE rank0=${HCA0}/gid${GID0}  rank1=${HCA1}/gid${GID1}"
fi

# --- guard: are we already running? ---------------------------------------
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    err "server already running (pid $(cat "$PID_FILE")). ./stop.sh first."
fi

# --- assemble the command -------------------------------------------------
FLAGS=(
    --model-dir "$MODEL_DIR"
    --host "$HOST" --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --default-thinking "$DEFAULT_THINKING"
    --default-effort "$DEFAULT_EFFORT"
    --engine v41
    --max-seq "$MAX_SEQ"
)
[[ -n "$ARENA_GB" ]] && FLAGS+=(--arena-gb "$ARENA_GB")
[[ "$SPEC" == "0" ]] && FLAGS+=(--no-spec)
# Pruned all-resident mode (RESULTS.md v0.2.0-wip): PRUNE_KEEP=0.31 keeps the top 31 % experts per
# layer routable and resident; pair it with ARENA_GB=90.5 TRANSIENT_SLOTS=16 KEEP_FREE_GB=10 on a
# 128 GB box. Unset = the full model with expert streaming.
EK="{"
[[ -n "${PRUNE_KEEP:-}" ]] && EK="$EK\"prune_keep\": $PRUNE_KEEP,"
# EXPERT_FORMAT=cb3 packs the resident arena into the 3-bit per-row codebook format (14.45 MB per
# expert instead of 18.80), so the same 90.5 GB holds ~40.8 % of all routed experts instead of
# 31.3 %; pair it with PRUNE_KEEP=0.40. Warm start pays the packing (see NOTES 2026-09-11).
[[ -n "${EXPERT_FORMAT:-}" ]] && EK="$EK\"expert_format\": \"$EXPERT_FORMAT\","
[[ -n "${PRUNE_SELECT:-}" ]] && EK="$EK\"prune_select\": \"$PRUNE_SELECT\","
[[ -n "${TRANSIENT_SLOTS:-}" ]] && EK="$EK\"transient_slots\": $TRANSIENT_SLOTS,"
[[ -n "${KEEP_FREE_GB:-}" ]] && EK="$EK\"keep_free_gb\": $KEEP_FREE_GB,"
EK="${EK%,}}"
[[ "$EK" != "{}" ]] && FLAGS+=(--engine-kwargs "$EK")

# Which coverage.json ranks the warm start. Without one the arena is filled in
# (layer, expert) index order, which is a measurably worse hot set. Trace
# directories carry a name and a date (results/trace-full-YYYYMMDD/), so when
# TRACE_STATS is unset -- or points at something that is not there -- take the
# newest results/trace-*/stats/coverage.json rather than nothing.
newest_trace_stats() {
    local c
    c=$(ls -1d results/trace-*/stats/coverage.json 2>/dev/null | sort | tail -1 || true)
    [[ -n "$c" ]] && echo "$c"
}
TRACE_USED=""
if [[ -n "$TRACE_STATS" ]]; then
    if [[ -f "$TRACE_STATS" ]]; then
        TRACE_USED="$TRACE_STATS"
    else
        TRACE_USED="$(newest_trace_stats)"
        [[ -n "$TRACE_USED" ]] && info "trace stats $TRACE_STATS missing; using $TRACE_USED instead"
    fi
else
    TRACE_USED="$(newest_trace_stats)"
fi
[[ -z "$TRACE_USED" ]] && info "no results/trace-*/stats/coverage.json -- warm start will use index order"
[[ -n "$TRACE_USED" ]] && FLAGS+=(--trace-stats "$TRACE_USED")
# shellcheck disable=SC2206
[[ -n "$EXTRA_FLAGS" ]] && FLAGS+=($EXTRA_FLAGS)

mkdir -p "$LOG_DIR"
info "model=$MODEL_DIR  max_seq=$MAX_SEQ  arena=${ARENA_GB:-auto}  spec=$SPEC  thinking=$DEFAULT_THINKING/$DEFAULT_EFFORT  trace=${TRACE_USED:-none}"
info "log: $LOG_FILE"

: > "$LOG_FILE"
# EP2 env for rank 0 itself: EPDistributed reads RANK/WORLD_SIZE/MASTER_* from the
# environment; at WORLD_SIZE=1 nothing is added and the command is what it always was.
EPEVN=()
if [[ "$WORLD_SIZE" != "1" ]]; then
    # Pin both ranks onto the CX7 200G interface. Without this NCCL may bind WiFi
    # (192.168.x) and hang at init_process_group with empty logs -- Gate G0 failure mode.
    EPEVN=(
        RANK=0 WORLD_SIZE="$WORLD_SIZE" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT"
        NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
        GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp1s0f1np1}"
        NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
        TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        TORCH_NCCL_BLOCKING_WAIT=1
        DSV41_DIST_TIMEOUT_S="${DSV41_DIST_TIMEOUT_S:-600}"
    )
    # `if`, not `[[ ]] && ...`: under `set -e` a false test as the last statement of this block
    # would exit the script instead of just skipping the append.
    if [[ -n "$HCA0" ]]; then EPEVN+=(NCCL_IB_HCA="$HCA0"); fi
    if [[ -n "$GID0" ]]; then EPEVN+=(NCCL_IB_GID_INDEX="$GID0"); fi
fi
nohup env ${EPEVN[@]+"${EPEVN[@]}"} "$PYTHON" server/app.py "${FLAGS[@]}" >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
info "started pid $SERVER_PID"

# rank 1: same checkout, same flags, --headless (binds no port). Its engine load is
# the mirror of the head's (same ~63 s of weights + its own half of the arena warm
# start), and start.sh's health wait doubles as the wait for the rendezvous: the
# head never binds the socket until its constructor -- including process-group init
# and the dummy combine -- is through, which needs BOTH ranks present.
if [[ "$WORLD_SIZE" != "1" ]]; then
    FLAGS_Q=$(printf '%q ' "${FLAGS[@]}")
    NCCL_IF="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
    GLOO_IF="${GLOO_SOCKET_IFNAME:-enp1s0f1np1}"
    PEER_PID=$(ssh -o BatchMode=yes "$PEER" \
        "cd '$SCRIPT_DIR' && mkdir -p logs && \
         RANK=1 WORLD_SIZE='$WORLD_SIZE' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' \
         NCCL_SOCKET_IFNAME='$NCCL_IF' GLOO_SOCKET_IFNAME='$GLOO_IF' NCCL_IB_DISABLE=0 \
         ${HCA1:+NCCL_IB_HCA='$HCA1'} ${GID1:+NCCL_IB_GID_INDEX='$GID1'} \
         TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1 \
         DSV41_DIST_TIMEOUT_S='${DSV41_DIST_TIMEOUT_S:-600}' \
         nohup '$PYTHON' server/app.py $FLAGS_Q --headless >> logs/server_peer.log 2>&1 & echo \$!") \
        || err "failed to start rank 1 on $PEER -- check its $SCRIPT_DIR/logs/server_peer.log"
    [[ -n "$PEER_PID" ]] && echo "$PEER_PID" > "$LOG_DIR/server_peer.pid"
    info "rank 1 started on $PEER (pid $PEER_PID; watch it: ssh $PEER tail -f $SCRIPT_DIR/logs/server_peer.log)"
fi

if [[ "$WAIT" == false ]]; then
    echo "not waiting (--no-wait). Watch it come up with: tail -f $LOG_FILE"
    exit 0
fi

# --- wait for /health -----------------------------------------------------
# The socket is bound only after the engine is constructed, so this loop is
# mostly watching an 80 GB warm start, not an HTTP handshake.
echo -n "waiting for http://$HOST:$PORT/health (up to $((HEALTH_TIMEOUT_S / 60)) min, warm start reads ~80 GB from NVMe): "
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
while :; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "--- last 30 log lines ---" >&2
        tail -n 30 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        if [[ "$WORLD_SIZE" != "1" ]]; then
            echo "--- rank 1 on $PEER, last 15 log lines ---" >&2
            ssh -o BatchMode=yes "$PEER" "tail -n 15 '$SCRIPT_DIR/logs/server_peer.log'" 2>/dev/null >&2
            ssh -o BatchMode=yes "$PEER" "pkill -f 'python.*server/app\.p[y]' 2>/dev/null" || true
        fi
        err "server exited during startup (see $LOG_FILE)"
    fi
    if curl -sf -o /dev/null "http://$HOST:$PORT/health"; then
        echo " up"
        break
    fi
    if (( $(date +%s) >= deadline )); then
        echo
        echo "--- last 30 log lines ---" >&2
        tail -n 30 "$LOG_FILE" >&2
        err "no /health after ${HEALTH_TIMEOUT_S}s. It may still be warming up: watch $LOG_FILE, or ./stop.sh."
    fi
    echo -n "."
    sleep 5
done

curl -sf "http://$HOST:$PORT/health" && echo
cat <<MSG

  endpoint  http://$HOST:$PORT/v1
  model     $SERVED_MODEL_NAME
  logs      $LOG_FILE
  bench     python3 bench/bench.py --workload prose --runs 3 --out results/prose.json
  stop      ./stop.sh
MSG
