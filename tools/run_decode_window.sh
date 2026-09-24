#!/usr/bin/env bash
# One maintenance window for the decode projection work (docs/decode-projection-fusion.md).
# Serving must be stopped on BOTH nodes. Order:
#   1. snapshot engine/ tools/ server/ to the same path on both nodes (the image holds older code)
#   2. single-node unit tests + microbenchmark on this node's GPU
#   3. step 0: decode timeline with the current .env config (bench_decode_timeline_tp.py)
#   4. full-engine A/B per candidate, then all together (bench_decode_projection_tp.py)
# A failing A/B is recorded and the window continues; the summary at the end lists outcomes.
#
#   GATE_IMAGE=$(docker image inspect -f '{{.Id}}' deepseek-v41-flash-spark:local) \
#   bash tools/run_decode_window.sh [experiment ...]      # default: prune-miss merged act-qdq block-n all
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
image=${GATE_IMAGE:?set GATE_IMAGE to an image id present on both nodes}
PEER=$(sed -n 's/^PEER=//p' .env | tail -1)
[[ -n "$PEER" ]] || { echo 'PEER is not set in .env' >&2; exit 1; }
for name in deepseek-v41-ep2-rank0; do
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "Stop serving first ($name is present)" >&2; exit 1
    fi
done
if ssh -o BatchMode=yes "$PEER" 'docker ps --format "{{.Names}}" | grep -q deepseek-v41-ep2'; then
    echo "Stop serving on $PEER first" >&2; exit 1
fi
free_gib=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if (( free_gib < 90 )); then
    echo "Only ${free_gib} GiB available here; the pool has not been released yet" >&2; exit 1
fi
[[ $# -gt 0 ]] && experiments=("$@") || experiments=(prune-miss merged act-qdq block-n all)

stamp=$(date +%Y%m%d-%H%M)
out="results/decode-window-$stamp"
snap="${SNAPSHOT_DIR:-$HOME/gate-src/decode-window-$stamp}"
mkdir -p "$out" "$snap"
ssh -o BatchMode=yes "$PEER" "mkdir -p '$snap' '$ROOT/$out'"
for directory in engine tools server; do
    rsync -a --delete --exclude __pycache__ "$directory/" "$snap/$directory/"
    rsync -a --delete --exclude __pycache__ "$directory/" "$PEER:$snap/$directory/"
done
( cd "$snap" && find engine tools server -type f \( -name '*.py' -o -name '*.cu' -o -name '*.cpp' \) | sort | xargs sha256sum ) > "$out/source.sha256"
ssh -o BatchMode=yes "$PEER" "cd '$snap' && find engine tools server -type f \( -name '*.py' -o -name '*.cu' -o -name '*.cpp' \) | sort | xargs sha256sum" > "$out/source-peer.sha256"
cmp -s "$out/source.sha256" "$out/source-peer.sha256" || { echo 'snapshot differs between nodes' >&2; exit 1; }

declare -A result
local_run() {
    docker run --rm --gpus all --ipc host \
        -v "$snap/engine:/app/engine:ro" -v "$snap/tools:/app/tools:ro" -v "$snap/server:/app/server:ro" \
        -v "$ROOT/.triton-cache:/app/.triton" -w /app --entrypoint python3 "$image" "$@"
}
echo "== unit tests"
if local_run -m unittest engine.test_prune_miss_fused tools.test_fp8_decode_tiles tools.test_fp8_act_qdq \
        > "$out/unit.log" 2>&1; then result[unit]=pass; else result[unit]=FAIL; fi
tail -3 "$out/unit.log"
echo "== microbenchmark"
if local_run tools/bench_decode_projection_micro.py > "$out/micro.log" 2>&1; then
    result[micro]=done; else result[micro]=FAIL; fi

gate() {   # gate <label> <script> [args...]
    local label=$1 script=$2; shift 2
    mkdir -p "$out/$label"; ssh -o BatchMode=yes "$PEER" "mkdir -p '$ROOT/$out/$label'"
    echo "== $label"
    if GATE_IMAGE="$image" GATE_LOG_DIR="$out/$label" GATE_SOURCE_ROOT="$snap" \
            bash tools/run_two_node_gate.sh "$script" "$@" --out "/app/$out/$label"; then
        result[$label]=pass
    else
        result[$label]=FAIL
    fi
    grep -h -E 'DECODE_PROJ_SUMMARY|DECODE_PROJ_(PASS|FAIL)' "$out/$label/rank0.log" 2>/dev/null | tail -2
}
gate timeline bench_decode_timeline_tp.py
for exp in "${experiments[@]}"; do
    gate "ab-$exp" bench_decode_projection_tp.py --experiment "$exp"
done

echo "== summary ($out)"
for key in "${!result[@]}"; do printf '%-16s %s\n' "$key" "${result[$key]}"; done | sort | tee "$out/summary.txt"
echo "Timeline analysis: python3 tools/bench_decode_timeline_analysis.py $out/timeline/rank0.json --trim-edges"
echo "Restart serving when done."
