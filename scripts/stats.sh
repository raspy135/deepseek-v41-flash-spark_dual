#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# stats.sh -- read the engine's own counters out of a running server.
#
# Everything here comes from `x_engine_stats` on a normal completion, so the cheap way to
# read it is a 1-token request: the counters that matter (miss_rate above all) are
# CUMULATIVE since the server booted, not per-request, so the probe's own prompt barely
# registers and you get the running totals for all the real traffic.
#
# Usage:
#   scripts/stats.sh              summary: miss rate, speed, expert residency
#   scripts/stats.sh miss         just the pruning miss rate
#   scripts/stats.sh layers       worst layers for pruning misses
#   scripts/stats.sh all          the whole x_engine_stats blob
#   scripts/stats.sh health       /health only -- makes NO generation request
#   scripts/stats.sh watch [SEC]  poll the summary every SEC seconds (default 30)
#   scripts/stats.sh keys         which stat keys this build exposes
#   scripts/stats.sh demand       analyse the persisted routing-demand DB (needs numpy, so it
#                                 runs inside the serving container; makes NO request)
#
# Env: BASE (default http://127.0.0.1:8000), MODEL (default from .env, else "deepseek"),
#      FORCE=1 to probe even while the server is busy with a real request.
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && { set -a; . "$ROOT/.env"; set +a; }
BASE="${BASE:-http://127.0.0.1:${PORT:-8000}}"
MODEL="${MODEL:-${SERVED_MODEL_NAME:-deepseek}}"

command -v jq >/dev/null || { echo "stats.sh needs jq (apt install jq)" >&2; exit 1; }

health() { curl -s --max-time 5 "$BASE/health"; }

require_up() {
  local h; h="$(health || true)"
  [[ -n "$h" ]] || { echo "no server at $BASE" >&2; exit 1; }
  printf '%s' "$h"
}

# A 1-token generation, purely to read the cumulative counters it returns.
probe() {
  local h busy
  h="$(require_up)"
  busy="$(printf '%s' "$h" | jq -r '.busy')"
  if [[ "$busy" == "true" && "${FORCE:-0}" != "1" ]]; then
    # Queueing behind a real request would both delay it and block here for its whole
    # duration, which is a surprising thing for a stats command to do.
    echo "server is busy with a request; try again when idle (or FORCE=1)" >&2
    exit 2
  fi
  curl -s --max-time 600 "$BASE/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"ok\",\"max_tokens\":1,\"temperature\":0}"
}

pct() { jq -r "$1 * 1000 | round / 10"; }

cmd_health() {
  require_up | jq '{status, busy, uptime_s, max_context,
                    ep: .engine_config.ep_world_size,
                    prune_keep: .engine_config.prune_keep,
                    arena_gb: .engine_config.arena_gb,
                    resident_expert_pct: .engine_config.resident_expert_pct}'
}

cmd_miss() {
  probe | jq -r '.x_engine_stats.prune_miss
    | if . == null then
        "pruning miss tracking is off (start with DSV41_PRUNE_MISS=1)"
      else
        "miss \(.miss_rate*1000|round/10)%   \(.missed_slots)/\(.total_slots) routing slots   \(.distinct_experts_wanted) distinct experts wanted"
      end'
}

cmd_layers() {
  probe | jq -r '.x_engine_stats.prune_miss
    | if . == null then "pruning miss tracking is off (DSV41_PRUNE_MISS=1)"
      else "worst layers:", (.worst_layers[] | "  layer \(.layer)\t\(.miss_rate*1000|round/10)%")
      end'
}

cmd_all()  { probe | jq '.x_engine_stats'; }

# The demand DB is numpy, and numpy lives in the serving image rather than on the host, so
# this borrows the running container to read a file that is bind-mounted from the host anyway.
cmd_demand() {
  local c="${CONTAINER:-deepseek-v41-ep2-rank0}"
  docker ps --format '{{.Names}}' | grep -qx "$c" \
    || { echo "container $c is not running (set CONTAINER=...)" >&2; exit 1; }
  docker exec -w /app "$c" python3 /app/tools/prune_miss_report.py \
    /app/results/prune_demand.npz --keep "${KEEP:-${PRUNE_KEEP:-0.6}}" --top "${TOP:-8}"
}
cmd_keys() { probe | jq -r '.x_engine_stats | keys[]'; }

cmd_summary() {
  local h s
  h="$(require_up)"
  s="$(probe)"
  printf '%s\n' "$s" | jq -r --argjson h "$h" '
    .x_engine_stats as $s |
    "server      : up \($h.uptime_s)s, ctx \($h.max_context), ep=\($h.engine_config.ep_world_size)x",
    "experts     : keep \($h.engine_config.prune_keep), \($h.engine_config.arena_gb) GB arena, \($h.engine_config.resident_expert_pct)% resident",
    ( if $s.prune_miss == null then "pruning     : miss tracking off (DSV41_PRUNE_MISS=1)"
      else "pruning     : miss \($s.prune_miss.miss_rate*1000|round/10)% cumulative (\($s.prune_miss.missed_slots)/\($s.prune_miss.total_slots) slots)" end ),
    "prefill     : \($s.prefill_tok_s // "n/a") tok/s",
    "decode      : \($s.decode_tok_s // "n/a") tok/s, accept \($s.accept_len_mean // "n/a")",
    "experts nvme: \($s.nvme_gb // 0) GB, hit rate \($s.expert_hit_rate // "n/a")"'
}

cmd_watch() {
  local n="${1:-30}"
  echo "polling $BASE every ${n}s -- ctrl-C to stop"
  while true; do
    printf '%s  ' "$(date +%H:%M:%S)"
    # a busy server is normal here, so skip that round rather than exiting the loop
    cmd_miss 2>/dev/null || echo "(busy)"
    sleep "$n"
  done
}

case "${1:-summary}" in
  summary|"") cmd_summary ;;
  miss)       cmd_miss ;;
  layers)     cmd_layers ;;
  all)        cmd_all ;;
  keys)       cmd_keys ;;
  demand)     cmd_demand ;;
  health)     cmd_health ;;
  watch)      cmd_watch "${2:-30}" ;;
  -h|--help|help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown command: $1 (try --help)" >&2; exit 1 ;;
esac
