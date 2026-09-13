#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# replicate-checkpoint.sh -- copy the DeepSeek-V4.1-Flash checkpoint to the
# second DGX Spark over the ConnectX-7 link, then verify it.
#
#   ./scripts/replicate-checkpoint.sh                 # watch the local download,
#                                                     # then replicate + verify
#   ./scripts/replicate-checkpoint.sh --now           # skip the watcher (download
#                                                     # already complete)
#   ./scripts/replicate-checkpoint.sh --peer user@10.0.0.2
#   PARTS=12 STREAMS=10 ./scripts/replicate-checkpoint.sh   # tuning knobs
#
# Why this exists: dual Spark needs the FULL checkpoint on each box's LOCAL
# NVMe -- the engine's expert streaming is O_DIRECT preadv, which is exactly
# the thing that must not cross a network mount (env.example says as much).
#
# Why it is parallel: one ssh stream is capped by its single-threaded AES at a
# few hundred MB/s, nowhere near the 25 GB/s the link can do; PARTS streams of
# ssh + rsync converge on the receiver's NVMe write rate (~3 GB/s), i.e. the
# whole ~510 GB move in minutes instead of an hour. rsync -W (whole file, no
# delta) because FP4 weights are incompressible noise to a delta algorithm;
# -I so a partial destination file is re-sent, not resumed (a resumed shard
# that was written by a crashed earlier run would otherwise silently stitch).
#
# Verification: size match per file against the source, plus one sha256 per
# stream (the sha is computed on the SENDER while sending and re-computed on
# the receiver after -- the only honest check, and the shards are 10 GB each
# so it rides the same disk queues for free).
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

err()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "--- $(date +%T) $*"; }

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }

NOW=0
PEER="${PEER:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --now) NOW=1 ;;
        --peer) shift; PEER="${1:?--peer needs a value}" ;;
        -h|--help) sed -n '3,12p' "$SCRIPT_DIR/scripts/replicate-checkpoint.sh" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) if [[ "$1" == *@* || "$1" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then PEER="$1"; else err "unknown argument: $1"; fi ;;
    esac
    shift
done
[[ -n "${PEER:-}" ]] || { echo "PEER is required (e.g. --peer user@10.0.0.2)" >&2; exit 1; }
[[ "$PEER" != *@* ]] && PEER="${USER}@$PEER"   # bare ip gets THIS box's login, not a hardcoded one
MODEL_DIR="${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash}"
[[ "$MODEL_DIR" != /* ]] && err "MODEL_DIR must be absolute (it is reused verbatim on $PEER): $MODEL_DIR"
PARTS="${PARTS:-12}"          # parallel streams
STREAM_TIMEOUT_S="${STREAM_TIMEOUT_S:-3600}"
[[ -d "$MODEL_DIR" ]] || err "source checkpoint dir not found: $MODEL_DIR"

# --- wait for the local download to finish (unless --now) ------------------
# The download wrapper (scripts/download-model.sh under nohup) exits when
# snapshot_download verifies every shard named in model.safetensors.index.json.
# Completion = the index check passes AND no download-model.sh process remains.
index_complete() {
    python3 - "$MODEL_DIR" <<'PY' 2>/dev/null
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(d, s))]
sys.exit(0 if not missing else 1)
PY
}
if [[ "${NOW:-0}" != "1" ]]; then
    info "waiting for the local download to complete (checking every 60s; --now to skip)"
    while :; do
        if index_complete && ! pgrep -f "download-model\.sh" >/dev/null 2>&1; then
            break
        fi
        done_n=$(ls -1 "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
        echo -n " [${done_n}/48 shards, still downloading] "
        sleep 60
    done
    echo
fi
index_complete || err "index says shards are missing even after the wait -- run scripts/download-model.sh again"
info "local checkpoint complete: $(du -sh "$MODEL_DIR" | cut -f1); replicating to $PEER with $PARTS streams"

# --- destination on the peer -----------------------------------------------
ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" "mkdir -p '$MODEL_DIR'" \
    || err "cannot ssh to $PEER (passwordless ssh + same MODEL_DIR path are prerequisites)"

# --- the manifest: everything except the volatile .incomplete staging -------
# safetensors + index + tokenizer + encoding/ + any *.json|*.py the server loads.
( cd "$MODEL_DIR" && find . -type f \
      ! -path './.cache/*' ! -name '*.incomplete' ! -name '.DS_Store' \
      -printf '%s\t%p\n' | sort -k2 ) > /tmp/dsv41-repl-manifest.tsv
N_FILES=$(wc -l < /tmp/dsv41-repl-manifest.tsv)
info "manifest: $N_FILES files"

# --- stream jobs: shards go to $PARTS streams by SIZE (descending, round-robin:
# the two 101.5 GB engram shards land on different streams and the tail of
# small files spreads the rest; everything non-shard rides stream 0.) -------
awk -F'\t' -v P="$PARTS" '
    $2 ~ /model-[0-9]+-of-00048\.safetensors$/ { jobs[$2] = $1 }
    END { for (j in jobs) print jobs[j] "\t" j }
' /tmp/dsv41-repl-manifest.tsv | sort -rn | \
    awk -v P="$PARTS" '{ print "s" NR % P "\t" $2 }' > /tmp/dsv41-repl-jobs.tsv
awk -F'\t' '$2 !~ /model-[0-9]+-of-00048\.safetensors$/ { print "s0\t" $2 }' \
    /tmp/dsv41-repl-manifest.tsv >> /tmp/dsv41-repl-jobs.tsv
for ((i=0; i<PARTS; i++)); do
    awk -F'\t' -v s="s$i" '$1 == s { print $2 }' /tmp/dsv41-repl-jobs.tsv > "/tmp/dsv41-repl-$i.list"
done

# --- one stream: rsync its file list, then spot-verify sha256 -------------
stream() {  # $i = stream id
    local i="$1" list="/tmp/dsv41-repl-$1.list"
    [[ -s "$list" ]] || return 0
    local n t0 rc
    n=$(wc -l < "$list")
    t0=$(date +%s)
    # --files-from: exact list, one name per line (NO -0: that would demand
    # NULL-separated input), -W whole file, -I restart partial files.
    rsync -a -W -I --files-from="$list" "$MODEL_DIR"/ "$PEER:$MODEL_DIR/" \
        > "/tmp/dsv41-repl-$i.log" 2>&1
    rc=$?
    local rate=$(( ( $(date +%s) - t0 ) > 0 ? n * 60 / ( $(date +%s) - t0 ) : n ))
    echo "stream $i: $n files in $(($(date +%s) - t0))s (~${rate} files/min, rc=$rc)" >> /tmp/dsv41-repl-status.log
    return $rc
}
pids=()
rm -f /tmp/dsv41-repl-status.log
for ((i=0; i<PARTS; i++)); do stream "$i" & pids+=($!); done

info "$PARTS streams running; watch: tail -f /tmp/dsv41-repl-*.log"
FAIL=0
for p in "${pids[@]}"; do wait "$p" || FAIL=1; done
[[ "$FAIL" == "1" ]] && err "at least one stream failed -- see /tmp/dsv41-repl-*.log (safe to re-run: rsync is idempotent)"

# --- verify: size parity over the whole manifest, then a sha256 sweep -----
# Sender sizes are already in the manifest; ask the peer for its sizes over one
# ssh and join offline. A mismatch means re-sync (rsync is idempotent), so a
# crashed stream can never be mistaken for a good copy.
info "streams done; verifying sizes"
remote_sizes=$(awk -F'\t' '{ print $2 }' /tmp/dsv41-repl-manifest.tsv | \
    ssh -o BatchMode=yes "$PEER" "cd '$MODEL_DIR' && xargs -d'\n' -r stat -c '%s\t%n' -- 2>/dev/null")
[[ -n "$remote_sizes" ]] || err "remote size sweep failed to run (ssh? files missing?)"
if ! join -t$'\t' \
        <(awk -F'\t' '{print $2"\t"$1}' /tmp/dsv41-repl-manifest.tsv | sort -t$'\t' -k1,1) \
        <(printf '%s\n' "$remote_sizes"   | awk -F'\t' '{print $2"\t"$1}' | sort -t$'\t' -k1,1) \
        -o 1.1,1.2,2.2 2>/dev/null | awk -F'\t' '$2 != $3 { bad++; print "MISMATCH " $0 } END { exit bad ? 1 : 0 }'; then
    err "size mismatch after sync -- re-run this script (rsync will repair)"
fi
info "sizes match on $N_FILES files; sha256 sweep (checkpoint file names are quote/space free -- guaranteed by the HF repo layout)"
names=$(cut -f2 /tmp/dsv41-repl-manifest.tsv)
sha_src=$(cd "$MODEL_DIR" && printf '%s\n' "$names" | xargs -d'\n' -r -P8 sha256sum)
ssh -o BatchMode=yes "$PEER" "cd '$MODEL_DIR' && printf '%s\n' '$names' | xargs -d'\n' -r -P8 sha256sum" > /tmp/dsv41-repl-remote.sha
printf '%s\n' "$sha_src" | sort > /tmp/dsv41-repl-local.sha
sort -o /tmp/dsv41-repl-remote.sha /tmp/dsv41-repl-remote.sha
if diff -q /tmp/dsv41-repl-local.sha /tmp/dsv41-repl-remote.sha >/dev/null; then
    info "VERIFIED: all $N_FILES files sha256-identical on $PEER:$MODEL_DIR"
else
    err "sha256 mismatch (see diff /tmp/dsv41-repl-{local,remote}.sha) -- do NOT serve from the peer copy"
fi
info "replication complete: $(du -sh "$MODEL_DIR" | cut -f1) in $PARTS streams"
