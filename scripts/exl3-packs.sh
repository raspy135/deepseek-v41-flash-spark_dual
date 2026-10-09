#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# exl3-packs.sh -- build the per-node EXL3 expert packs for EXPERT_FORMAT=exl3, once.
#
# The engine reads routed experts from one pack per TP rank: each expert's rank slice stored
# contiguously and 4096-aligned, so a load is one O_DIRECT read (tools/pack_exl3_experts.py).
# This builds both ranks' packs here, from the EXL3 checkpoint, inside the serving image (no host
# Python), then copies rank 1's to the peer. Only this node needs the EXL3 checkpoint; after the
# packs exist it can be deleted.
#
#   checkpoint   <models>/DeepSeek-V4.1-Flash-EXL3-2.9bpw   ~198 GB, this node only
#   packs        <models>/exl3-packs/exl3-experts-r{0,1}of2.bin   ~98.3 GB each
#   (<models> = MODELS_DIR, else MODEL_DIR's parent: where the engine looks for them)
#
# Usage: scripts/exl3-packs.sh [--check] [--force] [checkpoint-dir]
#   --check   verify the image, checkpoint and disk space on both nodes, build nothing
#   --force   rebuild packs that already exist
# Each rank takes ~8 minutes here; the copy is ~98 GB over the PEER link.
# ---------------------------------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
info() { echo "--- $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

CHECK=false
FORCE=false
SRC_ARG=""
for a in "$@"; do
    case "$a" in
        --check) CHECK=true ;;
        --force) FORCE=true ;;
        -*) err "unknown option $a (expected --check or --force)" ;;
        *) SRC_ARG="$a" ;;
    esac
done

# shellcheck disable=SC1091
[[ -f .env ]] && { set -a; . ./.env; set +a; }
IMAGE="${IMAGE:-deepseek-v41-flash-spark:local}"
PEER="${PEER:-}"
HOST_MODELS="${MODELS_DIR:-$(dirname "${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash}")}"
SRC="${SRC_ARG:-$HOST_MODELS/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
OUT="$HOST_MODELS/exl3-packs"
PACK_GB=99   # 98.24 GB a rank, plus slack

[[ -n "$PEER" ]] || err "PEER is required (set it in .env)"
docker image inspect "$IMAGE" >/dev/null 2>&1 || err "image $IMAGE missing: run scripts/dual-build.sh first"
[[ -f "$SRC/config.json" ]] || err "no EXL3 checkpoint at $SRC (download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw there, or pass its path)"
case "$SRC" in "$HOST_MODELS"/*) ;; *) err "the checkpoint must sit under $HOST_MODELS (it is mounted into the image as /models)";; esac
grep -q '"quant_method": *"exl3"' "$SRC/config.json" || err "$SRC/config.json is not an EXL3 checkpoint"

free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }
need=0
for r in 0 1; do
    [[ -f "$OUT/exl3-experts-r${r}of2.bin" && "$FORCE" == false ]] || need=$((need + PACK_GB))
done
mkdir -p "$OUT"
have=$(free_gb "$OUT")
(( have >= need )) || err "this node has ${have} GB free under $OUT, the packs need ${need} GB"
ssh -o BatchMode=yes "$PEER" "mkdir -p '$OUT'" || err "cannot reach $PEER"
peer_has=$(ssh -o BatchMode=yes "$PEER" "test -f '$OUT/exl3-experts-r1of2.bin' && stat -c %s '$OUT/exl3-experts-r1of2.bin' || echo 0")
peer_free=$(ssh -o BatchMode=yes "$PEER" "df -BG --output=avail '$OUT' | tail -1 | tr -dc 0-9")
(( peer_has > 0 || peer_free >= PACK_GB )) || err "$PEER has ${peer_free} GB free under $OUT, rank 1's pack needs ${PACK_GB} GB"
info "image $IMAGE, checkpoint $SRC, packs -> $OUT (this node ${have} GB free, $PEER ${peer_free} GB free)"
[[ "$CHECK" == true ]] && { info "--check: ready; nothing built"; exit 0; }

for r in 0 1; do
    pack="$OUT/exl3-experts-r${r}of2.bin"
    if [[ -f "$pack" && "$FORCE" == false ]]; then
        info "rank $r: $pack exists ($(stat -c %s "$pack") bytes), kept (--force rebuilds)"
        continue
    fi
    info "rank $r: packing (~8 min)"
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HOST_MODELS:/models" \
        --entrypoint python "$IMAGE" /app/tools/pack_exl3_experts.py \
        --source "/models/${SRC#"$HOST_MODELS"/}" --out "/models/exl3-packs/exl3-experts-r${r}of2.bin" \
        --rank "$r" --world 2
done

local_size=$(stat -c %s "$OUT/exl3-experts-r1of2.bin")
if [[ "$peer_has" != "$local_size" || "$FORCE" == true ]]; then
    info "copying rank 1's pack to $PEER:$OUT"
    rsync -W --info=progress2 "$OUT/exl3-experts-r1of2.bin" "$PEER:$OUT/"
fi
peer_size=$(ssh -o BatchMode=yes "$PEER" "stat -c %s '$OUT/exl3-experts-r1of2.bin'")
[[ "$peer_size" == "$local_size" ]] || err "rank 1's pack on $PEER is $peer_size bytes, here $local_size"
info "done: rank 0 here, rank 1 on $PEER. Rank 1's local copy ($OUT/exl3-experts-r1of2.bin) is no longer
    needed on this node; delete it to reclaim ~98 GB. The boot guard checks both packs share one source."
