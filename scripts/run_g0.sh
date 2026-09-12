#!/usr/bin/env bash
# Launch Gate G0 (docs/dual-spark-plan.md) on both Sparks and report the verdict.
#
# Ordering matters and the first version had it backwards. Rank 0 is the TCPStore SERVER: it
# binds MASTER_ADDR:MASTER_PORT, and every other rank is a client that can only retry until it
# appears. Starting rank 1 first meant the peer spent its whole connect budget against a port
# nothing was listening on yet, and reported the result as "TCPStore timeout" -- which reads
# like a broken fabric and is really a broken launch order. So: rank 0 first, WAIT for the port
# to actually accept, then rank 1.
#
# The other bring-up bug is in here as a lesson, and it has now bitten twice: a cleanup
# `pkill -f <gate name>` matches ANY process whose command line contains that name -- including
# the `bash -c` that ssh is running it under, and including the caller's own shell when the gate
# is selected as `G0_SCRIPT=scripts/gate_g1_....py ./scripts/run_g0.sh`. The pattern is therefore
# anchored on `python`, which the launcher's own command line never contains while every real
# gate process does. (`gate_g0[_]nccl`'s bracket only protected the one literal spelling; widening
# it to cover G1 silently removed that protection and killed the launcher.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PEER="${PEER:-ryan@10.0.0.2}"
MASTER_ADDR="${MASTER_ADDR:-10.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29613}"
IFACE="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
BACKEND="${G0_BACKEND:-nccl}"
# Which gate to launch. The ordering, the per-node GID resolution and the container plumbing
# below are the same for any 2-rank probe, so G1 (graph-captured collectives) reuses all of it
# rather than growing a second copy that drifts.
G0_SCRIPT="${G0_SCRIPT:-scripts/gate_g0_nccl.py}"
PY="${PY:-$ROOT/.venv/bin/python}"
LAUNCH_TIMEOUT_S="${LAUNCH_TIMEOUT_S:-180}"
# G0_DOCKER=1 runs the identical gate inside the serving image instead of the host venv --
# the way to prove the container path keeps RoCE, rather than assuming it. The flags below
# are the same ones scripts/dual-up.sh gives the engine; drop any of the verbs three
# (--device /dev/infiniband, --cap-add IPC_LOCK, --ulimit memlock=-1) and this gate is how
# you find out, because NCCL does not announce the fall back to TCP -- the median just goes
# from ~60 us to ~1.9 ms.
G0_DOCKER="${G0_DOCKER:-0}"
IMAGE="${IMAGE:-deepseek-v41-flash-spark:local}"
DOCKER_FLAGS=(--network host --gpus all --device /dev/infiniband
              --cap-add IPC_LOCK --ipc host --ulimit memlock=-1)

ssh_peer() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" "$@"; }

# --- RoCE pins, resolved PER NODE ---------------------------------------------------------
# These two boxes do not agree on GID indices: the RoCEv2/IPv4 entry for the CX7 port is index
# 5 on 10.0.0.1 and index 6 on 10.0.0.2 (index 5 there is an empty slot). Exporting one
# NCCL_IB_GID_INDEX to both ranks -- which is what the working single-box vLLM/SGLang configs
# looked like they licensed -- is correct on rank 0 and points rank 1 at nothing:
#   ibv_modify_qp failed with 61 ... local GID index 5, local GID ::, remote GID ::ffff:10.0.0.1
# So each rank derives its own from sysfs. See scripts/roce_gid.sh.
read -r HCA0 GID0 < <("$ROOT/scripts/roce_gid.sh" "$MASTER_ADDR") \
    || { echo "cannot resolve the local RoCE GID for $MASTER_ADDR" >&2; exit 1; }
PEER_IP="${PEER#*@}"
read -r HCA1 GID1 < <(ssh_peer "cd '$ROOT' && ./scripts/roce_gid.sh '$PEER_IP'") \
    || { echo "cannot resolve the peer's RoCE GID for $PEER_IP (is the checkout synced? scripts/sync-peer.sh)" >&2; exit 1; }

common_env=(
  WORLD_SIZE=2 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT"
  G0_BACKEND="$BACKEND"
  NCCL_DEBUG="${NCCL_DEBUG:-INFO}" NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
  NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1
)
rank0_env=(NCCL_IB_HCA="$HCA0" NCCL_IB_GID_INDEX="$GID0")
rank1_env=(NCCL_IB_HCA="$HCA1" NCCL_IB_GID_INDEX="$GID1")

cleanup() {
  pkill -9 -f 'python.*gate_g.*nccl' 2>/dev/null || true
  ssh_peer "pkill -9 -f 'python.*gate_g.*nccl' 2>/dev/null || true" >/dev/null 2>&1 || true
  if [[ "$G0_DOCKER" == "1" ]]; then
    docker rm -f g0-rank0 >/dev/null 2>&1 || true
    ssh_peer "docker rm -f g0-rank1 >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
cleanup
sleep 1
rm -f /tmp/g0_r0.log
ssh_peer 'rm -f /tmp/g0_r1.log' || true

echo "G0: backend=$BACKEND master=$MASTER_ADDR:$MASTER_PORT iface=$IFACE  rank0=$HCA0/gid$GID0  rank1=$HCA1/gid$GID1  mode=$([[ $G0_DOCKER == 1 ]] && echo container:$IMAGE || echo native)"

# --- rank 0 (store server) first, in the background --------------------------------------
cd "$ROOT"
if [[ "$G0_DOCKER" == "1" ]]; then
    docker run --rm --name g0-rank0 "${DOCKER_FLAGS[@]}" \
        $(printf -- '-e %q ' RANK=0 "${common_env[@]}" "${rank0_env[@]}") \
        "$IMAGE" python -u "/app/$G0_SCRIPT" > /tmp/g0_r0.log 2>&1 &
else
    env RANK=0 "${common_env[@]}" "${rank0_env[@]}" "$PY" -u "$G0_SCRIPT" > /tmp/g0_r0.log 2>&1 &
fi
R0=$!
echo "rank 0 pid $R0 -> /tmp/g0_r0.log"

# --- wait for the rendezvous port to accept before rank 1 gets to try --------------------
echo -n "waiting for $MASTER_ADDR:$MASTER_PORT to listen: "
for i in $(seq 1 "$LAUNCH_TIMEOUT_S"); do
    if ! kill -0 "$R0" 2>/dev/null; then
        echo "rank 0 died before binding"; echo "--- /tmp/g0_r0.log ---"; cat /tmp/g0_r0.log; exit 1
    fi
    if (exec 3<>"/dev/tcp/$MASTER_ADDR/$MASTER_PORT") 2>/dev/null; then echo "up after ${i}s"; break; fi
    sleep 1
    [[ $i -eq $LAUNCH_TIMEOUT_S ]] && { echo "TIMEOUT"; cat /tmp/g0_r0.log; exit 1; }
done

# --- rank 1 on the peer -------------------------------------------------------------------
if [[ "$G0_DOCKER" == "1" ]]; then
    # not --rm: the container must survive its own exit long enough for `docker logs`
    # below to read the measurement out of it.
    ssh_peer "docker run -d --name g0-rank1 $(printf '%q ' "${DOCKER_FLAGS[@]}") \
        $(printf -- '-e %q ' RANK=1 "${common_env[@]}" "${rank1_env[@]}") \
        '$IMAGE' python -u '/app/$G0_SCRIPT'" >/dev/null \
      || { echo 'failed to start rank 1 container' >&2; exit 1; }
else
    ssh_peer "cd '$ROOT' && nohup env RANK=1 $(printf '%q ' "${common_env[@]}" "${rank1_env[@]}") \
      '$PY' -u '$G0_SCRIPT' > /tmp/g0_r1.log 2>&1 & echo \$!" | tail -1 | tee /tmp/g0_peer_pid.txt
fi

# --- collect ------------------------------------------------------------------------------
rc=0; wait "$R0" || rc=$?
echo "=================== rank 0 (/tmp/g0_r0.log) ==================="
cat /tmp/g0_r0.log
echo "=================== rank 1 ($PEER:/tmp/g0_r1.log) ============="
if [[ "$G0_DOCKER" == "1" ]]; then
    ssh_peer 'docker logs g0-rank1 2>&1' || echo "(peer container log unavailable)"
else
    ssh_peer 'cat /tmp/g0_r1.log' 2>/dev/null || echo "(peer log unavailable)"
fi
echo "==============================================================="
echo "rank 0 exit: $rc"
exit $rc
