#!/usr/bin/env bash
# Print the RoCE device and GID index this box must use to talk RDMA out of a given local IPv4.
#
# Why this exists: the two Sparks do NOT agree on GID indices. On 10.0.0.1 the RoCEv2/IPv4 entry
# for the CX7 port is index 5; on 10.0.0.2 index 5 is empty and the same entry is at index 6
# (the box carries an extra netdev, and the GID table is packed in address order, not by role).
# A single NCCL_IB_GID_INDEX=5 for both ranks is therefore correct on one and garbage on the
# other, and NCCL reports that as:
#
#   ibv_modify_qp failed with 61, curr state INIT, next state RTR, local GID index 5,
#   local GID ::, remote GID ::ffff:10.0.0.1
#
# -- "local GID ::" being the whole story: the index points at an empty slot. That read as a
# broken fabric for a while; it is a lookup bug. The index is not a constant, so derive it.
#
# Usage: scripts/roce_gid.sh [local-ipv4]        (default: the address on the 10.0.0.0/24 link)
# Prints: "<ib-device> <gid-index>", or exits non-zero with a reason on stderr.
set -euo pipefail

ip4="${1:-}"
if [[ -z "$ip4" ]]; then
    ip4=$(ip -4 -br addr show | awk '$3 ~ /^10\.0\.0\./ {split($3,a,"/"); print a[1]; exit}')
    [[ -n "$ip4" ]] || { echo "no 10.0.0.0/24 address on this box; pass the IP explicitly" >&2; exit 1; }
fi

iface=$(ip -4 -br addr show | awk -v ip="$ip4" '$3 ~ "^"ip"/" {print $1; exit}')
[[ -n "$iface" ]] || { echo "no interface holds $ip4 (inside a container? dual-up.sh runs with --network host so the host's links are visible)" >&2; exit 1; }

# netdev -> RoCE device, from sysfs first: /sys/class/net/<iface>/device/infiniband/<dev>.
# sysfs is the portable answer -- it is there in a minimal container (scripts/dual-up.sh
# mounts nothing special for it), whereas ibdev2netdev comes from rdma-core/mlnx-tools and
# is absent from most images. ibdev2netdev stays as the fallback for odd topologies.
dev=$(ls "/sys/class/net/$iface/device/infiniband/" 2>/dev/null | head -1) || true
if [[ -z "$dev" ]]; then
    # `|| true`: ibdev2netdev exits non-zero whenever any port is Down (both boxes have
    # three), and under `set -o pipefail` that killed this script one line after it had
    # the answer.
    dev=$(ibdev2netdev 2>/dev/null | awk -v n="$iface" '$5 == n {print $1; exit}') || true
fi
[[ -n "$dev" ]] || { echo "no RoCE device maps to $iface (is the driver loaded?)" >&2; exit 1; }

# 10.0.0.1 -> the "0000:ffff:0a00:0001" tail of an IPv4-mapped GID.
hex=$(printf '%02x%02x:%02x%02x' $(echo "$ip4" | tr '.' ' '))
want="0000:0000:0000:0000:0000:ffff:$hex"

for i in $(seq 0 63); do
    g="/sys/class/infiniband/$dev/ports/1/gids/$i"
    [[ -r "$g" ]] || continue
    # RoCE v2 only: v1 is L2-scoped and will not route/bridge the same way, and the two boxes
    # must agree on the version as well as the address.
    [[ "$(cat "/sys/class/infiniband/$dev/ports/1/gid_attrs/types/$i" 2>/dev/null)" == "RoCE v2" ]] || continue
    if [[ "$(cat "$g")" == "$want" ]]; then
        echo "$dev $i"
        exit 0
    fi
done
echo "no RoCEv2 GID for $ip4 on $dev (checked indices 0-63)" >&2
exit 1
