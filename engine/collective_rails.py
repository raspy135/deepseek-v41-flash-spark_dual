"""Phase-specific NCCL groups for a two-Spark, one-GPU-per-node TP/EP pair.

NCCL caches IB_HCA and NETDEVS_POLICY process-wide. Changing those around new_group
does NOT isolate a rail. Instead, discover the dual-rail graph once, retain its
first rail's channels for the default communicator, and warm both at boot. No
environment changes, file I/O, or communicator creation occur during inference.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import torch
import torch.distributed as dist

_prefill = ContextVar("collective_prefill", default=False)
_prefill_group = None
VERSION = 1


@contextmanager
def phase(prefill):
    token = _prefill.set(bool(prefill))
    try:
        yield
    finally:
        _prefill.reset(token)


def group():
    return _prefill_group if _prefill.get() else None


def forward_phase(fn):
    @wraps(fn)
    def wrapped(self, ids, S, prefill, *args, **kwargs):
        with phase(prefill):
            return fn(self, ids, S, prefill, *args, **kwargs)
    return wrapped


def prefill_phase(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with phase(True):
            return fn(*args, **kwargs)
    return wrapped


def single_rail_graph(source, destination):
    """Retain only channels entirely on one NET device, preserving GPU identities.

    Never invent bus IDs or network numbering. Ring AND tree must have channels
    on the same selected rail; otherwise refuse to silently fall back to dual rail.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    nets = {n.get("dev") for n in root.iter("net")}
    if root.tag != "graphs" or len(nets) != 2:
        raise RuntimeError(f"prefill dual rail requires exactly two unmerged NET devices: {nets}")
    rail = min(nets, key=lambda n: int(n, 16))
    seen = set()
    for graph in root:
        original = list(graph)
        for channel in original:
            channel_nets = channel.findall("net")
            if not channel_nets or any(n.get("dev") != rail for n in channel_nets):
                graph.remove(channel)
        if original and not len(graph):
            raise RuntimeError(f"no single-rail channels for NCCL graph {graph.get('id')}")
        graph.set("nchannels", str(len(graph)))
        if len(graph):
            seen.add(graph.get("id"))
    if not {"0", "1"} <= seen:
        raise RuntimeError("NCCL graph must contain single-rail ring and tree routes")
    ET.indent(tree)
    tree.write(destination, encoding="unicode")
    return rail


@contextmanager
def _environment(**values):
    old = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def initialize(enabled, device, timeout):
    """Called on ALL ranks immediately after the mixed Gloo/NCCL default group.

    The early Gloo agreement precedes any optional collective. V41Engine also
    includes this policy in its full boot config guard. CUDA default group must
    still be lazy here (no device_id in init_process_group and no CUDA collective).
    """
    global _prefill_group
    cfg = (bool(enabled), VERSION)
    peers = [None] * dist.get_world_size()
    dist.all_gather_object(peers, cfg)
    if any(p != cfg for p in peers):
        raise RuntimeError(f"prefill dual-rail config differs across ranks: {peers}")
    if not enabled:
        return
    errors = []
    if device.type != "cuda" or dist.get_world_size() != 2:
        errors.append("requires two CUDA ranks")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        errors.append("requires one visible GPU per node")
    if os.environ.get("NCCL_IB_MERGE_NICS") != "0":
        errors.append("requires NCCL_IB_MERGE_NICS=0 to keep rails independently selectable")
    if os.environ.get("NCCL_CROSS_NIC") != "0":
        errors.append("requires NCCL_CROSS_NIC=0")
    if len(os.environ.get("NCCL_IB_HCA", "").split(",")) != 2:
        errors.append("requires two explicitly selected HCAs")
    if os.environ.get("NCCL_IB_GID_INDEX") or not os.environ.get("NCCL_IB_ADDR_RANGE"):
        errors.append("requires dynamic GIDs constrained by NCCL_IB_ADDR_RANGE")
    for name in ("NCCL_GRAPH_FILE", "NCCL_GRAPH_DUMP_FILE", "NCCL_GRAPH_DUMP_FILE_RANK",
                 "NCCL_NETDEVS_POLICY"):
        if name in os.environ:
            errors.append(f"incompatible external override: {name}")
    # This uses NCCL's graph XML format; only versions actually exercised are enabled.
    if device.type == "cuda" and torch.cuda.nccl.version()[:2] != (2, 29):
        errors.append("graph routing validated with NCCL 2.29 only")
    dist.all_gather_object(peers, errors)
    if any(peers):
        raise RuntimeError(f"invalid prefill dual-rail configuration: {peers}")
    with tempfile.TemporaryDirectory(prefix="dsv41-rails-") as directory:
        both = str(Path(directory) / "prefill.xml")
        single = str(Path(directory) / "decode.xml")
        with _environment(NCCL_GRAPH_DUMP_FILE=both,
                          NCCL_GRAPH_DUMP_FILE_RANK=str(dist.get_rank())):
            _prefill_group = dist.new_group(backend="nccl", timeout=timeout)
            probe = torch.ones(1, device=device)
            dist.all_reduce(probe, group=_prefill_group)
            torch.cuda.synchronize(device)
        error = None
        try:
            rail = single_rail_graph(both, single)
        except Exception as exc:
            error = str(exc)
        dist.all_gather_object(peers, error)
        if any(peers):
            raise RuntimeError(f"cannot construct decode rail graph: {peers}")
        with _environment(NCCL_GRAPH_FILE=single):
            # Force lazy default NCCL initialization while its single-rail graph is set.
            probe.fill_(1)
            dist.all_reduce(probe)
            torch.cuda.synchronize(device)
        print(f"[rails] rank {dist.get_rank()}: decode NET/{rail}; prefill both rails", flush=True)


def destroy():
    global _prefill_group
    if _prefill_group is not None:
        dist.destroy_process_group(_prefill_group)
        _prefill_group = None
