"""Fast decode-sized all-gathers: the one-shot RoCE transport (tools/roce/) with an NCCL fallback.

``DSV41_COMM_BACKEND=roce`` (default ``nccl``) replaces ``torch.distributed.all_gather_into_tensor``
for the decode exchanges only: the default group, contiguous tensors, and one rank's shard at most
``DSV41_ROCE_MAX_KB`` (default 256). Prefill gathers (larger) and every control exchange stay on NCCL.
The transport moves the same bytes in the same order, so the bits are unchanged.

A run-time timeout poisons the RoCE runtime (later gathers raise); the engine's ``check()`` after a
step turns that into a diagnosis instead of a wrong answer.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
_ROCE_DIR = os.path.join(_TOOLS, "roce")
# tools/roce on the path BEFORE tools/: `import roce` must find tools/roce/roce.py, not resolve
# tools/roce/ itself as a namespace package.
for _p in (_TOOLS, _ROCE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    import roce as _roce  # tools/roce/roce.py
except Exception:  # noqa: BLE001
    _roce = None

_COMM = None


class _Base:
    """The default process group in the shape tools/roce expects (NCCL fallback for the setup probe)."""

    def __init__(self, device):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = device

    def all_gather(self, send, recv):
        dist.all_gather_into_tensor(recv, send)

    def barrier(self):
        dist.barrier()


def enabled() -> bool:
    return os.environ.get("DSV41_COMM_BACKEND", "nccl").strip().lower() == "roce"


def init(device):
    """Collective on both ranks; must run after the default process group exists and before any
    captured graph. Leaves ``_COMM`` None (plain NCCL) when the backend is nccl or setup fails and
    the fallback allows it. ``tools/roce.roce.select`` does the settings agreement check."""
    global _COMM
    if not enabled():
        return
    if _roce is None:
        raise RuntimeError("DSV41_COMM_BACKEND=roce but tools/roce could not be imported")
    comm = _roce.select(_Base(device))
    _COMM = comm if isinstance(comm, _roce.RoceComm) else None


def check():
    """Raise the recorded RoCE diagnosis if a gather timed out (call after a step)."""
    if _COMM is not None:
        _COMM.check()


def all_gather_fast(recv, send, group=None):
    """``all_gather_into_tensor(recv, send, group)`` with the RoCE path when it applies."""
    if _COMM is not None and group is None and send.is_contiguous() and recv.is_contiguous():
        n = send.numel() * send.element_size()
        if 0 < n <= _COMM.max_bytes:
            _COMM.rt.gather(send, recv)
            return
    dist.all_gather_into_tensor(recv, send, group=group)
