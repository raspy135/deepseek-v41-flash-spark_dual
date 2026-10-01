"""One-shot RoCE all-gather for the decode-sized exchanges of this engine's two-Spark tensor parallelism
(``DSV41_COMM_BACKEND=roce``). Ported from TensorFold's patch 0230
(https://github.com/ashhart/TensorFold), which adapted b12x's "RoCEnante"
(https://github.com/local-inference-lab/b12x, ``b12x/comm/roce``, Copyright 2026 Luke Alonso and the b12x
contributors, Apache-2.0; see tools/roce/NOTICE).

Why
---
A decode step exchanges ~170 small all-gathers (48-97 KiB a rank here). NCCL's all-gather captured in a CUDA
graph costs 44-70 us however small the payload: its kernel hands the send to NCCL's network proxy thread, and the
step spends 7.6 ms of its ~98 ms there, all latency (measured two-node: tools/bench_roce_gather.py). This
transport is b12x's "RoCEnante": each rank owns a pinned host region that the ConnectX-7 RDMA-writes into and the
GB10 reads in place (no GPUDirect RDMA needed); one kernel stages the shard, rings a doorbell that a busy-spinning
C++ proxy thread turns into RDMA writes (payload, then a 4-byte sequence flag on the same RC queue pair, striped
over both CX7 PCIe functions), waits for the peer's flags with system-scope loads and copies the shards out in rank
order. The epoch lives on the device, so a replayed CUDA graph continues the sequence. RoCE graph gathers measure
12.7 (16 KiB) / 15.5 (48 KiB) / 19.9 (96 KiB) us against NCCL's 44-70, bits identical.

Exactness: an all-gather moves bytes. The output is every rank's shard in rank order, exactly NCCL's layout and
bits; the consumers (``glue.hc_post``, the MTP head's residual add, the drafter's row sum, the samplers) are
unchanged, so both ranks still add the same partials rank 0 first and hold identical bits.

What goes over RoCE
-------------------
Only the model's own exchanges, through ``comm.fast_gather``: ``forward.gather`` (every block's partials; decode,
verify, MTP, batched rounds and prefill chunks that fit), the samplers' candidate exchanges, the DFlash2 drafter's
rows and candidates, and 0084's pipelined prefill exchanges -- each only when one rank's shard is at most
``DSV41_ROCE_MAX_KB`` (default 256 KiB: a 16-row fp32 window; larger ones stay on NCCL). Everything else stays
on NCCL: the engine's control exchanges (settings checks, request headers and prompts sent to rank 1, calibration),
in particular the follower's idle wait for the next request, which must be able to wait forever.

Knobs (both ranks must agree on all but the CPU; checked at load)
-----------------------------------------------------------------
``DSV41_COMM_BACKEND``   ``nccl`` (default) | ``roce``
``DSV41_ROCE_MAX_KB``    largest shard (KiB a rank) sent over RoCE; also the slot size (pinned: 6 slots). 256
``DSV41_ROCE_TIMEOUT_S`` a wait for the peer's flag gives up after this many seconds (fail-stop, below). 120
``DSV41_ROCE_HCA``       ``name[:gid],...`` RDMA devices (and optionally RoCE v2 GID indices); default: every
                            ACTIVE Ethernet port, GID auto-detected
``DSV41_ROCE_HCAS``      stripe over at most this many HCAs (1 or 2). 2
``DSV41_ROCE_BLOCKS`` / ``_THREADS``   largest kernel grid (power of two) / threads a block. 8 / 512
``DSV41_ROCE_TC``        RoCE traffic class (DSCP/ECN byte); default ``NCCL_IB_TC`` or 0
``DSV41_ROCE_CPU``       pin the proxy thread to this CPU (default: not pinned; the thread busy-spins one core)
``DSV41_ROCE_FALLBACK``  ``nccl`` (default): a failed setup or load-time probe on either rank -> both ranks
                            serve on NCCL, with a log line; ``error``: refuse to start
``DSV41_ROCE_MARK``      file written when a RoCE collective fails at run time; while it exists (on either rank)
                            both ranks start on NCCL. Default ``/cache/roce-failed`` when ``/cache`` exists; empty:
                            none. Delete it to try RoCE again.

GIDs: RoCE v2 GID indices move on reboot and differ per port, so they are never hard-coded: for each ACTIVE port
the lowest index whose type is ``RoCE v2`` and whose GID is IPv4-mapped (``::ffff:a.b.c.d``) is used (the same rule
as the vLLM kit's ``detect-gids.sh``). The ranks' HCAs are paired by IPv4 subnet (e.g. rocep1s0f1 <-> a.b.100.x,
roceP2p1s0f1 <-> a.b.101.x), so names and order may differ between the nodes.

Failure model (b12x issues #313 / PR #438, sparkring #278)
-----------------------------------------------------------
Setup (HCA detection, queue pairs, a probe gather compared with NCCL's result) is collective: any failure on any
rank makes both ranks fall back to NCCL (or both refuse, with ``DSV41_ROCE_FALLBACK=error``). At run time a wait
that times out records the peer, HCA and sequence in the host-visible control record, sets a wrap-safe failure word
and poisons the runtime (later launches do nothing, so a wedged peer costs one timeout, not one per exchange). The
engine checks after every sampler exchange (each step has one, after the step's graphs) and raises a clear error
that says whether the peer's flag had reached this host's memory (lost in the GPU's view, sparkring #278's
signature) or never arrived, and what this rank's proxy had posted; the marker file makes the restarted pair use
NCCL. A timed-out step's data are not trusted: nothing derived from it is returned.
"""

from __future__ import annotations

import ctypes
import fcntl
import json
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BACKEND_ENV = "DSV41_COMM_BACKEND"
BACKENDS = ("nccl", "roce")
PACK = 16
SLOT_ALIGN = 4096
SYSFS = "/sys/class/infiniband"
PROBE_TIMEOUT_S = 15.0
DEFAULT_MARK = "/cache/roce-failed"
# control record words (roce_common.h)
CTRL_SEQ, CTRL_NBYTES, CTRL_ERR_SEQ, CTRL_ERR_PEER, CTRL_SLOT_NBYTES, CTRL_ERR_HCA, CTRL_FAILED, CTRL_COMPLETED = \
    0, 1, 2, 3, 4, 6, 7, 8
CTRL_WORDS = 9
MAX_HCAS = 2


class RoceError(RuntimeError):
    """A RoCE collective failed at run time (timeout or proxy error); the runtime is poisoned."""


class SetupError(RuntimeError):
    """RoCE setup failed on some rank (every rank raises it, with every rank's reason)."""


def _log(msg: str) -> None:
    print(f"[roce] {msg}", file=sys.stderr, flush=True)


# -- knobs ------------------------------------------------------------------------------------------------------------
def backend(raw: str | None = None) -> str:
    v = (os.environ.get(BACKEND_ENV, "") if raw is None else raw).strip().lower() or "nccl"
    if v not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV}={v!r}: expected nccl or roce")
    return v


@dataclass(frozen=True)
class Settings:
    max_bytes: int = 256 * 1024
    timeout_s: float = 120.0
    hcas: int = 2
    blocks: int = 8
    threads: int = 512
    traffic_class: int = 0
    fallback: str = "nccl"
    cpu: int = -1
    hca_spec: str = ""

    def agreed(self) -> list[int]:
        """The integers both ranks must share (everything but the proxy's CPU and the device names)."""

        return [self.max_bytes, int(round(self.timeout_s * 1000)), self.hcas, self.blocks, self.threads,
                self.traffic_class, int(self.fallback == "error")]


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw, 0)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: expected an integer") from None
    if not lo <= v <= hi:
        raise ValueError(f"{name}={v}: expected {lo}..{hi}")
    return v


def settings() -> Settings:
    kb = _int("DSV41_ROCE_MAX_KB", 256, 1, 65536)
    raw_t = os.environ.get("DSV41_ROCE_TIMEOUT_S", "").strip()
    try:
        timeout = float(raw_t) if raw_t else 120.0
    except ValueError:
        raise ValueError(f"DSV41_ROCE_TIMEOUT_S={raw_t!r}: expected seconds") from None
    if not 0.01 <= timeout <= 86400:
        raise ValueError(f"DSV41_ROCE_TIMEOUT_S={timeout}: expected 0.01..86400")
    blocks = _int("DSV41_ROCE_BLOCKS", 8, 1, 64)
    if blocks & (blocks - 1):
        raise ValueError(f"DSV41_ROCE_BLOCKS={blocks}: expected a power of two")   # the arrival counters' modulus
    threads = _int("DSV41_ROCE_THREADS", 512, 32, 1024)
    if threads % 32:
        raise ValueError(f"DSV41_ROCE_THREADS={threads}: expected a multiple of 32")
    tc_name = "DSV41_ROCE_TC" if os.environ.get("DSV41_ROCE_TC", "").strip() else "NCCL_IB_TC"
    fallback = os.environ.get("DSV41_ROCE_FALLBACK", "").strip().lower() or "nccl"
    if fallback not in ("nccl", "error"):
        raise ValueError(f"DSV41_ROCE_FALLBACK={fallback!r}: expected nccl or error")
    return Settings(max_bytes=kb * 1024, timeout_s=timeout, hcas=_int("DSV41_ROCE_HCAS", 2, 1, MAX_HCAS),
                    blocks=blocks, threads=threads, traffic_class=_int(tc_name, 0, 0, 255), fallback=fallback,
                    cpu=_int("DSV41_ROCE_CPU", -1, -1, 4095), hca_spec=os.environ.get("DSV41_ROCE_HCA", ""))


def marker() -> Path | None:
    raw = os.environ.get("DSV41_ROCE_MARK")
    if raw is None:
        return Path(DEFAULT_MARK) if Path(DEFAULT_MARK).parent.is_dir() else None
    return Path(raw) if raw.strip() else None


# -- HCAs and GIDs ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Hca:
    name: str
    gid_index: int
    ipv4: str | None = None
    prefix: int = 24
    netdev: str | None = None

    def key(self) -> tuple:
        """What pairs this port with a peer's: its IPv4 subnet (else its name)."""

        if self.ipv4 is None:
            return ("name", self.name)
        ip = int.from_bytes(socket.inet_aton(self.ipv4), "big")
        mask = (0xFFFFFFFF << (32 - self.prefix)) & 0xFFFFFFFF
        return ("net", ip & mask, self.prefix)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def ipv4_of_gid(gid: str) -> str | None:
    """``0000:...:ffff:c633:640a`` -> ``198.51.100.10``; None for a GID that is not IPv4-mapped."""

    h = gid.replace(":", "").lower()
    if len(h) != 32 or h[:20] != "0" * 20 or h[20:24] != "ffff":
        return None
    return ".".join(str(int(h[24 + 2 * i:26 + 2 * i], 16)) for i in range(4))


def _prefix(netdev: str | None) -> int:
    """The IPv4 prefix length of ``netdev`` (SIOCGIFNETMASK), 24 when unknown."""

    if not netdev:
        return 24
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            res = fcntl.ioctl(s.fileno(), 0x891B, struct.pack("256s", netdev.encode()[:15]))
        return bin(int.from_bytes(res[20:24], "big")).count("1")
    except OSError:
        return 24


def parse_hca_spec(spec: str) -> list[tuple[str, int | None]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, gid = item.partition(":")
        try:
            out.append((name.strip(), int(gid) if gid.strip() else None))
        except ValueError:
            raise ValueError(f"DSV41_ROCE_HCA: {item!r}: expected name or name:gid_index") from None
    return out


def detect(spec: str = "", root: str | None = None) -> list[Hca]:
    """Active RoCE ports with their RoCE v2 GID index (the lowest IPv4-mapped one; ``name:gid`` in ``spec`` forces
    it). ``spec`` (``DSV41_ROCE_HCA``) restricts to the named devices, in its order."""

    wanted = parse_hca_spec(spec)
    base = Path(root or SYSFS)
    names = [n for n, _ in wanted] if wanted else sorted(p.name for p in base.iterdir()) if base.is_dir() else []
    forced = dict(wanted)
    out = []
    for name in names:
        port = base / name / "ports" / "1"
        state = _read(port / "state")
        if state is None or "ACTIVE" not in state:
            if wanted:
                raise ValueError(f"DSV41_ROCE_HCA: {name} port 1 is {state or 'missing'}, not ACTIVE")
            continue
        link = _read(port / "link_layer")
        if link is not None and link != "Ethernet":
            continue
        try:
            indices = sorted(int(x) for x in os.listdir(port / "gids"))
        except OSError:
            indices = []
        chosen = None
        for i in indices:
            gid = _read(port / "gids" / str(i))
            if gid is None:
                continue
            if forced.get(name) is not None and i != forced[name]:
                continue
            if forced.get(name) is None and _read(port / "gid_attrs" / "types" / str(i)) != "RoCE v2":
                continue
            ip = ipv4_of_gid(gid)
            if ip is None and forced.get(name) is None:
                continue
            netdev = _read(port / "gid_attrs" / "ndevs" / str(i))
            chosen = Hca(name, i, ip, _prefix(netdev) if ip else 24, netdev)
            break
        if chosen is None and forced.get(name) is not None:
            chosen = Hca(name, int(forced[name]))
        if chosen is None:
            if wanted:
                raise ValueError(f"DSV41_ROCE_HCA: {name} has no IPv4-mapped RoCE v2 GID")
            continue
        out.append(chosen)
    return out


def pair(lists: list[list[Hca]], most: int) -> list[list[int]]:
    """Per rank, the indices of the HCAs to use, HCA h of every rank on one subnet (rank 0's order); at most
    ``most``. The same answer on every rank (a function of every rank's list)."""

    chosen: list[list[int]] = [[] for _ in lists]
    for i, h0 in enumerate(lists[0]):
        if len(chosen[0]) >= most:
            break
        picks = [i]
        for r in range(1, len(lists)):
            j = next((j for j, h in enumerate(lists[r]) if j not in chosen[r] and h.key() == h0.key()), None)
            if j is None:
                break
            picks.append(j)
        if len(picks) == len(lists):
            for r, j in enumerate(picks):
                chosen[r].append(j)
    return chosen


# -- the extension ----------------------------------------------------------------------------------------------------
_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        here = Path(__file__).parent
        # Keep the built extension out of the container's ephemeral home so a gate run reuses it:
        # /app/.triton is the host-mounted cache, tools/roce/.build the native/host case.
        if not os.environ.get("TORCH_EXTENSIONS_DIR"):
            os.environ["TORCH_EXTENSIONS_DIR"] = (str(Path("/app/.triton/roce-ext"))
                                                  if Path("/app/.triton").is_dir() else str(here / ".build"))
        _EXT = load(name="dsv41_roce_v1", sources=[str(here / "roce.cpp"), str(here / "roce.cu")],
                    extra_cflags=["-O2"], extra_cuda_cflags=["-O3"], extra_ldflags=["-libverbs"], verbose=False)
    return _EXT


def _align(v: int, a: int) -> int:
    return (int(v) + a - 1) // a * a


def grid_blocks(nbytes: int, threads: int, most: int) -> int:
    """A power-of-two grid with about two 16-byte packs a thread (b12x's choice), at most ``most`` blocks."""

    packs = max(1, (int(nbytes) + PACK - 1) // PACK)
    need = max(1, (packs + 2 * threads - 1) // (2 * threads))
    return min(1 << (need - 1).bit_length(), most)


class Runtime:
    """This rank's pinned region, device counters, kernel launcher and RDMA proxy (``hcas``), or, for tests on one
    GPU, a loop-back thread that plays the peers (``loop_xor``: every peer's shard = this rank's XOR the key)."""

    def __init__(self, *, rank: int, world: int, hcas: list[Hca], s: Settings, loop_xor: int | None = None,
                 loop_stall_after: int = -1, n_hca: int | None = None) -> None:
        ext = _ext()
        self.ext, self.rank, self.world, self.s = ext, rank, world, s
        self.hcas = list(hcas)
        self.n_hca = len(self.hcas) if n_hca is None else int(n_hca)
        if not 1 <= self.n_hca <= MAX_HCAS:
            raise ValueError(f"roce: {self.n_hca} HCAs")
        self.slot_bytes = _align(max(s.max_bytes, PACK), SLOT_ALIGN)
        (self.recv_off, self.flag_off, self.send_off, self.ctrl_off, self.total, self.flag_stride,
         self.slots) = (int(v) for v in ext.layout(world, self.slot_bytes))
        self.host, self.dev = (int(v) for v in ext.alloc_pinned(self.total))
        self._freed = False
        self.proxy = None
        self.loop = None
        self.classes = s.blocks.bit_length()
        # epoch, stage arrivals per grid size, tail arrivals per grid size, poison
        self.counters = torch.zeros(2 + 2 * self.classes, dtype=torch.int32, device="cuda")
        base = self.counters.data_ptr()
        self._epoch_addr, self._poison_addr = base, base + 4 * (1 + 2 * self.classes)
        words = (ctypes.c_uint32 * CTRL_WORDS).from_address(self.host + self.ctrl_off)
        self.ctrl = np.frombuffer(words, dtype=np.uint32)
        nflag = world * self.slots * MAX_HCAS * self.flag_stride // 4
        self.flags = np.frombuffer((ctypes.c_uint32 * nflag).from_address(self.host + self.flag_off), dtype=np.uint32)
        d = self.dev
        self.launcher = ext.Launcher(d + self.recv_off, d + self.flag_off, d + self.send_off, d + self.ctrl_off,
                                     self.slot_bytes, self._epoch_addr, self._poison_addr, world, rank, self.n_hca,
                                     self.flag_stride, self.slots, s.threads)
        self.timeout_ns = int(s.timeout_s * 1e9)
        self.lock = threading.Lock()
        self.event = torch.cuda.Event()
        self._last: tuple | None = None           # (stream, capture id) of the last collective
        self.ops = 0
        self._reported = False
        self.marking = True                        # a run-time failure leaves the marker (not the load-time probe)
        try:
            if loop_xor is not None:
                self.loop = ext.Loop(self.host, self.recv_off, self.flag_off, self.send_off, self.ctrl_off,
                                     self.slot_bytes, world, rank, self.n_hca, self.flag_stride, self.slots,
                                     int(loop_xor) & 0xFF, int(loop_stall_after))
            else:
                self.proxy = ext.Proxy(world, rank, [h.name for h in self.hcas], [h.gid_index for h in self.hcas],
                                       s.traffic_class, self.host, self.total, self.slot_bytes)
        except Exception:
            self._free()
            raise

    # -- setup
    def blob(self) -> bytes:
        return bytes(self.proxy.local_blob())

    def connect(self, blobs: list[bytes]) -> None:
        self.proxy.connect(b"".join(blobs))
        self.proxy.start(self.s.cpu)

    # -- the collective
    def _counter_addrs(self, blocks: int) -> tuple[int, int]:
        c = blocks.bit_length() - 1
        return self._epoch_addr + 4 * (1 + c), self._epoch_addr + 4 * (1 + self.classes + c)

    def _order(self) -> None:
        """One epoch, one doorbell, two slots: collectives must run in launch order even across streams. Eager, or
        both in one capture: a switch of stream waits for the previous stream (an event recorded there now covers
        the last collective). A capture starts after a device sync, so its first collective needs no edge."""

        cur = torch.cuda.current_stream()
        cid = int(self.ext.capture_id()) if torch.cuda.is_current_stream_capturing() else 0
        last = self._last
        if last is not None and last[0] != cur and last[1] == cid:
            self.event.record(last[0])
            cur.wait_event(self.event)
        self._last = (cur, cid)

    def gather(self, send: torch.Tensor, recv: torch.Tensor, timeout_ns: int | None = None) -> None:
        """recv [world x n] <- every rank's send [n], rank order (NCCL's all-gather), on the current stream."""

        nbytes = send.numel() * send.element_size()
        if recv.numel() * recv.element_size() != self.world * nbytes or recv.dtype != send.dtype:
            raise ValueError("roce all_gather: recv must hold world x send of the same dtype")
        if not (send.is_contiguous() and recv.is_contiguous() and send.is_cuda and recv.is_cuda):
            raise ValueError("roce all_gather: contiguous CUDA tensors only")
        if nbytes == 0:
            return
        with self.lock:
            capturing = torch.cuda.is_current_stream_capturing()
            if not capturing:
                self.check()
            self._order()
            blocks = grid_blocks(nbytes, self.s.threads, self.s.blocks)
            stage, tail = self._counter_addrs(blocks)
            self.launcher.run(send.data_ptr(), recv.data_ptr(), nbytes, blocks, stage, tail,
                              self.timeout_ns if timeout_ns is None else int(timeout_ns))
            self.ops += 1

    # -- health
    @property
    def failed(self) -> bool:
        return bool(self.ctrl[CTRL_FAILED]) or (self.proxy is not None and self.proxy.failed())

    def check(self) -> None:
        """Raise ``RoceError`` if a wait timed out or the proxy stopped on an error (two host memory reads). Reads no
        device memory and never synchronizes, so it is safe while a kernel is still running."""

        proxy_failed = self.proxy is not None and self.proxy.failed()
        if not self.ctrl[CTRL_FAILED] and not proxy_failed:
            return
        msg = self.diagnose()
        if not self._reported:
            self._reported = True
            _log(msg)
            mark = marker() if self.marking else None
            if mark is not None:
                try:
                    mark.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} rank {self.rank}: {msg}\n")
                except OSError:
                    pass
        raise RoceError(msg)

    def snapshot(self) -> dict:
        """The control record, every flag word per (peer, slot, HCA) and the proxy's counters, from host memory."""

        flags = {}
        for p in range(self.world):
            if p == self.rank:
                continue
            for slot in range(self.slots):
                for h in range(self.n_hca):
                    flags[f"{p}/{slot}/{h}"] = int(self.flags[self._flag_word(p, slot, h)])
        out = {"doorbell": int(self.ctrl[CTRL_SEQ]), "completed": int(self.ctrl[CTRL_COMPLETED]),
               "failed": int(self.ctrl[CTRL_FAILED]), "err_seq": int(self.ctrl[CTRL_ERR_SEQ]),
               "err_peer": int(self.ctrl[CTRL_ERR_PEER]), "err_hca": int(self.ctrl[CTRL_ERR_HCA]),
               "flags": flags, "ops": self.ops}
        if self.proxy is not None:
            st = [int(v) for v in self.proxy.stats()]
            out.update(ops_posted=st[0], writes_completed=st[1], proxy_last_seq=st[2],
                       per_hca=[{"writes": st[3 + 2 * h], "bytes": st[4 + 2 * h]} for h in range(self.n_hca)],
                       proxy_error=self.proxy.error())
        return out

    def _flag_word(self, peer: int, slot: int, hca: int) -> int:
        return ((peer * self.slots + slot) * self.n_hca + hca) * self.flag_stride // 4

    def diagnose(self) -> str:
        snap = self.snapshot()
        if self.proxy is not None and self.proxy.failed():
            return (f"RoCE proxy on rank {self.rank} stopped: {snap.get('proxy_error')} (doorbell "
                    f"{snap['doorbell']}, completed {snap['completed']}); the runtime is unusable")
        seq, peer, h = snap["err_seq"], snap["err_peer"], snap["err_hca"]
        seen = int(self.flags[self._flag_word(peer, seq & 1, h)]) if peer < self.world and h < self.n_hca else -1
        name = self.hcas[h].name if h < len(self.hcas) else f"HCA {h}"
        where = ("the flag HAS reached this host's memory (the GPU did not observe it: sparkring #278's signature)"
                 if seen == seq else f"the flag never arrived (host memory holds {seen})")
        posted = f", this rank's proxy posted up to {snap['proxy_last_seq']}" if "proxy_last_seq" in snap else ""
        return (f"RoCE all-gather on rank {self.rank} timed out after {self.s.timeout_s:g} s waiting for rank "
                f"{peer} on {name} at sequence {seq}: {where}; doorbell {snap['doorbell']}, completed "
                f"{snap['completed']}{posted}. The runtime is poisoned (later exchanges do nothing) and this step's "
                f"data are not trusted: restart both ranks (DSV41_COMM_BACKEND=nccl avoids RoCE; "
                f"{marker() or 'DSV41_ROCE_MARK'} makes the next start use NCCL)")

    # -- teardown
    def _free(self) -> None:
        if not self._freed:
            self._freed = True
            self.ext.free_pinned(self.host)

    def close(self, free: bool = True) -> None:
        """Stop the proxy (or loop) and release the queue pairs; ``free``: also the pinned region (only when no
        kernel can still use it: after a device sync)."""

        for x in (self.proxy, self.loop):
            if x is not None:
                x.destroy()
        self.proxy = self.loop = None
        if free:
            self._free()


class RoceComm:
    """The engine's communicator with the RoCE runtime beside NCCL: ``all_gather`` (control exchanges) stays on
    ``base``; ``all_gather_fast`` (the model's exchanges) goes over RoCE when one rank's shard is at most
    ``max_bytes`` (a decision of the size alone, so both ranks take the same path)."""

    def __init__(self, base, rt, max_bytes: int) -> None:
        self.base, self.rt, self.max_bytes = base, rt, int(max_bytes)
        self.rank, self.world = base.rank, base.world
        self.fast_ops = self.slow_ops = 0

    def __getattr__(self, name):                   # store, lib, comm, ... of the NCCL communicator
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        self.base.all_gather(send, recv)

    def all_gather_fast(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        n = send.numel() * send.element_size()
        if 0 < n <= self.max_bytes and send.is_contiguous() and recv.is_contiguous():
            self.rt.gather(send, recv)
            self.fast_ops += 1
        else:
            self.base.all_gather(send, recv)
            self.slow_ops += 1

    def barrier(self) -> None:
        self.base.barrier()
        self.rt.check()

    def check(self) -> None:
        self.rt.check()


# -- collective setup -------------------------------------------------------------------------------------------------
def _dev(base) -> str:
    return getattr(base, "device", "cuda")      # host-only tests pass a communicator on the CPU


def _gather_ints(base, values: list[int]) -> list[list[int]]:
    mine = torch.tensor(values, dtype=torch.int64, device=_dev(base))
    got = torch.empty((base.world * len(values),), dtype=torch.int64, device=_dev(base))
    base.all_gather(mine, got)
    flat = got.tolist()
    return [flat[r * len(values):(r + 1) * len(values)] for r in range(base.world)]


def exchange(base, obj) -> list:
    """Every rank's JSON-able ``obj`` in rank order, over ``base``'s all-gather."""

    data = json.dumps(obj).encode()
    lens = [v[0] for v in _gather_ints(base, [len(data)])]
    width = max(1, (max(lens) + 3) // 4)
    buf = np.zeros((width * 4,), dtype=np.uint8)
    buf[:len(data)] = np.frombuffer(data, dtype=np.uint8)
    mine = torch.from_numpy(buf.view(np.int32).copy()).to(_dev(base))
    got = torch.empty((base.world * width,), dtype=torch.int32, device=_dev(base))
    base.all_gather(mine, got)
    raw = got.cpu().numpy().tobytes()
    return [json.loads(raw[r * width * 4:r * width * 4 + lens[r]]) for r in range(base.world)]


def _verdict(base, error: str | None, what: str) -> list:
    got = exchange(base, {"error": error})
    bad = [f"rank {r}: {g['error']}" for r, g in enumerate(got) if g.get("error")]
    if bad:
        raise SetupError(f"{what}: " + "; ".join(bad))
    return got


def pattern(rank: int, nbytes: int, salt: int) -> torch.Tensor:
    """Deterministic bytes a rank sends in the probe."""

    i = torch.arange(nbytes, dtype=torch.int64, device="cuda")
    return ((i * 131 + rank * 71 + salt * 29 + (i >> 8) * 7) & 0xFF).to(torch.uint8)


def probe(rt: Runtime, base, sizes=None, timeout_s: float = PROBE_TIMEOUT_S) -> float:
    """Gathers of several sizes (odd, unaligned, the largest slot) compared byte for byte with NCCL's result for the
    same inputs; returns the mean eager latency (us) of 16-KiB gathers. Raises on any difference or timeout."""

    sizes = sizes or [4, 7, 100, 16 * 1024, 16 * 1024 + 4, min(rt.s.max_bytes, 128 * 1024), rt.s.max_bytes]
    timeout_ns = int(timeout_s * 1e9)
    for salt, n in enumerate(sizes):
        off = (4 if n % 16 else 0) if n % 4 == 0 else 1     # unaligned views too (NCCL takes any pointer)
        src = torch.empty((n + off,), dtype=torch.uint8, device="cuda")
        src[off:] = pattern(rt.rank, n, salt)
        send = src[off:]
        mine = torch.empty((rt.world * n,), dtype=torch.uint8, device="cuda")
        ref = torch.empty_like(mine)
        rt.gather(send, mine, timeout_ns)
        base.all_gather(send, ref)
        torch.cuda.synchronize()
        rt.check()
        if not torch.equal(mine, ref):
            bad = int((mine != ref).nonzero()[0].item())
            raise RuntimeError(f"probe of {n} bytes: RoCE result differs from NCCL's at byte {bad}")
    x = pattern(rt.rank, 16 * 1024, 99)
    y = torch.empty((rt.world * x.numel(),), dtype=torch.uint8, device="cuda")
    for _ in range(8):
        rt.gather(x, y, timeout_ns)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(64):
        rt.gather(x, y, timeout_ns)
    torch.cuda.synchronize()
    rt.check()
    return (time.perf_counter() - t) / 64 * 1e6


def connect(base, s: Settings) -> RoceComm:
    """Collective: detect and pair the HCAs, open and connect the queue pairs, start the proxy, probe. Every rank
    raises ``SetupError`` (with every rank's reason) when any rank fails."""

    rank, world = base.rank, base.world
    try:
        mine = [h.__dict__ for h in detect(s.hca_spec)]
        err = None if mine else f"no ACTIVE RoCE port with an IPv4-mapped RoCE v2 GID under {SYSFS}"
    except Exception as exc:  # noqa: BLE001 - reported collectively
        mine, err = [], f"{type(exc).__name__}: {exc}"
    got = exchange(base, {"hcas": mine, "error": err})
    bad = [f"rank {r}: {g['error']}" for r, g in enumerate(got) if g.get("error")]
    if bad:
        raise SetupError("HCA detection: " + "; ".join(bad))
    lists = [[Hca(**h) for h in g["hcas"]] for g in got]
    chosen = pair(lists, s.hcas)
    if not chosen[0]:
        raise SetupError("no pair of RoCE ports on a common subnet: " +
                         "; ".join(f"rank {r}: {[(h.name, h.ipv4) for h in lst]}" for r, lst in enumerate(lists)))
    hcas = [lists[rank][j] for j in chosen[rank]]
    rt, err, blob = None, None, ""
    try:
        rt = Runtime(rank=rank, world=world, hcas=hcas, s=s)
        blob = rt.blob().hex()
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    try:
        got = exchange(base, {"blob": blob, "error": err})
        bad = [f"rank {r}: {g['error']}" for r, g in enumerate(got) if g.get("error")]
        if bad:
            raise SetupError("queue pairs: " + "; ".join(bad))
        err = None
        try:
            rt.connect([bytes.fromhex(g["blob"]) for g in got])
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        _verdict(base, err, "connect")
        err, us = None, 0.0
        rt.marking = False                         # a failed probe falls back now; it need not pin later starts
        try:
            us = probe(rt, base)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        _verdict(base, err, "probe")
        rt.marking = True
    except BaseException:
        if rt is not None:
            torch.cuda.synchronize()
            rt.close()
        raise
    if rank == 0:
        ports = ", ".join(f"{h.name} (GID {h.gid_index}, {h.ipv4})" for h in hcas)
        _log(f"all-gathers of up to {s.max_bytes // 1024} KiB a rank over RoCE on {ports}; timeout "
             f"{s.timeout_s:g} s; probe {us:.1f} us eager at 16 KiB")
    return RoceComm(base, rt, s.max_bytes)


def select(base):
    """``DSV41_COMM_BACKEND``: ``base`` (NCCL) or a ``RoceComm`` over it. Collective (both ranks call it at the same
    point); refuses to start when the ranks' settings differ."""

    want = backend()
    s = settings()
    mark = marker()
    marked = mark is not None and mark.exists()
    both = _gather_ints(base, [BACKENDS.index(want)] + s.agreed() + [int(marked)])
    if any(b[:-1] != both[0][:-1] for b in both):
        raise RuntimeError("the two ranks were started with different DSV41_COMM_BACKEND / DSV41_ROCE_* "
                           f"settings: {[b[:-1] for b in both]} (backend, max bytes, timeout ms, HCAs, blocks, "
                           "threads, traffic class, fallback=error)")
    if want == "nccl":
        return base
    if any(b[-1] for b in both):
        who = [r for r, b in enumerate(both) if b[-1]]
        _log(f"a RoCE failure marker is present on rank(s) {who} ({mark}): serving on NCCL; delete it to use RoCE")
        return base
    try:
        return connect(base, s)
    except SetupError as exc:
        if s.fallback == "error":
            raise RuntimeError(f"DSV41_COMM_BACKEND=roce: setup failed ({exc}); DSV41_ROCE_FALLBACK=error") \
                from None
        _log(f"setup failed, serving on NCCL: {exc}")
        return base
