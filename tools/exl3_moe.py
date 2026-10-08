"""EXL3 routed experts for the engine: the arena and the reference forward.

The arena mirrors ``tools/fp4_moe.py::ExpertArena`` (``slots``, ``bytes_per_slot``,
``load_slot``, a zero null slot) so ``engine/experts.py``'s LRU / transient ring / swaps and
``V41Engine.make_expert_arena`` reuse it unchanged.  Version 1 sizes every slot for the widest
width in the pack (3 bits); a 2-bit expert wastes a third of its words, which the plan accepts
(<= ~4 GB if layers 18-22 are fully resident).  ``self.bits[slot]`` says how many 16-bit words
of a trellis are live, for the reference and the kernels.

``moe_forward_exl3_ref`` is the correctness gate, not the endpoint: it dequantizes each touched
expert with the torch decoder and runs BF16 grouped matmuls with the same SwiGLU limit, routing
weight and combine point as ``fp4_moe.moe_forward``.  The performance path is the packed
grouped kernel in ``tools/exl3_moe_cuda`` (P3/P4).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402

DIM = 5120
INTER = 2304


class Exl3Arena:
    """GPU-resident slots of one rank's EXL3 expert slices (see tools/exl3_ref.py for the layout)."""

    def __init__(self, slots: int, device: torch.device | str = "cuda", tp_rank: int = 0,
                 tp_world: int = 1, bits: float = 3.0, codebook: str = "mul1"):
        self.slots = int(slots)
        self.device = torch.device(device)
        if tp_world not in (1, 2) or not 0 <= tp_rank < tp_world:
            raise ValueError("EXL3 expert TP supports world=1 or 2")
        self.tp_rank, self.tp_world = tp_rank, tp_world
        self.codebook = codebook
        self.slot_bits = float(bits)
        self.shapes = R.record_shapes(self.slot_bits, tp_rank, tp_world)
        self.inter = self.shapes["svh1"][0]          # intermediate columns this rank holds
        self.down_n = self.shapes["svh2"][0]         # hidden outputs this rank holds
        dev = self.device
        i16, f16 = dict(dtype=torch.int16, device=dev), dict(dtype=torch.float16, device=dev)
        self.t1 = torch.empty((slots, *self.shapes["t1"]), **i16)
        self.t3 = torch.empty((slots, *self.shapes["t3"]), **i16)
        self.t2 = torch.empty((slots, *self.shapes["t2"]), **i16)
        self.suh1 = torch.empty((slots, *self.shapes["suh1"]), **f16)
        self.suh3 = torch.empty((slots, *self.shapes["suh3"]), **f16)
        self.suh2 = torch.empty((slots, *self.shapes["suh2"]), **f16)
        self.svh1 = torch.empty((slots, *self.shapes["svh1"]), **f16)
        self.svh3 = torch.empty((slots, *self.shapes["svh3"]), **f16)
        self.svh2 = torch.empty((slots, *self.shapes["svh2"]), **f16)
        # Every slot is a ZERO expert until loaded: zero the scales so an unloaded slot decodes to
        # exactly 0 (the plan's null slot), not to the codebook value of a garbage trellis. This is
        # what makes fastdecode's discarded warm-up pass safe -- it runs _layer_b before _layer_ab
        # fills self.slots, so the table is still zeros -- and it matches the FP4 arena's null slot.
        for _n in ("suh1", "suh3", "suh2", "svh1", "svh3", "svh2"):
            getattr(self, _n).zero_()
        self.bits = [int(self.slot_bits)] * self.slots
        self._loaded = set()
        self._tensors = ("t1", "t3", "t2", "suh1", "suh3", "suh2", "svh1", "svh3", "svh2")

    @property
    def bytes_per_slot(self) -> int:
        """Bytes copied per expert -- the 9 record tensors, not the per-slot `bits` metadata."""

        return sum(getattr(self, n)[0].numel() * getattr(self, n).element_size() for n in self._tensors)

    def load_slot(self, slot: int, record: dict, bits: float, non_blocking: bool = False) -> None:
        """Copy one pack record's 9 tensors into `slot`, zero-padding a 2-bit trellis to the slot width.

        `record` may hold numpy (as ``exl3_ref.PackReader`` returns) or CPU torch tensors; either way
        it is the *rank slice* the pack stores, so no slicing happens here."""
        for name in self._tensors:
            src = record[name]
            if not isinstance(src, torch.Tensor):
                src = torch.tensor(np.asarray(src))      # copies, so a read-only pack view is fine
            dst = getattr(self, name)[slot]
            if name.startswith("t"):
                # Store the trellis COMPACT: a 2-bit tile is 16 uint32, not the 24 a 3-bit slot
                # has room for. The CUDA kernel's tile stride is 4*K2 words, so a padded 2-bit
                # slot would be addressed wrongly; compaction makes K2=4 and K2=6 both exact, and
                # the per-slot trellis pointer makes the different strides legal.
                flat, n = dst.reshape(-1), src.numel()
                if n > flat.numel():
                    raise ValueError(f"{name}: record {src.shape} does not fit slot {tuple(dst.shape)}")
                if n < flat.numel():
                    flat[n:].zero_()
                flat[:n].copy_(src.reshape(-1).to(dst.dtype), non_blocking=non_blocking)
            else:
                dst.copy_(src.to(dst.dtype), non_blocking=non_blocking)
        self.bits[slot] = int(bits)
        self._loaded.add(int(slot))

    def read_slot(self, slot: int) -> dict[str, torch.Tensor]:
        """The stored tensors of one slot, trellis cut back to its real width (for the reference)."""
        if slot < 0:
            raise IndexError(f"EXL3 slot {slot}: a routed pair reached an unmapped (-1) table entry")
        b = int(self.bits[slot])
        if b <= 0:
            raise IndexError(f"EXL3 slot {slot} was never loaded (bits={b}, loaded={slot in self._loaded}); "
                             f"arena has {self.slots} slots, {len(self._loaded)} loaded")
        w = R.tile_words(b)
        out = {}
        for name in self._tensors:
            t = getattr(self, name)[slot]
            if name.startswith("t"):
                n = int(np.prod(t.shape[:-1])) * w          # compact: drop any 3-bit padding
                out[name] = t.reshape(-1)[:n].view(*t.shape[:-1], w)
            else:
                out[name] = t
        return out

    @torch.no_grad()
    def dequant_slot(self, slot: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(w1, w2, w3) as fp32 logical matrices, the rank's slice.

        w1/w3 are [DIM, inter] (gate/up: 5,120 inputs -> this rank's intermediate columns), w2 is
        [INTER, down_n] (down, full intermediate -> this rank's hidden outputs) -- the true weight
        matrices the kernels multiply into, so a caller uses ``x @ w1`` not ``x @ w1.T``."""
        rec = self.read_slot(slot)
        bits = float(self.bits[slot])
        w1 = R.dequantize(rec["t1"], rec["suh1"], rec["svh1"], bits, self.codebook).to(torch.float32)
        w3 = R.dequantize(rec["t3"], rec["suh3"], rec["svh3"], bits, self.codebook).to(torch.float32)
        w2 = R.dequantize(rec["t2"], rec["suh2"], rec["svh2"], bits, self.codebook).to(torch.float32)
        return w1, w2, w3


@torch.no_grad()
def moe_forward_exl3_ref(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor,
                         arena: Exl3Arena, swiglu_limit: float = 10.0,
                         out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Grouped EXL3 MoE by dequantizing each touched expert, mirroring ``fp4_moe``'s contract.

    x bf16 [T, 5120] or fp32; slots int [T, K]; weights fp32 [T, K] -> [T, 5120] in `out_dtype`.
    Under TP-output (the pack's layout) each rank owns half the intermediate columns and half the
    hidden outputs; this all-gathers the bf16 intermediate before the down projection and the
    per-token totals after it, exactly where ``fp4_moe.moe_forward`` does -- so the two ranks keep
    the same BF16 rounding boundary.  Correctness gate only; the packed kernels are P3/P4."""
    assert x.shape[1] == DIM, x.shape
    assert slots.shape == weights.shape
    T, k = slots.shape
    dev, world = x.device, arena.tp_world
    inter, down_n = arena.inter, arena.down_n
    wgt = weights.reshape(-1).float()
    pairs = slots.reshape(-1)
    # one dequant per touched expert; the arena's slot is the expert here (the reference has no LRU)
    decoded = {}
    h = torch.empty((T * k, inter), dtype=torch.bfloat16, device=dev)
    for s in torch.unique(slots).tolist():
        w1, w2, w3 = arena.dequant_slot(int(s))
        decoded[int(s)] = w2
        t_idx, k_idx = torch.nonzero(slots == s, as_tuple=True)
        xs = x[t_idx].float()
        gate = xs @ w1
        up = xs @ w3
        if swiglu_limit > 0:
            up = up.clamp(-swiglu_limit, swiglu_limit)
            gate = gate.clamp(max=swiglu_limit)
        hs = torch.nn.functional.silu(gate) * up * wgt.reshape(T, k)[t_idx, k_idx][:, None]
        h[(t_idx * k + k_idx)] = hs.to(torch.bfloat16)
    if world > 1:
        from engine.collective_rails import group as collective_group
        gathered = torch.empty((world * T * k, inter), dtype=torch.bfloat16, device=dev)
        torch.distributed.all_gather_into_tensor(gathered, h, group=collective_group())
        h = gathered.view(world, T * k, inter).transpose(0, 1).reshape(T * k, world * inter)
    parts = torch.zeros((T, down_n), dtype=torch.float32, device=dev)
    for s in torch.unique(slots).tolist():
        w2 = decoded[int(s)]
        t_idx, k_idx = torch.nonzero(slots == s, as_tuple=True)
        flat = t_idx * k + k_idx
        parts.index_add_(0, t_idx, (h[flat].to(torch.bfloat16) @ w2.to(torch.bfloat16)).float())
    local = parts.to(out_dtype)
    if world == 1:
        return local
    from engine.collective_rails import group as collective_group
    out = torch.empty((world * T, down_n), dtype=out_dtype, device=dev)
    torch.distributed.all_gather_into_tensor(out, local, group=collective_group())
    return out.view(world, T, down_n).transpose(0, 1).reshape(T, DIM)
