"""DSV41_ABLIT_WOB: serve the abliterated attention output projection from a Keys overlay.

`drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed` publishes
`wo_b_l10_35.safetensors` (~1.1 GB: fp8 e4m3 `weight` + e8m0 `scale`, the native checkpoint's own
storage format) for `deepseek-ai/DeepSeek-V4.1-Flash`. Replacing `layers.{10..35}.attn.wo_b`
with it is the whole abliteration: experts, Engram, the MTP/DSpark heads, vision and every other
tensor stay the checkpoint's.

It is applied in the loader, before the dense-format step, so `DSV41_DENSE_FP4` re-quantizes the
abliterated fp8 exactly the way it re-quantizes the stock weights. No copy of the checkpoint is
made (the native tree here is 510 GB and the box has no room for a second one) and the stock tree
is never written.

This changes the model by design -- that is the point -- so it is off by default and pinned in the
EP2 boot guard (the file's digest too, so the two ranks cannot load different overlays).
"""
from __future__ import annotations

import hashlib
import os

from safetensors import safe_open

FIRST, LAST = 10, 35


def path() -> str | None:
    return os.environ.get("DSV41_ABLIT_WOB") or None


def enabled() -> bool:
    return path() is not None


def expected() -> list[str]:
    return [f"layers.{L}.attn.wo_b.{k}" for L in range(FIRST, LAST + 1) for k in ("weight", "scale")]


def digest() -> str:
    """sha256 of the overlay file, 16 hex chars: what the boot guard compares across ranks."""
    p = path()
    if not p:
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def loader(get):
    """Wrap the checkpoint's `get` so the L10-35 wo_b tensors come from the overlay.

    Fails before any weight is used if the file is not exactly the 52 expected tensors under the
    checkpoint's own names, rather than silently serving a half-abliterated model.
    """
    p = path()
    if not p:
        return get
    if not os.path.exists(p):
        raise RuntimeError(f"DSV41_ABLIT_WOB={p} does not exist")
    handle = safe_open(p, framework="pt", device="cpu")
    have = set(handle.keys())
    want = set(expected())
    if have != want:
        missing, extra = sorted(want - have), sorted(have - want)
        raise RuntimeError(
            f"DSV41_ABLIT_WOB={p} is not a native wo_b overlay: missing {len(missing)} "
            f"(e.g. {missing[:2]}), unexpected {len(extra)} (e.g. {extra[:2]}); it must be "
            f"layers.{{{FIRST}..{LAST}}}.attn.wo_b.{{weight,scale}} in the checkpoint's names")

    def wrapped(name):
        return handle.get_tensor(name) if name in have else get(name)

    return wrapped
