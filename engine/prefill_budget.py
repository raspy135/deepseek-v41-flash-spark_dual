"""Conservative request-boundary prefill sizing, agreed across TP ranks.

This is a scratch estimate, not an OOM guarantee. Plan once before Engram
read-ahead so its row boundaries and the model's boundaries remain identical.
Never count the shared CUDA graph pool's inactive bytes as freely reusable.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillBudget:
    enabled: bool = False
    floor_gib: float = 4.0
    min_rows: int = 256
    row_mib: float = 2.0

    @classmethod
    def from_env(cls):
        flag = os.environ.get('DSV41_PREFILL_ADAPT', '0')
        if flag not in ('0', '1'):
            raise ValueError('DSV41_PREFILL_ADAPT must be 0 or 1')
        value = cls(flag == '1', float(os.environ.get('DSV41_PREFILL_FLOOR_GIB', '4')),
                    int(os.environ.get('DSV41_PREFILL_MIN_ROWS', '256')),
                    float(os.environ.get('DSV41_PREFILL_ROW_MIB', '2')))
        import math
        if (not math.isfinite(value.floor_gib) or value.floor_gib <= 0 or
                not math.isfinite(value.row_mib) or value.row_mib <= 0 or
                value.min_rows < 256 or value.min_rows % 128):
            raise ValueError('invalid prefill memory budget')
        return value

    def choose(self, maximum, context, available):
        """Use the tighter rank, not an average. None means unavailable: use minimum.

        Eight bytes per visible key/query prices index scores plus a scratch copy;
        row_mib covers gathered sparse KV, activations and kernel workspace. It is
        intentionally configurable and conservative, not a fitted universal bound.
        """
        if maximum < 1 or context < 0 or not available:
            raise ValueError('invalid prefill dimensions or rank memory reports')
        minimum = min(maximum, self.min_rows)
        if not self.enabled:
            return dict(rows=maximum, enabled=False)
        usable = min(0 if v is None else max(0, int(v)) for v in available)
        row_bytes = int(self.row_mib * 2**20) + 8 * context
        headroom = max(0, usable - int(self.floor_gib * 2**30))
        rows = maximum
        while rows > minimum and rows * row_bytes > headroom:
            rows = max(minimum, (rows // 2 // 128) * 128)
        return dict(enabled=True, rows=rows, maximum=maximum, context=context,
                    available_bytes=usable, floor_bytes=int(self.floor_gib * 2**30),
                    estimated_scratch_bytes=rows * row_bytes,
                    estimate_fits=rows * row_bytes <= headroom)
