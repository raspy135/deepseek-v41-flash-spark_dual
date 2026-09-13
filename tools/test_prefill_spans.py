"""No image span may straddle a prefill chunk boundary, whatever its size or position."""
import os, sys, types as pytypes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import engine.v41_engine as V
from engine.model import MAX_CHUNK

def spans(P, images):
    eng = pytypes.SimpleNamespace(_images=[
        pytypes.SimpleNamespace(start=s, types=torch.zeros(n, dtype=torch.int64)) for s, n in images])
    return V.V41Engine._prefill_spans(eng, P)

def check(tag, P, images):
    out = spans(P, images)
    # 1. the chunks tile [0, P) exactly
    assert out[0][0] == 0 and out[-1][1] == P, (tag, out)
    for (a, b), (c, d) in zip(out, out[1:]):
        assert b == c and a < b, (tag, out)
    # 2. no span crosses a boundary
    bounds = {b for _, b in out[:-1]}
    for s, n in images:
        crossing = [x for x in bounds if s < x < s + n]
        assert not crossing, f"{tag}: span [{s},{s+n}) crossed by {crossing}"
    # 3. ordinary chunks stay within MAX_CHUNK; only a span-bearing chunk may exceed it
    for a, b in out:
        if b - a > MAX_CHUNK:
            assert any(s == a and s + n == b or (s == a and s + n > MAX_CHUNK) for s, n in images), \
                f"{tag}: oversized chunk [{a},{b}) carries no long span"
    print(f"  {tag:34} {len(out):>2} chunks  max {max(b-a for a,b in out):>5}  {out[:4]}{'...' if len(out)>4 else ''}")

print(f"MAX_CHUNK={MAX_CHUNK}")
check("no images", 5000, [])
check("one 990 span mid-chunk", 5000, [(1500, 990)])
check("span would straddle", 5000, [(2000, 990)])
check("two spans, same chunk", 6000, [(100, 990), (1200, 990)])
check("span longer than MAX_CHUNK", 8000, [(1000, 3000)])
check("span at position 0", 5000, [(0, 990)])
check("span exactly at a boundary", 6000, [(2048, 990)])
check("span ending exactly at P", 4000, [(3010, 990)])
check("three images", 12000, [(500, 990), (2600, 990), (7000, 990)])
print("\nALL SPAN-BOUNDARY CASES OK")
