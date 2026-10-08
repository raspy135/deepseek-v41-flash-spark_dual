"""Exhaustive BF16 activation-level checks for the experimental bit-scale path."""
import sys
sys.path[:0] = ['/app', '/app/tools']
import torch
import triton
import triton.language as tl
from fp8_linear import _act_qdq_tile

@triton.jit
def check(X, A, B, SCALE_A, SCALE_B, K: tl.constexpr):
    k = tl.program_id(0) * 128 + tl.arange(0, 128)
    x = tl.load(X + k, k < K, 0.0)
    x = tl.reshape(x, (1, 128))
    a = _act_qdq_tile(x, 1, 128)
    b = _act_qdq_tile(x, 1, 128, True)
    tl.store(A + k, tl.reshape(a, (128,)), k < K)
    tl.store(B + k, tl.reshape(b, (128,)), k < K)
    s = k % 256
    sa = tl.exp2((s - 127).to(tl.float32)).to(tl.bfloat16)
    sb = (s << 7).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(SCALE_A + k, sa, k < K)
    tl.store(SCALE_B + k, sb, k < K)

def run(x):
    outs = [torch.empty_like(x) for _ in range(4)]
    check[(triton.cdiv(x.numel(),128),)](x, *outs, x.numel(), num_warps=4)
    for a,b in (outs[:2], outs[2:]):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), int((a.view(torch.int16)!=b.view(torch.int16)).sum())

bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
x = bits.view(torch.bfloat16)
x = x[torch.isfinite(x)]
# Every finite BF16 value as the amax in its own 32-value group, then mixed signs/magnitudes.
run(x.repeat_interleave(32).contiguous())
torch.manual_seed(71)
run(x[torch.randperm(x.numel(), device='cuda')].contiguous())
print('BIT_SCALE_EXHAUSTIVE_PASS', flush=True)
