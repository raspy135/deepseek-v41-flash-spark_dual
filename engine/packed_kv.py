"""FP4 KV storage: 16 E2M1 values + one E4M3 scale per group (9 bytes).

Only replaces the existing FP4 QDQ history, never the BF16 window/index keys.
Gather reconstructs BF16 before attention, preserving its reduction order.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['START'])
def _pack(X, Ids, Cache, START, INDEXED: tl.constexpr,
          D: tl.constexpr, STRIDE: tl.constexpr):
    row = tl.program_id(0)
    g = tl.arange(0, D // 16)
    j = tl.arange(0, 16)
    x = tl.load(X + row * D + g[:, None] * 16 + j[None, :]).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 6.0 * 2.0 ** -9)
    scale8 = tl.div_rn(amax, 6.0).to(tl.float8e4nv)
    scale = scale8.to(tl.float32)
    v = tl.minimum(tl.abs(tl.div_rn(x, scale[:, None])), 6.0)
    # torch.bucketize(..., right=False): midpoint ties select the lower code.
    code = ((v > .25).to(tl.int32) + (v > .75).to(tl.int32)
            + (v > 1.25).to(tl.int32) + (v > 1.75).to(tl.int32)
            + (v > 2.5).to(tl.int32) + (v > 3.5).to(tl.int32)
            + (v > 5.).to(tl.int32)) | ((x < 0).to(tl.int32) << 3)
    packed = tl.sum(code.to(tl.uint64) << (j[None, :] * 4), 1)
    scale_bytes = tl.reshape(scale8.to(tl.uint8, bitcast=True), (D // 128, 8))
    scale_words = tl.sum(scale_bytes.to(tl.uint64)
                         << (tl.arange(0, 8)[None, :] * 8), 1)
    dst = tl.load(Ids + row) if INDEXED else START + row
    tl.store(Cache + dst * STRIDE + g, packed.to(tl.int64))
    tl.store(Cache + dst * STRIDE + D // 16 + tl.arange(0, D // 128),
             scale_words.to(tl.int64))


@triton.jit
def _gather(Cache, Ids, Out, N: tl.constexpr, D: tl.constexpr,
            STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = group // (D // 16), group % (D // 16)
    idx = tl.load(Ids + row, row < N, 0)
    word = tl.load(Cache + idx * STRIDE + col, row < N, 0).to(tl.uint64)
    j = tl.arange(0, 16)
    code = ((word[:, None] >> (j[None, :] * 4)) & 15).to(tl.int32)
    mag = code & 7
    value = tl.where(mag < 2, mag * .5,
                    tl.where(mag < 4, mag * .5,
                             tl.where(mag < 6, mag - 2., 2. * mag - 8.)))
    sw = tl.load(Cache + idx * STRIDE + D // 16 + col // 8, row < N, 0).to(tl.uint64)
    sb = ((sw >> ((col % 8) * 8)) & 255).to(tl.uint8)
    scale = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # Set the sign after reconstruction so compiler simplification cannot erase
    # negative zero (the QDQ oracle preserves it for negative values rounded to 0).
    bits = (value * scale[:, None]).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    bits = bits | ((code >= 8).to(tl.uint16) << 15)
    tl.store(Out + group[:, None] * 16 + j[None, :],
             bits.to(tl.bfloat16, bitcast=True), row[:, None] < N)


def write(cache, values, positions):
    """Quantize pre-QDQ BF16 rows directly into packed cache slots."""
    if values.dtype != torch.bfloat16 or not values.is_contiguous():
        raise ValueError('packed KV write requires contiguous BF16 rows')
    d = values.shape[-1]
    indexed = isinstance(positions, torch.Tensor)
    if indexed and not positions.is_contiguous():
        positions = positions.contiguous()
    _pack[(values.numel() // d,)](
        values, positions if indexed else values, cache,
        START=0 if indexed else positions, INDEXED=indexed,
        D=d, STRIDE=cache.shape[1], num_warps=4, enable_fp_fusion=False)


def gather(cache, indices):
    """Same BF16 rows as cache[indices], without a full-history expansion."""
    if cache.dtype != torch.int64:
        return cache[indices]
    indices = indices.contiguous()
    d = cache.shape[1] * 128 // 9
    out = torch.empty((*indices.shape, d), dtype=torch.bfloat16, device=cache.device)
    if indices.numel():
        _gather[(triton.cdiv(out.numel(), 2048),)](
            cache, indices, out, N=indices.numel(), D=d,
            STRIDE=cache.shape[1], BLOCK=128, num_warps=4)
    return out
