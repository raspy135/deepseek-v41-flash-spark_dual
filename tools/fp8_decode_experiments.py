"""Isolated, native-precision FP8 decode experiments; not imported by serving."""
import torch
import triton
import triton.language as tl
from fp8_linear import _act_qdq_tile, _fp8_linear_kernel, decode_block_n

@triton.jit
def _qdq(X,Y,M:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
 r=tl.program_id(0); k=tl.program_id(1)*BK+tl.arange(0,BK)
 x=tl.load(X+r*K+k,k<K,0)
 q=_act_qdq_tile(tl.reshape(x,(1,BK)),1,BK)
 tl.store(Y+r*K+k,tl.reshape(q,(BK,)),k<K)

def qdq(x):
 y=torch.empty_like(x)
 _qdq[(x.shape[0],triton.cdiv(x.shape[1],128))](x,y,x.shape[0],x.shape[1],128,num_warps=4)
 return y

@triton.jit
def _partial(X,W,S,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,ACT:tl.constexpr):
 n=tl.program_id(0)*BN+tl.arange(0,BN); kt=tl.program_id(1)
 m=tl.arange(0,16);k=kt*BK+tl.arange(0,BK)
 x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0.0)
 if ACT:x=_act_qdq_tile(x,16,BK)
 w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0.0)
 sk=kt*(BK//32)+tl.arange(0,BK//32)
 s=tl.load(S+(n[:,None]//32)*(K//32)+sk[None,:],(n[:,None]<N)&(sk[None,:]<K//32),127).to(tl.int32)
 scale=tl.exp2((s-127).to(tl.float32)).to(tl.bfloat16)
 w=tl.reshape(tl.reshape(w.to(tl.bfloat16),(BN,BK//32,32))*scale[:,:,None],(BN,BK))
 y=tl.dot(x,tl.trans(w),out_dtype=tl.float32)
 tl.store(P+kt*M*N+m[:,None]*N+n[None,:],y,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _ordered(P,Y,M:tl.constexpr,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);a=tl.full((B,),0,tl.float32)
 for k in range(T):a=a+tl.load(P+k*M*N+i,i<M*N,0)
 tl.store(Y+i,a,i<M*N)

def parallel(x,w,act=False,bn=64,dtype=torch.bfloat16):
 m,k=x.shape;n=w.N;bk=128 if k%128==0 else 64
 p=torch.empty((triton.cdiv(k,bk),m,n),device=x.device,dtype=torch.float32)
 y=torch.empty((m,n),device=x.device,dtype=dtype)
 _partial[(triton.cdiv(n,bn),triton.cdiv(k,bk))](x,w.w,w.s,p,m,n,k,bn,bk,act,num_warps=4)
 _ordered[(triton.cdiv(m*n,256),)](p,y,m,n,triton.cdiv(k,bk),256,num_warps=4,enable_fp_fusion=False)
 return y

# Bit-math candidate: experimental until exhaustive activation checks pass.
@triton.jit
def _bit_qdq_tile(x, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """v41_ref.act_qdq_fp8 on a [BLOCK_M, BLOCK_K] bf16 tile whose K offset is a multiple of 32.

    Bit-identical to the torch spelling, which is why each step is spelled the way it is:
    amax is order-independent; `/` and log2/exp2 go through libdevice's IEEE div_rn / log2f /
    exp2f like torch's CUDA kernels (Triton's default fp32 `/` is div.full and tl.log2 is
    lg2.approx -- both can move ceil(log2(.)) across an integer); multiplying by the power-of-two
    scale is exact; e4m3 conversion is round-to-nearest-even on both sides; the product is exact
    in bf16 (3 mantissa bits). tools/test_fp8_act_qdq.py checks it through an identity weight.

    Triton builds libdevice with flush-to-zero (the PTX has div.rn.ftz, ex2.approx.ftz); torch
    does not. That cannot change a result here: amax / 448 >= 2.2e-7 is normal; exp2 only sees
    integers and returns exact powers of two >= 2^-22; a subnormal x, or a quotient that
    underflows, quantizes to a same-signed zero in e4m3 either way. And with bf16 x, amax / 448
    is either an exact power of two (exact log2 in both libraries) or >= ~0.4% away from one, so
    a last-ulp log2 difference cannot move ceil(). An fp32 x would lose that margin, which is one
    reason fp8_linear refuses it.
    """
    xg = tl.reshape(x.to(tl.float32), (BLOCK_M, BLOCK_K // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-4)
    # The expensive IEEE div/log2 run once per 32-group. Per element, x / s is spelled
    # x * 2^-e: s = 2^e is a power of two, so the quotient and the product are the same real
    # number and round identically. The first version divided per element with div_rn inside the
    # K loop and was 2.3x SLOWER than the torch spelling (wq_a 56 -> 129 us, 2026-09-23).
    bits = amax.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 255) - 135 + ((bits & 0x7fffff) > 0x600000).to(tl.int32)
    s = ((e + 127) << 23).to(tl.float32, bitcast=True)[:, :, None]
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)[:, :, None]
    q = tl.minimum(tl.maximum(xg * inv, -448.0), 448.0)
    q = q.to(tl.float8e4nv).to(tl.float32) * s
    return tl.reshape(q, (BLOCK_M, BLOCK_K)).to(tl.bfloat16)


@triton.jit
def _bit_linear_kernel(X, W, S, Y, M, N, K,
                       stride_xm, stride_wn, stride_sn, stride_ym,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       ACT_QDQ: tl.constexpr = False):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rs = tl.arange(0, BLOCK_K // 32)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_scale_row = rn // 32
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None] & (kk[None, :] < K), other=0.0)
        if ACT_QDQ:
            x = _bit_qdq_tile(x, BLOCK_M, BLOCK_K)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & (kk[None, :] < K), other=0.0)
        # one ue8m0 scale per 32-wide K group: exact in bf16 (3 mantissa bits, power-of-two scale)
        s = tl.load(S + n_scale_row[:, None] * stride_sn + (k0 // 32 + rs)[None, :],
                    mask=n_mask[:, None] & ((k0 // 32 + rs)[None, :] < (K + 31) // 32), other=127).to(tl.int32)
        scale = (s << 7).to(tl.uint16).to(tl.bfloat16, bitcast=True)  # [BLOCK_N, BLOCK_K // 32]
        w3 = tl.reshape(w.to(tl.bfloat16), (BLOCK_N, BLOCK_K // 32, 32)) * scale[:, :, None]
        wb = tl.reshape(w3, (BLOCK_N, BLOCK_K))
        acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


