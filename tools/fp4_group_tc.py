"""Exact-arithmetic proof for one native32K FP4 expert group, benchmark-only.

Each real token has eight virtual MMA rows. Four FP16 dots contain one product
each, followed by explicit RN pair sums and the eight-leaf tree; do not
aggregate across scale groups or pre-scale.
The CUDA reference extracts the serving group_partial/load/decode helpers so
the proof does not silently compare two rewritten implementations.

Retained negative: the apparently cheaper even/odd two-product MMA variant
changed 854/7,488 raw FP32 group partials on signed logrange inputs on GB10
(2026-10-07), despite 18,304 uniform-code and 4,096 boundary-code checks
passing. Its PTX preserves the explicit add.rn tree; FP16 MMA accumulation
is not IEEE RN-FMA. Results live in results/fp4-group-tc-20261007/proof.json.
An isolated x0=1,x2=1.5*2**-23,w0=w2=1 gives CUDA 1+2**-22 but
the two-product MMA gives 1+2**-23; the explicit tree is add.rn in PTX.

The four-one-product variant passed 35,776 adversarial finite raw FP32 checks,
including that counterexample, but is also a retained performance negative.
A 256-group/N1152 hot helper screen with six balanced timing quartets gave
M1 CUDA 8.063us versus TC 109.108us and M2 CUDA 15.220us versus TC 108.539us.
Sparse virtual rows plus four padded MMAs and layout/gather overhead overwhelm
the weight-reuse benefit in this spelling. No full-K/serving integration is
justified by that result; single-products-proof.json retains the gate/timing.
This isolated hot component excludes UE8M0 scaling, ordered full-K chains,
SiLU/routing and cold actual MoE; it rejects this spelling, not every possible
tensor-core relayout of the algorithm.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import re
import subprocess

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout, DotOperandLayout, NVMMADistributedLayout, SliceLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@g.jit
def _group(X, W, OUT, M: gl.constexpr, N: gl.constexpr, BN: gl.constexpr,
           SINGLE_PRODUCTS: gl.constexpr):
    xl: gl.constexpr = BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
    wl: gl.constexpr = BlockedLayout([2, 1], [4, 8], [1, 4], [0, 1])
    pl: gl.constexpr = BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    ml: gl.constexpr = NVMMADistributedLayout([2, 0], [1, 4], [16, 8])
    al: gl.constexpr = DotOperandLayout(0, ml, 2)
    bl: gl.constexpr = DotOperandLayout(1, ml, 2)
    batch = gl.program_id(1)
    vr = gl.arange(0, 16, layout=SliceLayout(1, xl))
    kx = gl.arange(0, 32, layout=SliceLayout(0, xl))
    x = gl.load(X + batch*M*32 + (vr[:, None]//8)*32 + kx[None, :],
                vr[:, None]//8 < M, 0).to(gl.float16)
    own = kx[None, :]//4 == vr[:, None]%8
    ns = gl.program_id(0)*BN + gl.arange(0, BN, layout=SliceLayout(0, wl))
    kw = gl.arange(0, 32, layout=SliceLayout(1, wl))
    packed = gl.load(W + batch*N*16 + ns[None, :]*16 + kw[:, None]//2,
                     ns[None, :] < N, 0).to(gl.uint16)
    code = (packed >> ((kw[:, None]&1)*4)) & 15
    magnitude = code & 7
    bits = gl.where(magnitude < 2, magnitude*0x3800,
                    0x3c00 + (magnitude-2)*0x0200).to(gl.uint16)
    bits = bits | ((code & 8) << 12)
    weight = gl.convert_layout(bits.to(gl.uint16).to(gl.float16, bitcast=True), bl)
    zero = gl.full((16, BN), 0, gl.float32, ml)
    if SINGLE_PRODUCTS:
        x0 = gl.where(own & (kx[None, :]%4 == 0), x, 0)
        x1 = gl.where(own & (kx[None, :]%4 == 1), x, 0)
        x2 = gl.where(own & (kx[None, :]%4 == 2), x, 0)
        x3 = gl.where(own & (kx[None, :]%4 == 3), x, 0)
        p0 = mma_v2(gl.convert_layout(x0, al), weight, zero)
        p1 = mma_v2(gl.convert_layout(x1, al), weight, zero)
        p2 = mma_v2(gl.convert_layout(x2, al), weight, zero)
        p3 = mma_v2(gl.convert_layout(x3, al), weight, zero)
        pe = p0+p2
        po = p1+p3
    else:
        xe = gl.where(own & (kx[None, :]%2 == 0), x, 0)
        xo = gl.where(own & (kx[None, :]%2 == 1), x, 0)
        pe = mma_v2(gl.convert_layout(xe, al), weight, zero)
        po = mma_v2(gl.convert_layout(xo, al), weight, zero)
    s = gl.convert_layout(pe+po, pl)
    token = gl.arange(0, 2, layout=SliceLayout(1, pl))
    no = gl.program_id(0)*BN + gl.arange(0, BN, layout=SliceLayout(0, pl))
    i = token[:, None]*8 + gl.full((2, BN), 0, gl.int32, pl)
    s0 = gl.gather(s, i+0, axis=0)
    s1 = gl.gather(s, i+1, axis=0)
    s2 = gl.gather(s, i+2, axis=0)
    s3 = gl.gather(s, i+3, axis=0)
    s4 = gl.gather(s, i+4, axis=0)
    s5 = gl.gather(s, i+5, axis=0)
    s6 = gl.gather(s, i+6, axis=0)
    s7 = gl.gather(s, i+7, axis=0)
    value = ((s0+s4)+(s2+s6))+((s1+s5)+(s3+s7))
    gl.store(OUT + batch*M*N + token[:, None]*N + no[None, :], value,
              (token[:, None] < M) & (no[None, :] < N))


def _validate(x, weight):
    if (x.dtype != torch.bfloat16 or x.ndim != 3 or x.shape[-1] != 32
            or x.shape[0] < 1 or x.shape[1] not in (1, 2) or not x.is_contiguous()
            or x.device.type != 'cuda' or weight.device != x.device
            or weight.dtype != torch.uint8 or weight.ndim != 3
            or weight.shape[0] != x.shape[0] or weight.shape[1] < 1
            or weight.shape[2] != 16 or not weight.is_contiguous()):
        raise ValueError('requires CUDA BF16[G,1|2,32], uint8[G,N,16] packed native codes')


def project_group(x, weight, *, bn=32, single_products=True):
    _validate(x, weight)
    out = torch.empty((*x.shape[:2], weight.shape[1]), device=x.device, dtype=torch.float32)
    project_group_out(x, weight, out, bn=bn, single_products=single_products)
    return out


def project_group_out(x, weight, out, *, bn=32, single_products=True):
    _validate(x, weight)
    if bn not in (32, 64):
        raise ValueError('initial group proof supports BN32/64')
    if (out.device != x.device or out.dtype != torch.float32 or not out.is_contiguous()
            or out.shape != (*x.shape[:2], weight.shape[1])):
        raise ValueError('requires contiguous FP32 group output')
    _group[(triton.cdiv(weight.shape[1], bn), x.shape[0])](x, weight, out,
        x.shape[1], weight.shape[1], BN=bn, SINGLE_PRODUCTS=single_products,
        num_warps=4, num_stages=1,
        enable_fp_fusion=False)
    return out


def _extract_helper(source, name):
    match = re.search(r'__device__[^\n]*\b'+re.escape(name)+r'\s*\(', source)
    if not match:
        raise RuntimeError('serving helper missing: '+name)
    body = source.index('{', match.start())
    depth = 1
    end = body+1
    while depth:
        depth += (source[end] == '{')-(source[end] == '}')
        end += 1
    return source[match.start():end]


_REFERENCE = None


def reference_library():
    global _REFERENCE
    if _REFERENCE is not None:
        return _REFERENCE
    serving = Path(__file__).with_name('fp4_moe_cuda.cu').read_text()
    helpers = '\n\n'.join(_extract_helper(serving, name) for name in
        ('decode_e2m1x2', 'load_activation32', 'group_partial'))
    code = '''#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
'''+helpers+'''
__global__ void raw_group(const __nv_bfloat16* x, const uint8_t* w, float* out,
                          int groups, int m, int n) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=groups*m*n) return;
    int row=idx%n, token=(idx/n)%m, group=idx/(m*n);
    float xv[32];
    load_activation32(x+(group*m+token)*32,xv);
    out[idx]=group_partial(*reinterpret_cast<const uint4*>(w+(group*n+row)*16),xv);
}
extern "C" int fp4_group_ref(cudaStream_t stream, const void* x, const void* w,
                           void* out, int groups, int m, int n) {
    raw_group<<<(groups*m*n+127)/128,128,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x),static_cast<const uint8_t*>(w),
        static_cast<float*>(out),groups,m,n);
    return static_cast<int>(cudaGetLastError());
}
'''
    nvcc = os.environ.get('NVCC', '/usr/local/cuda/bin/nvcc')
    digest = hashlib.sha256(code.encode()+subprocess.check_output([nvcc,'--version'])).hexdigest()[:24]
    folder = Path(os.environ.get('TRITON_CACHE_DIR','/tmp/dsv41-native'))/'fp4-group-proof'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{digest}.so'
    if not path.exists():
        cpp = folder/f'{digest}.cu'; cpp.write_text(code)
        subprocess.run([nvcc,'-std=c++17','-O3','-shared','-Xcompiler=-fPIC',
            '-gencode','arch=compute_121a,code=sm_121a',str(cpp),'-o',str(path)],check=True)
    lib = ctypes.CDLL(str(path))
    lib.fp4_group_ref.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*3
    lib.fp4_group_ref.restype = ctypes.c_int
    _REFERENCE = lib
    return lib


def reference_group(x, weight):
    _validate(x, weight)
    out = torch.empty((*x.shape[:2],weight.shape[1]),device=x.device,dtype=torch.float32)
    reference_group_out(x, weight, out)
    return out


def reference_group_out(x, weight, out):
    _validate(x, weight)
    code = reference_library().fp4_group_ref(
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),x.shape[0],x.shape[1],weight.shape[1])
    if code:
        raise RuntimeError(f'group reference launch CUDA error {code}')
    return out
