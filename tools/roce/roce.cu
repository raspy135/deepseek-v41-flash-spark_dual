// One-shot RoCE all-gather kernel for the two-Spark tensor parallelism (tools/roce/, DSV41_COMM_BACKEND=roce).
//
// Adapted from b12x's RoCEnante all-gather (b12x/comm/roce/_allgather_cute.py and _cute_intrinsics.py,
// https://github.com/local-inference-lab/b12x, Copyright 2026 Luke Alonso and the b12x contributors, Apache
// License 2.0; see LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md). Changes: rewritten from CuTe DSL into
// CUDA C++; only the dim-0 concatenation (NCCL's all-gather layout); any input/output alignment and any byte
// count (the send slot is padded to 16 bytes, only nbytes are copied out); a wall-clock timeout (%globaltimer)
// instead of a poll count; b12x PR #438's host-visible completion word and wrap-safe failure flag; the poison word
// read once per block (no divergent early exit around a barrier).
//
// Protocol (one op, sequence seq = epoch + 1, slot = seq & 1; see roce.cpp for the pinned region's layout):
//   1. every block stages its share of the input into the pinned send slot;
//   2. the last block to finish staging publishes the padded byte count and rings the doorbell (ctrl[0] = seq)
//      that the host proxy thread polls; the proxy RDMA-writes the slot to the peer, then a 4-byte seq flag on
//      the same reliable-connected queue pair (so the flag lands after the payload), striped over the HCAs;
//   3. one thread per (peer, HCA) spins on the peer's flag in this rank's pinned region with system-scope
//      acquire loads, until it reads seq or the timeout expires (then: error words + poison, fail-stop);
//   4. every block copies its share of every rank's shard into the output, rank order (the local shard from the
//      input, the peer's from the NIC-written receive slot with system-scope loads);
//   5. the last block to finish advances the device-resident epoch (so a replayed CUDA graph carries on with
//      the next sequence) and mirrors it into the host-visible completion word.
// Data are only moved, never combined: the gathered bytes equal NCCL's.

#include <cuda_runtime.h>
#include <stdint.h>

#include "roce_common.h"

namespace dsv41_roce {

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const void *p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys_u8(const uint8_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u8 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_relaxed_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_gpu(uint32_t *p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t atom_add_relaxed_gpu(uint32_t *p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Spin until *flag == want (system scope, acquire). false after timeout_ns of wall-clock time without a match.
__device__ __noinline__ bool wait_flag(const uint32_t *flag, uint32_t want, uint64_t timeout_ns) {
    const uint64_t t0 = globaltimer();
    uint32_t polls = 0;
    while (true) {
        if (ld_acquire_sys(flag) == want) {
            return true;
        }
        if ((++polls & 1023u) == 0u && globaltimer() - t0 > timeout_ns) {
            return ld_acquire_sys(flag) == want;   // one last look
        }
    }
}

// dst[i] <- src[i] for i in [0, n), a grid-stride loop; vector width chosen from the alignment of both pointers and n.
// ``sys_src``: the source is NIC-written pinned memory, read with system-scope loads.
template <bool SysSrc>
__device__ __forceinline__ void copy_bytes(uint8_t *dst, const uint8_t *src, uint64_t n, uint64_t index,
                                           uint64_t stride) {
    const uint64_t a = reinterpret_cast<uint64_t>(dst) | reinterpret_cast<uint64_t>(src) | n;
    if ((a & 15u) == 0u) {
        const uint64_t packs = n >> 4;
        for (uint64_t i = index; i < packs; i += stride) {
            uint4 v;
            if (SysSrc) {
                v = ld_relaxed_sys_v4(src + (i << 4));
            } else {
                v = reinterpret_cast<const uint4 *>(src)[i];
            }
            reinterpret_cast<uint4 *>(dst)[i] = v;
        }
    } else if ((a & 3u) == 0u) {
        const uint64_t words = n >> 2;
        for (uint64_t i = index; i < words; i += stride) {
            uint32_t v;
            if (SysSrc) {
                v = ld_relaxed_sys(reinterpret_cast<const uint32_t *>(src) + i);
            } else {
                v = reinterpret_cast<const uint32_t *>(src)[i];
            }
            reinterpret_cast<uint32_t *>(dst)[i] = v;
        }
    } else {
        for (uint64_t i = index; i < n; i += stride) {
            uint32_t v;
            if (SysSrc) {
                v = ld_relaxed_sys_u8(src + i);
            } else {
                v = src[i];
            }
            dst[i] = static_cast<uint8_t>(v);
        }
    }
}

__global__ void __launch_bounds__(1024) gather_kernel(GatherArgs a) {
    __shared__ uint32_t s_flag;
    const uint32_t tid = threadIdx.x;
    const uint32_t gdim = gridDim.x;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    const uint64_t stride = static_cast<uint64_t>(gdim) * blockDim.x;
    uint32_t *ctrl = reinterpret_cast<uint32_t *>(a.ctrl_base);

    // A recorded timeout poisons the runtime: later launches do nothing (the host raises on its next check).
    // One read per block, so no thread of the block leaves before a barrier the others wait at.
    if (tid == 0) {
        s_flag = ld_relaxed_gpu(a.poison);
    }
    __syncthreads();
    if (s_flag != 0u) {
        return;
    }
    const uint32_t seq = ld_relaxed_gpu(a.epoch) + 1u;
    const uint64_t slot = seq & 1u;
    uint8_t *send = reinterpret_cast<uint8_t *>(a.send_base + slot * a.slot_bytes);

    // 1. stage this rank's shard into the pinned send slot
    copy_bytes<false>(send, a.in, a.nbytes, index, stride);
    __syncthreads();

    // 2. the last block to finish staging rings the proxy's doorbell (one system fence per block after the
    //    barrier orders the block's staging stores before its arrival)
    if (tid == 0) {
        fence_sc_sys();
        const uint32_t prior = atom_add_relaxed_gpu(a.stage_ctr, 1u);
        if (((prior + 1u) & (gdim - 1u)) == 0u) {
            const uint32_t padded = static_cast<uint32_t>((a.nbytes + 15u) & ~static_cast<uint64_t>(15u));
            st_relaxed_sys(ctrl + CTRL_NBYTES, padded);
            st_relaxed_sys(ctrl + CTRL_SLOT_NBYTES + slot, padded);
            fence_sc_sys();
            st_relaxed_sys(ctrl + CTRL_SEQ, seq);
        }
    }

    // 3. wait for every peer's per-HCA flag of this sequence
    if (tid < static_cast<uint32_t>(a.world * a.n_hca)) {
        const int peer = static_cast<int>(tid) / a.n_hca;
        const int hca = static_cast<int>(tid) - peer * a.n_hca;
        if (peer != a.rank) {
            const uint32_t *flag = reinterpret_cast<const uint32_t *>(
                a.flag_base + ((static_cast<uint64_t>(peer) * a.slots + slot) * a.n_hca + hca) * a.flag_stride);
            if (!wait_flag(flag, seq, a.timeout_ns)) {
                st_relaxed_sys(ctrl + CTRL_ERR_PEER, static_cast<uint32_t>(peer));
                st_relaxed_sys(ctrl + CTRL_ERR_HCA, static_cast<uint32_t>(hca));
                st_relaxed_sys(ctrl + CTRL_ERR_SEQ, seq);
                st_relaxed_sys(ctrl + CTRL_FAILED, 1u);
                fence_sc_sys();
                st_release_gpu(a.poison, 1u);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        s_flag = ld_relaxed_gpu(a.poison);
    }
    __syncthreads();

    // 4. every rank's shard, in rank order (a timed-out wait leaves the peer slot unreliable: nothing is copied)
    if (s_flag == 0u) {
        for (int src = 0; src < a.world; src++) {
            uint8_t *dst = a.out + static_cast<uint64_t>(src) * a.nbytes;
            if (src == a.rank) {
                copy_bytes<false>(dst, a.in, a.nbytes, index, stride);
            } else {
                const uint8_t *peer_slot = reinterpret_cast<const uint8_t *>(
                    a.recv_base + (static_cast<uint64_t>(src) * a.slots + slot) * a.slot_bytes);
                copy_bytes<true>(dst, peer_slot, a.nbytes, index, stride);
            }
        }
    }

    // 5. the last block to finish publishes the next epoch (not after a failure: later launches stay no-ops)
    fence_sc_gpu();
    __syncthreads();
    if (tid == 0) {
        const uint32_t prior = atom_add_relaxed_gpu(a.tail_ctr, 1u);
        if (((prior + 1u) & (gdim - 1u)) == 0u) {
            fence_sc_gpu();
            if (ld_relaxed_sys(ctrl + CTRL_FAILED) == 0u) {
                st_release_gpu(a.epoch, seq);
                st_relaxed_sys(ctrl + CTRL_COMPLETED, seq);
            }
        }
    }
}

cudaError_t launch_gather(const GatherArgs &args, int grid, int threads, cudaStream_t stream) {
    gather_kernel<<<grid, threads, 0, stream>>>(args);
    return cudaGetLastError();
}

}  // namespace dsv41_roce
