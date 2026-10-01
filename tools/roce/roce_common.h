// Shared by roce.cpp (host proxy, bindings) and roce.cu (kernel): the control record and the kernel arguments.
// patches/0230; the protocol is adapted from b12x's RoCEnante (Apache-2.0, see roce.cu).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace dsv41_roce {

// 32-bit words of the control record (one FLAG_STRIDE block at the end of the pinned region). The kernel writes all
// of them but CTRL_SEQ's reader (the proxy); the host reads them without touching the device.
enum : int {
    CTRL_SEQ = 0,          // doorbell: the newest sequence staged (proxy polls it)
    CTRL_NBYTES = 1,       // padded bytes of the newest op
    CTRL_ERR_SEQ = 2,      // sequence whose wait timed out
    CTRL_ERR_PEER = 3,     // the peer it waited for
    CTRL_SLOT_NBYTES = 4,  // [4], [5]: padded bytes per slot (the proxy's catch-up after a missed doorbell)
    CTRL_ERR_HCA = 6,      // the HCA whose flag never came
    CTRL_FAILED = 7,       // 1 once any wait timed out (wrap-safe, unlike a sequence number)
    CTRL_COMPLETED = 8,    // the last sequence completed on this rank (doorbell ahead of it: an op is in flight)
    CTRL_WORDS = 9,
};

struct GatherArgs {
    const uint8_t *in;
    uint8_t *out;
    uint64_t nbytes;       // bytes of one rank's shard
    uint64_t recv_base;    // device addresses of the pinned region's parts
    uint64_t flag_base;
    uint64_t send_base;
    uint64_t ctrl_base;
    uint64_t slot_bytes;
    uint32_t *epoch;       // device counters: epoch, this grid size's stage / tail arrivals, poison
    uint32_t *stage_ctr;
    uint32_t *tail_ctr;
    uint32_t *poison;
    uint64_t timeout_ns;
    int world;
    int rank;
    int n_hca;
    int flag_stride;
    int slots;
};

cudaError_t launch_gather(const GatherArgs &args, int grid, int threads, cudaStream_t stream);

}  // namespace dsv41_roce
