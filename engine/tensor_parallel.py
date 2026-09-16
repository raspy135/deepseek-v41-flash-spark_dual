"""Dense TP primitives. Keep the checkpoint's rounding AFTER each complete linear.

Attention shards heads/groups, not queries or shared compressed KV. Output projections
gather inputs and disjoint results by default. The legacy reduction path remains for A/B
tests. The vocabulary head gathers disjoint logits, without sums.
"""
import os
import torch
import torch.distributed as dist


def shard(weight, dim, rank, world):
    if isinstance(weight, torch.Tensor):
        if weight.shape[dim] % world:
            raise ValueError('unaligned tensor shard')
        return weight.chunk(world, dim=dim)[rank].clone().contiguous()
    return weight.shard(dim, rank, world)


class RowParallelWeight:
    def __init__(self, local):
        self.local = local
        self.shape = local.shape

    def tp_linear(self, x):
        import v41_ref as R
        from fp8_linear import FP8Weight, fp8_linear
        x = x.to(torch.bfloat16)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if isinstance(self.local, FP8Weight) and flat.size(0) <= 16:
            partial = fp8_linear(flat, self.local, out_dtype=torch.float32)
        else:
            weight = self.local if isinstance(self.local, torch.Tensor) else self.local.dequant()
            # Pad small BF16 tensor GEMMs just as R.mm does, preserving decode block invariance.
            rows = flat.size(0)
            if 0 < rows < R.MM_TILE:
                flat = torch.cat((flat, flat.new_zeros(R.MM_TILE - rows, flat.size(1))))
            partial = torch.mm(flat, weight.t(), out_dtype=torch.float32)[:rows].contiguous()
        dist.all_reduce(partial)
        return partial.to(torch.bfloat16).view(*shape[:-1], self.shape[0])


class OutputParallelWeight:
    """Gather BF16 inputs, compute complete dot products for disjoint output rows.

    Keeps half the weight bytes without reducing separately accumulated K partitions.
    Both collectives concatenate values; neither introduces a floating-point sum.
    """
    def __init__(self, local, world=2):
        if world != 2 or local.shape[1] % world:
            raise ValueError('output-parallel linear requires two aligned input shards')
        self.local, self.world = local, world
        self.shape = (local.shape[0] * world, local.shape[1] // world)

    def tp_linear(self, x):
        import v41_ref as R
        shape = x.shape
        flat = x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous()
        rows, width = flat.shape
        gathered = torch.empty((self.world * rows, width), dtype=flat.dtype, device=flat.device)
        dist.all_gather_into_tensor(gathered, flat)
        full = gathered.view(self.world, rows, width).transpose(0, 1).reshape(rows, self.world * width)
        local = R.mm(full, self.local).contiguous()
        output = torch.empty((self.world * rows, local.shape[-1]), dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(output, local)
        return output.view(self.world, rows, local.shape[-1]).transpose(0, 1).reshape(
            *shape[:-1], self.shape[0])


def shard_attention(weight, args, rank, world):
    from fp8_linear import FP8GroupedWeight
    if world != 2 or not 0 <= rank < world:
        raise ValueError('attention TP currently requires two ranks')
    if args.n_heads % world or args.o_groups % world:
        raise ValueError('attention heads/groups must divide TP world')
    weight.tp_heads, weight.tp_groups = args.n_heads // world, args.o_groups // world
    weight.wq_b = shard(weight.wq_b, 0, rank, world)
    weight.attn_sink = shard(weight.attn_sink, 0, rank, world)
    wo = weight.wo_a
    if isinstance(wo, FP8GroupedWeight):
        rows = wo.G * wo.R // world
        lo, hi = rank * rows, (rank + 1) * rows
        weight.wo_a = FP8GroupedWeight(wo.w[lo:hi].clone(), wo.s[lo//32:hi//32].clone(),
                                      wo.G // world, wo.R)
    elif isinstance(wo, torch.Tensor):
        weight.wo_a = shard(wo, 0, rank, world)
    else:
        raise ValueError('attention TP supports native FP8 or BF16 wo_a only')
    layout = os.environ.get('DSV41_TP_LINEAR_LAYOUT', 'output')
    if layout == 'output':
        weight.wo_b = OutputParallelWeight(shard(weight.wo_b, 0, rank, world), world)
    elif layout == 'intermediate':
        weight.wo_b = RowParallelWeight(shard(weight.wo_b, 1, rank, world))
    else:
        raise ValueError('TP linear layout must be intermediate or output')


class FeatureParallelEmbedding:
    """Keep disjoint feature columns; concatenate lookups without rounding or sums.

    All ranks must index the same IDs, including draft and CUDA-graph paths.
    The caller slices on CPU before uploading to avoid a full GPU allocation.
    """
    def __init__(self, local, world):
        if world != 2 or local.ndim != 2:
            raise ValueError('embedding TP requires two feature shards')
        self.local, self.world = local, world
        self.shape = (local.shape[0], local.shape[1] * world)
        self.device, self.dtype = local.device, local.dtype

    def __getitem__(self, ids):
        local = self.local[ids].contiguous()
        flat = local.reshape(-1, local.shape[-1])
        rows, width = flat.shape
        gathered = torch.empty((self.world * rows, width), dtype=flat.dtype, device=flat.device)
        dist.all_gather_into_tensor(gathered, flat)
        return gathered.view(self.world, rows, width).transpose(0, 1).reshape(
            *local.shape[:-1], self.shape[1])


class VocabParallelHead:
    def __init__(self, local, world):
        self.local, self.world = local, world
        self.shape = (local.shape[0] * world, local.shape[1])

    def tp_logits(self, x):
        import v41_ref as R
        local = R.head_logits(x, self.local).contiguous()
        flat = local.reshape(-1, local.shape[-1])
        gathered = torch.empty((self.world * flat.shape[0], flat.shape[1]),
                               dtype=flat.dtype, device=flat.device)
        dist.all_gather_into_tensor(gathered, flat)
        return gathered.view(self.world, flat.shape[0], flat.shape[1]).transpose(0, 1).reshape(
            *x.shape[:-1], self.shape[0])
