"""Vision tower at serve time: the checkpoint's own ViT + Aligner, fed from its own weights.

The tower is 32 dense bf16 layers with 2D RoPE and full bidirectional attention over one image
-- no MoE, no KV cache, no quantization -- so there is nothing here the engine's kernels do
better, and re-implementing it would only add a surface for porting bugs. This imports
`inference/vision.py` from the checkpoint and loads the weights into it, exactly the way
engine/engram.py imports the checkpoint's NgramHashState.

STATUS (2026-09-13): loading is verified (263/263 parameters bitwise), the ViT+aligner takes
0.65-0.77 s for a 1.2 MP image's 8547 patches, and the splice lands 962 aligner rows on exactly
the 962 IMAGE slots with the three delimiters carrying their learned embeddings. The apparent
post-splice GPU loop was an asymmetric EP failure: the configured prune set exceeded rank 1's
LRU capacity, the image router touched more missing experts in one layer than its 16 transient
slots, and rank 1 raised while rank 0 remained blocked in an NCCL combine. Pruned mode now
refuses to start unless every routable expert fits on its owning rank.

Vision costs ~0.9 GB resident (tower 0.767 + aligner 0.137), about 2.5 experts per layer at
fp4, and it only runs during prefill: the aligner's output replaces the embeddings at the
image-token positions before layer 0, and nothing downstream of that knows an image was there
-- except the MoE router, which has a SEPARATE bias for image tokens (`bias_vl`), and the
n-gram hasher, which must not let an n-gram span an image.
"""

from __future__ import annotations

import functools
import json
import os
import sys

import torch


def _vision_args(model_dir: str):
    cfg = json.load(open(os.path.join(model_dir, "inference", "config.json")))

    class A:
        pass

    for k, v in cfg.items():
        setattr(A, k, v)
    # vision.py reads these directly off the args object
    A.dim = cfg["dim"]
    return A, cfg


def load_vision(model_dir: str, device: str = "cuda", dtype=torch.bfloat16):
    """Returns (vit, aligner, cfg) with the checkpoint's weights loaded, in eval mode.

    Raises if the checkpoint has no vision tower (vision_n_layers == 0), which is how a
    text-only checkpoint declares it.
    """
    from safetensors import safe_open

    args, cfg = _vision_args(model_dir)
    if not getattr(args, "vision_n_layers", 0):
        raise RuntimeError("this checkpoint has no vision tower (vision_n_layers == 0)")
    sys.path.insert(0, os.path.join(model_dir, "inference"))
    import vision as V  # noqa: E402

    # The reference builds its 2D-RoPE tables with no device argument, so they are always CPU --
    # fine in the reference, which runs the tower on CPU tensors it has just created, but here the
    # ViT is on the GPU and apply_rotary then multiplies cuda activations by cpu tables:
    # "Expected all tensors to be on the same device". It raises rather than falling back, and
    # under EP2 a raise mid-forward leaves the peer mid-generate and the pair out of step, so the
    # symptom is a hung request rather than an error. Cache them per device instead.
    if not getattr(V, "_dsv41_device_rope", False):
        _orig_cos_sin = V.get_vision_cos_sin

        @functools.lru_cache(16)
        def _cos_sin_on(n_h, n_w, dim, theta, _dev=None):
            cos, sin = _orig_cos_sin(n_h, n_w, dim, theta)
            return cos.to(_dev), sin.to(_dev)

        def get_vision_cos_sin(n_h, n_w, dim, theta):
            return _cos_sin_on(n_h, n_w, dim, theta, _dev=V._dsv41_rope_device)

        V.get_vision_cos_sin = get_vision_cos_sin
        V._dsv41_device_rope = True
    V._dsv41_rope_device = torch.device(device)

    # The reference hands scaled_dot_product_attention 3D tensors [heads, n, head_dim]. SDPA wants
    # 4D [batch, heads, n, head_dim] and rejects EVERY fused kernel otherwise ("All fused kernels
    # requires query, key and value to be 4 dimensional"), falling back to the math path -- which
    # materialises the full n x n score matrix: 11 GB peak for a 1.2MP image's 8547 patches, and
    # 9.1 s. That is survivable on an idle box and fatal inside the server, where the expert arena
    # leaves only a few GB and the allocation thrashes instead of failing. One unsqueeze lets the
    # fused kernels run.
    if not getattr(V, "_dsv41_sdpa_4d", False):
        import torch.nn.functional as _F

        def _attn_forward(self, x, cos, sin):
            n = x.size(0)
            q, k, v = (t.view(n, self.n_heads, self.head_dim)
                       for t in self.wqkv(x).chunk(3, dim=-1))
            q = V.apply_rotary(q, cos, sin)
            k = V.apply_rotary(k, cos, sin)
            o = _F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
                                                k.transpose(0, 1).unsqueeze(0),
                                                v.transpose(0, 1).unsqueeze(0)).squeeze(0)
            return self.wo(o.transpose(0, 1).reshape(n, -1))

        V.Attention.forward = _attn_forward
        V._dsv41_sdpa_4d = True

    vit, aligner = V.ViT(args), V.Aligner(args)
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    want = {}
    for name in list(vit.state_dict()) + ["aligner." + n for n in aligner.state_dict()]:
        key = name if name.startswith("aligner.") else "vision." + name
        want.setdefault(index[key], []).append((key, name))
    vs, als = {}, {}
    for shard, pairs in want.items():
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for key, name in pairs:
                t = f.get_tensor(key)
                if name.startswith("aligner."):
                    als[name[len("aligner."):]] = t
                else:
                    vs[name] = t
    # Follow the CHECKPOINT's dtype, not the freshly-built module's. Every vision tensor ships
    # BF16 (including the RMSNorm weights the reference declares fp32), and a module built from
    # nn.Linear defaults is fp32 everywhere -- so testing the module's dtype keeps the whole tower
    # in fp32 and the first matmul dies on bf16 patches against fp32 weights.
    vit.load_state_dict({k: v.to(dtype) for k, v in vs.items()})
    aligner.load_state_dict({k: v.to(dtype) for k, v in als.items()})
    # load_state_dict copies INTO the existing parameters and keeps their dtype, so the module
    # itself has to be cast -- loading bf16 tensors into an fp32 module leaves it fp32.
    return (vit.to(device=device, dtype=dtype).eval(),
            aligner.to(device=device, dtype=dtype).eval(), cfg)


@torch.inference_mode()
def embed_image(vit, aligner, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
    """patches [n_patches, 3, P, P] -> [n_IMAGE_slots, dim].

    These rows fill only the IMAGE positions of the span. The delimiters carry learned embeddings
    instead (see splice_images)."""
    return aligner(vit(patches, n_h, n_w), n_h, n_w)


class VisionTower:
    """ViT + Aligner + the three learned span delimiters, and the splice that uses them."""

    # image_processor's token types; TEXT is -1 so `types >= 0` selects a whole image span.
    IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

    def __init__(self, model_dir: str, device: str = "cuda", dtype=torch.bfloat16):
        from safetensors import safe_open
        self.vit, self.aligner, self.cfg = load_vision(model_dir, device, dtype)
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.delim = {}
        for name in ("image_start", "image_end", "image_newline"):
            with safe_open(os.path.join(model_dir, index[name]), framework="pt") as f:
                self.delim[name] = f.get_tensor(name).to(device).to(dtype)
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def splice(self, h: torch.Tensor, images) -> torch.Tensor:
        """Overwrite each image's span in `h` [T, dim] with its features, in place.

        The IMAGE slots take the aligner rows in row-major order; the three delimiters take their
        learned embeddings. Every position in the span carries `image_token_id` in the token ids,
        so the types tensor is the ONLY thing that tells them apart -- replacing every occurrence
        of the image token id would put aligner rows in the delimiter slots and silently shift the
        rest of the grid by one.
        """
        import time as _t
        for img in images or ():
            _t0 = _t.perf_counter()
            types = img.types.to(h.device)
            span = h[img.start:img.start + types.numel()]
            assert span.shape[0] == types.numel(), (
                f"image span at {img.start} runs past the chunk ({span.shape[0]} of "
                f"{types.numel()} positions); image spans must lie inside one prefill chunk")
            span[types == self.IMAGE_START] = self.delim["image_start"].to(h.dtype)
            span[types == self.IMAGE_END] = self.delim["image_end"].to(h.dtype)
            span[types == self.IMAGE_NEW_LINE] = self.delim["image_newline"].to(h.dtype)
            rows = embed_image(self.vit, self.aligner,
                               img.patches.to(h.device).to(self.dtype), img.n_vit_h, img.n_vit_w)
            n_slots = int((types == self.IMAGE).sum())
            assert rows.shape[0] == n_slots, (
                f"aligner produced {rows.shape[0]} rows for {n_slots} IMAGE slots")
            span[types == self.IMAGE] = rows.to(h.dtype)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[vision] span {types.numel()} tok, {img.patches.shape[0]} patches, "
                  f"{rows.shape[0]} rows in {_t.perf_counter() - _t0:.2f}s", flush=True)
        return h
