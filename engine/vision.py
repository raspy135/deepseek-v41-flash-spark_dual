"""Vision tower at serve time: the checkpoint's own ViT + Aligner, fed from its own weights.

The tower is 32 dense bf16 layers with 2D RoPE and full bidirectional attention over one image
-- no MoE, no KV cache, no quantization -- so there is nothing here the engine's kernels do
better, and re-implementing it would only add a surface for porting bugs. This imports
`inference/vision.py` from the checkpoint and loads the weights into it, exactly the way
engine/engram.py imports the checkpoint's NgramHashState.

Vision costs ~0.9 GB resident (tower 0.767 + aligner 0.137), about 2.5 experts per layer at
fp4, and it only runs during prefill: the aligner's output replaces the embeddings at the
image-token positions before layer 0, and nothing downstream of that knows an image was there
-- except the MoE router, which has a SEPARATE bias for image tokens (`bias_vl`), and the
n-gram hasher, which must not let an n-gram span an image.
"""

from __future__ import annotations

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
    # RMSNorm weights are declared fp32 in the reference; everything else follows `dtype`
    sd = vit.state_dict()
    vit.load_state_dict({k: v.to(torch.float32 if sd[k].dtype == torch.float32 else dtype)
                         for k, v in vs.items()})
    sda = aligner.state_dict()
    aligner.load_state_dict({k: v.to(torch.float32 if sda[k].dtype == torch.float32 else dtype)
                             for k, v in als.items()})
    return vit.to(device).eval(), aligner.to(device).eval(), cfg


@torch.inference_mode()
def embed_image(vit, aligner, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
    """patches [n_patches, 3, P, P] -> [n_image_tokens, dim], ready to splice over the
    image-token positions in the embedding stream."""
    return aligner(vit(patches, n_h, n_w), n_h, n_w)
