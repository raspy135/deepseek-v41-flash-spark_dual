"""A real image, end to end: preprocess -> ViT/aligner -> splice into an embedding stream.

What this catches is the layout. Every position of an image span carries image_token_id, and only
token_types tells IMAGE apart from the three delimiters -- so a splice that keys off the token id
would fill the delimiters with aligner rows and shift the whole grid.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from engine.vision import VisionTower

MD = os.environ.get("MODEL_DIR", "/models/DeepSeek-V4.1-Flash")
DEV = os.environ.get("DEV", "cpu")
sys.path.insert(0, os.path.join(MD, "inference"))
import image_processor as IP

class A: pass
for k, v in json.load(open(os.path.join(MD, "inference", "config.json"))).items():
    setattr(A, k, v)
A.vision_enabled = A.vision_n_layers > 0

tower = VisionTower(MD, device=DEV, dtype=torch.bfloat16)
print(f"tower loaded: {A.vision_n_layers} layers, delimiters {sorted(tower.delim)}")

img = os.path.join(MD, "assets", "dsv41_kv_cache.png")
patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = IP.load_image({"url": img}, A)
types = IP.image_token_types(n_llm_h, n_llm_w)
print(f"image: vit grid {n_vit_h}x{n_vit_w} ({patches.shape[0]} patches) -> llm grid {n_llm_h}x{n_llm_w}")
print(f"span: {types.numel()} tokens = {int((types==IP.IMAGE).sum())} IMAGE + "
      f"{int((types==IP.IMAGE_NEW_LINE).sum())} NEWLINE + 2 delimiters")
assert types.numel() == IP.num_image_tokens(n_llm_h, n_llm_w)

# an embedding stream with the image span starting at position 5
T, DIM = types.numel() + 12, A.dim
h = torch.zeros(T, DIM, dtype=torch.bfloat16)
img_in = IP.ImageInput(5, patches, n_vit_h, n_vit_w, types)
before = h.clone()
tower.splice(h, [img_in])

span = slice(5, 5 + types.numel())
touched = (h != before).any(dim=-1)
print(f"\n1. positions written: {int(touched.sum())} (span is {types.numel()}) "
      f"-> only inside the span: {bool(touched[span].all() and touched.sum()==types.numel())}")
assert bool(touched[span].all()) and int(touched.sum()) == types.numel()

s = h[span]
for name, t in (("IMAGE_START", IP.IMAGE_START), ("IMAGE_END", IP.IMAGE_END), ("IMAGE_NEW_LINE", IP.IMAGE_NEW_LINE)):
    rows = s[types == t]
    key = {"IMAGE_START":"image_start","IMAGE_END":"image_end","IMAGE_NEW_LINE":"image_newline"}[name]
    ok = bool(torch.equal(rows, tower.delim[key].expand_as(rows).to(rows.dtype)))
    print(f"2. {name:14} {rows.shape[0]:>3} slots carry the learned embedding: {ok}")
    assert ok

rows = s[types == IP.IMAGE]
uniq = torch.unique(rows.float(), dim=0).shape[0]
print(f"3. IMAGE slots: {rows.shape[0]} rows, {uniq} distinct, finite={bool(torch.isfinite(rows.float()).all())}")
assert rows.shape[0] == int((types==IP.IMAGE).sum()) and uniq > rows.shape[0] * 0.5
print(f"   (distinct rows >> 1 means real per-patch features, not a broadcast constant)")
print("\nVISION SPLICE OK")
