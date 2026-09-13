"""One-load correctness and throughput gate for the batch-2 decode core.

Run on one Spark with the server stopped.  Identical lanes are deliberate: this is the maximum
expert-reuse case and therefore an upper-bound feasibility gate.  If batching cannot beat two
serial steps here, building the HTTP scheduler has no upside.
"""

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from engine.batch2 import Batch2FastDecoder, clone_lane  # noqa: E402
from engine.v41_engine import V41Engine  # noqa: E402


model_dir = os.environ.get("MODEL_DIR") or "/models/DeepSeek-V4.1-Flash"
trace = os.environ.get("TRACE_STATS", "results/trace-union/stats/coverage.json")
eng = V41Engine(
    model_dir, max_seq=int(os.environ.get("BATCH2_MAX_SEQ", "8192")), trace_stats=trace,
    spec=True, prune_keep=float(os.environ.get("PK", "0.25")),
    arena_gb=float(os.environ.get("AG", "82")), transient_slots=8, keep_free_gb=10,
    expert_format="fp4", world_size=1, rank=0,
)

prompt = "Explain why a B-tree remains balanced, then give compact insertion pseudocode."
ids = eng.tokenizer.encode(prompt, add_special_tokens=False)
for _ in eng.generate(ids, max_tokens=24, temperature=0.0):
    pass

lane0 = eng.fast
pos = lane0.c.len
tok = 128799
drafts, _ = lane0.draft(tok, pos - 1, 0.0)
block = torch.cat((torch.tensor([tok], device=eng.device), drafts.clone()))
hashes = lane0.m.hash_state(block[None], pos)[0]
rows = {layer: eng.tables[layer].rows(hashes[:, li, :])
        for li, layer in enumerate(eng.args.engram_layer_ids)}

# Serial oracle from the established path, then roll back to the exact starting boundary.
lane0.c.len = pos
ref_logits, _ = lane0.step(block, pos, rows)
ref_logits = ref_logits.clone()
lane0.c.rollback(pos)

lane1 = clone_lane(eng, lane0)
batch = Batch2FastDecoder(lane0, lane1)
(got0, _), (got1, _) = batch.step((block, block), (pos, pos), (rows, rows))
torch.cuda.synchronize()
for label, got in (("lane0", got0), ("lane1", got1)):
    rel = float((got - ref_logits).norm() / ref_logits.norm())
    agree = float((got.argmax(-1) == ref_logits.argmax(-1)).float().mean())
    print(f"{label}: logits rel={rel:.6g} argmax={agree:.3f}")
    if rel > 2e-4 or agree < 1.0:
        raise SystemExit(f"batch-2 equivalence failed for {label}")

def serial_pair():
    for lane in (lane0, lane1):
        lane.c.len = pos
        lane.step(block, pos, rows)
        lane.c.rollback(pos)

def batch_pair():
    for lane in (lane0, lane1):
        lane.c.len = pos
    batch.step((block, block), (pos, pos), (rows, rows))
    for lane in (lane0, lane1):
        lane.c.rollback(pos)

for fn in (serial_pair, batch_pair):
    fn()
torch.cuda.synchronize()
for label, fn in (("two serial", serial_pair), ("batch two", batch_pair)):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / 5
    print(f"{label}: {ms:.2f} ms/pair, {2000 / ms:.2f} pairs-equivalent steps/s")
