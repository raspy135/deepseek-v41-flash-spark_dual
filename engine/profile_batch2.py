"""One-load correctness and throughput gate for the batch-2 decode core.

Run on the TP pair with the server stopped. Identical lanes are deliberate: this is the maximum
expert-reuse case and therefore an upper-bound feasibility gate.  If batching cannot beat two
serial steps here, building the HTTP scheduler has no upside.
"""

import os
import sys
import time
import traceback

def fail_fast(kind, value, tb):
    traceback.print_exception(kind, value, tb)
    sys.stderr.flush()
    # CUDA/process-group destructors can hang after a failed assertion, leaving
    # the peer waiting until the outer timeout. This is a disposable test process.
    os._exit(1)

sys.excepthook = fail_fast

for flag in ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL',
             'DSV41_PREFIX_CACHE', 'DSV41_PREFIX_DISK'):
    os.environ[flag] = '0'

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from engine.batch2 import Batch2FastDecoder, clone_lane  # noqa: E402
from engine.v41_engine import V41Engine  # noqa: E402
import engine.v41_engine as V
V.save_prune_db = lambda *args, **kwargs: None


model_dir = os.environ.get("MODEL_DIR") or "/models/DeepSeek-V4.1-Flash"
trace = os.environ.get("TRACE_STATS", "results/trace-union/stats/coverage.json")
eng = V41Engine(
    model_dir, max_seq=int(os.environ.get("BATCH2_MAX_SEQ", "8192")), trace_stats=trace,
    spec=True, prune_keep=float(os.environ.get("PK", "0.60")),
    arena_gb=float(os.environ.get("AG", "88")), transient_slots=16, keep_free_gb=6,
    expert_format="fp4",
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

if '--requests' in sys.argv:
    from engine.serving import DecodeRuntime
    from engine.decode_events import VerifyStep
    from tools.nesting_arm import NEST_PROMPT, grade_nest
    sys.path.insert(0, os.path.join(eng.model_dir, 'encoding'))
    from encoding import encode_messages

    def encode(text):
        encoded = encode_messages([{'role': 'user', 'content': text}], thinking_mode='chat')
        return eng.tokenizer.encode(encoded[0] if isinstance(encoded, tuple) else encoded,
                                    add_special_tokens=False)

    prompts = [encode(NEST_PROMPT.format(d=8, leaf=48)),
               encode('日本語だけで、二分探索の仕組みを短く説明してください。')]
    kwargs = dict(max_tokens=96, temperature=0.0, seed=17)
    references = []
    for prompt_ids in prompts:
        out = []
        for burst in eng.generate(prompt_ids, **kwargs):
            out.extend(burst)
        references.append(out)
    runtime = DecodeRuntime(eng)

    def requests(indices, temperature=0.0, cancel=None):
        out = {i: [] for i in indices}
        for i in indices:
            runtime.execute({'op': 'start', 'lane': i, 'prompt_ids': prompts[i],
                             'kwargs': {**kwargs, 'temperature': temperature}})
        active = set(indices)
        while active:
            for i in sorted(active):
                event = runtime.events[i]
                if event is None:
                    active.remove(i)
                elif not isinstance(event, VerifyStep):
                    out[i].extend(event)
                    runtime.execute({'op': 'advance', 'lane': i})
            if cancel in active and isinstance(runtime.events[cancel], VerifyStep):
                runtime.execute({'op': 'close', 'lane': cancel})
                active.remove(cancel)
                cancel = None
            ready = sorted(i for i in active if isinstance(runtime.events[i], VerifyStep))
            if ready:
                runtime.execute({'op': 'verify', 'lanes': ready})
                for i in ready:
                    runtime.execute({'op': 'advance', 'lane': i})
        return out

    for i in (0, 1):
        solo = requests([i])[i]
        print(f'solo scheduler lane={i}: exact={solo == references[i]}', flush=True)
        if solo != references[i]:
            print('solo:', eng.tokenizer.decode(solo), 'reference:', eng.tokenizer.decode(references[i]), flush=True)
        assert solo == references[i], f'cloned lane mismatch lane={i}'
    got = requests([0, 1])
    for i in (0, 1):
        print(f'request lane={i}: serial={len(references[i])} paired={len(got[i])} '
              f'exact={references[i] == got[i]}', flush=True)
        if references[i] != got[i]:
            first = next((j for j, (a, b) in enumerate(zip(references[i], got[i])) if a != b),
                         min(len(references[i]), len(got[i])))
            print(f'first mismatch at {first}: expected={references[i][first:first+8]} '
                  f'got={got[i][first:first+8]}', flush=True)
        assert references[i] == got[i], f'independent request mismatch lane={i}'
    text = eng.tokenizer.decode(got[0], skip_special_tokens=True)
    baseline_grade = grade_nest(eng.tokenizer.decode(references[0], skip_special_tokens=True), 8, 48)
    paired_grade = grade_nest(text, 8, 48)
    print('nesting serial/paired:', baseline_grade, paired_grade, repr(text), flush=True)
    # This gate freezes the current pruned set. A failing baseline is recorded,
    # not attributed to batching, and must not prevent the isolation checks below.
    assert paired_grade == baseline_grade
    sampled = {i: requests([i], temperature=0.7)[i] for i in (0, 1)}
    got = requests([0, 1], temperature=0.7)
    assert got == sampled, 'request-local sampling RNG changed under interleaving'
    got = requests([0, 1], cancel=0)
    assert got[1] == references[1], 'cancelling lane zero affected lane one'
    got = requests([0, 1])
    assert all(got[i] == references[i] for i in (0, 1)), 'lane reuse changed output'
    print('independent prompts, seeded sampling, cancellation and lane reuse: PASS', flush=True)
assert all(eng.ep.gather_objects(True))
sys.stdout.flush()
os._exit(0)
