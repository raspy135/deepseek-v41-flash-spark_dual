"""CPU-only real-checkpoint read pipeline screening; no inference/GPU copies.

The production baseline consumes leased pinned bytes without a clone. The staged
arm uses read_expert's owning CPU copies, including their extra memcpy cost. An
optional sleep models an installation interval; it is a simulation, not an H2D
measurement or a model-throughput claim. No global page cache is flushed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.experts import EXPERT_BYTES, ExpertStore
from engine.swap_staging import SwapStager


class _Arena:
    slots = 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--install-ms", default="0,0.25,1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 128 or args.rounds < 1:
        parser.error("count must be 1..128 and rounds positive")
    index = json.loads(Path(args.model_dir, "model.safetensors.index.json").read_text())
    store = ExpertStore(args.model_dir, index, _Arena(), 40, io_threads=max(4, args.workers))
    stager = SwapStager(store.read_expert, bytes_per_expert=EXPERT_BYTES,
                        max_bytes=EXPERT_BYTES * args.window, workers=args.workers)
    # Spread reads over layers and expert IDs; all outgoing keys are symbolic, as
    # this runner has no arena/mask state and never invokes apply_swaps.
    plan = [(i % 40, i, 128 + i, 1.0 / (i + 1)) for i in range(args.count)]
    rows = []
    try:
        # Verify a staged record against a fresh independent baseline read, then
        # warm the exact working set. Cold/JIT/OS effects are not mixed with arms.
        import torch
        reference = store.read_expert(plan[0][0], plan[0][2])
        test = stager.start(plan[:1], 0)
        test.wait(plan[:1], 0)
        with test.take((plan[0][0], plan[0][2]), plan[:1], 0) as record:
            if not all(torch.equal(a, b) for a, b in zip(reference, record.views)):
                raise AssertionError("staged checkpoint bytes differ")
        test.cancel()
        del reference
        for layer, _out, incoming, _gain in plan:
            store._read_leased(layer, incoming, None, lambda views: None)
        for delay_ms in [float(value) for value in args.install_ms.split(",")]:
            if delay_ms < 0:
                parser.error("install-ms must be nonnegative")
            arms = {"serial_leased": [], "staged_owning": []}
            for repeat in range(args.rounds):
                for arm in (tuple(arms) if repeat % 2 == 0 else tuple(reversed(arms))):
                    def consume(views):
                        if delay_ms:
                            time.sleep(delay_ms / 1000)
                    start = time.perf_counter()
                    if arm == "serial_leased":
                        for layer, _out, incoming, _gain in plan:
                            store._read_leased(layer, incoming, None, consume)
                    else:
                        staged = stager.start(plan, repeat)
                        try:
                            staged.wait(plan, repeat)
                            for layer, _out, incoming, _gain in plan:
                                with staged.take((layer, incoming), plan, repeat) as record:
                                    consume(record.views)
                        finally:
                            staged.cancel()
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    arms[arm].append(elapsed_ms)
                    print(f"install_simulated={delay_ms:g}ms {arm} round={repeat} "
                          f"total={elapsed_ms:.3f}ms", flush=True)
            medians = {arm: statistics.median(samples) for arm, samples in arms.items()}
            rows.append({"simulated_install_ms_per_expert": delay_ms,
                         "samples_ms": arms, "median_ms": medians,
                         "change_pct": 100 * (medians["staged_owning"] /
                                               medians["serial_leased"] - 1)})
    finally:
        stager.close()
        store.pool.shutdown(wait=True)
        store.read_pool.shutdown(wait=True)
    result = {"scope": "CPU real-checkpoint O_DIRECT reads; simulated installation only; "
                       "no model, GPU copies, collectives or throughput measurement",
              "args": vars(args), "byte_exact": True,
              "torch_version": str(torch.__version__),
              "cpu_intraop_threads": torch.get_num_threads(),
              "retained_payload_budget_bytes": stager.max_bytes,
              "peak_retained_payload_bytes": stager.peak_reserved_bytes,
              "reserved_after_close": stager.reserved_bytes,
              "rows": rows}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
