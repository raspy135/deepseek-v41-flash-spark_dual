"""One nesting arm: build the engine, greedily generate the four nesting depths, grade.

The arm is chosen by the environment, so run ONE arm per process:

    ARM=base      DSV41_MOE_FALLBACK=0 DSV41_ACT_QUANT=0  python tools/nesting_arm.py
    ARM=fallback  DSV41_MOE_FALLBACK=1 SPEC=0             python tools/nesting_arm.py
    ARM=actquant                     DSV41_ACT_QUANT=1    python tools/nesting_arm.py

Why one process per arm: `Model.__init__` monkeypatches `v41_ref.act_qdq_fp8` to the identity
when `act_quant=False`, and monkeypatches are process-global. A second engine built in the same
process would silently inherit the first arm's choice. Fresh process, fresh global.

The prompt and the grader are copied from llm_benchmark/quality_quant2.py so the number is
comparable to the served runs in git/llm_benchmark.

MARGIN=1 teacher-forces the canonical spaced depth-8 answer using single-token decode;
MARGIN_OUT saves token IDs, top five logits, and target-minus-best-other margins as JSON.
MARGIN_GENERATE=1 also checks actual greedy output at depth 8. A different token may merely
choose different whitespace/tokenization; margins are diagnostics, not a correctness grader.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from engine.v41_engine import V41Engine, log  # noqa: E402

NEST_DEPTHS = [4, 6, 8, 10]
NEST_PROMPT = ('Output one JSON object nested exactly {d} levels deep and nothing else. '
               'Each level has exactly one key "n" whose value is the next level down. '
               'The innermost "n" is the integer {leaf}. '
               'So depth 2 would be: {{"n": {{"n": {leaf}}}}}')


def grade_nest(text, d, leaf):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return 0.0, {"parsed": False}
    try:
        o = json.loads(t[i:j + 1])
    except Exception:  # noqa: BLE001
        return 0.0, {"parsed": False}
    depth, cur = 0, o
    while isinstance(cur, dict) and "n" in cur:
        depth += 1
        cur = cur["n"]
    ok_leaf = cur == leaf
    score = max(0.0, 1 - abs(depth - d) / d) * (1.0 if ok_leaf else 0.5)
    return score, {"depth": depth, "want": d, "leaf_ok": ok_leaf}


def main() -> int:
    md = os.environ.get("MODEL_DIR") or os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
    arm = os.environ.get("ARM", "base")
    kw = dict(
        max_seq=int(os.environ.get("MAX_SEQ", "8192")),
        prune_keep=float(os.environ["PRUNE_KEEP"]) if os.environ.get("PRUNE_KEEP") else None,
        keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "20")),
        expert_format=os.environ.get("EXPERT_FORMAT", "fp4"),
        spec=os.environ.get("SPEC", "1") == "1",
    )
    log(f"ARM {arm}: DSV41_MOE_FALLBACK={os.environ.get('DSV41_MOE_FALLBACK', '0')} "
        f"DSV41_ACT_QUANT={os.environ.get('DSV41_ACT_QUANT', '0')} kwargs={kw}")
    eng = V41Engine(md, **kw)

    sys.path.insert(0, os.path.join(md, "encoding"))
    from encoding import encode_messages  # noqa: E402

    if os.environ.get("MARGIN", "0") == "1":
        # Replay the identical correct prefix in each fresh process, using actual single-token
        # decode after the prompt. Batched teacher forcing can select different matmul kernels.
        d = 8
        pr = encode_messages([{"role": "user", "content": NEST_PROMPT.format(d=d, leaf=48)}],
                             thinking_mode="chat")
        pr = pr[0] if isinstance(pr, tuple) else pr
        ids = eng.tokenizer.encode(pr, add_special_tokens=False)
        answer = '{"n": ' * d + '48' + '}' * d
        targets = eng.tokenizer.encode(answer, add_special_tokens=False)
        rows = []
        with torch.inference_mode():
            eng._reset()
            eng.model.begin_prompt()
            start = time.perf_counter()
            logits, _ = eng.model.forward(torch.tensor(ids, device=eng.device), 0,
                                          prefill=True, need_logits=True)
            for j, target in enumerate(targets):
                scores = logits[-1].float()
                values, indices = scores.topk(5)
                winner = int(indices[0])
                competitor = int(indices[1] if winner == target else indices[0])
                row = dict(answer_pos=j, target=target, target_text=eng.tokenizer.decode([target]),
                           winner=winner, winner_text=eng.tokenizer.decode([winner]),
                           margin=float(scores[target] - scores[competitor]),
                           top5=[(int(i), eng.tokenizer.decode([int(i)]), float(v))
                                 for i, v in zip(indices, values)])
                rows.append(row)
                log(json.dumps(row))
                if j + 1 < len(targets):
                    logits, _ = eng.model.forward(torch.tensor([target], device=eng.device),
                                                  len(ids) + j, prefill=False)
            log(f"margin arm {arm}: {time.perf_counter() - start:.1f}s")
        if os.environ.get("MARGIN_OUT"):
            with open(os.environ["MARGIN_OUT"], "w") as f:
                json.dump(dict(arm=arm, prompt_ids=ids, answer_ids=targets, rows=rows), f, indent=2)
        if os.environ.get("MARGIN_GENERATE", "0") == "1":
            out = []
            for burst in eng.generate(ids, max_tokens=64, temperature=0.0):
                out += burst
            text = eng.tokenizer.decode(out)
            log(f"depth-8 greedy: {grade_nest(text, d, 48)} {text!r} ids={out}")
        return 0

    scores = []
    for d in NEST_DEPTHS:
        leaf = 40 + d
        prompt = NEST_PROMPT.format(d=d, leaf=leaf)
        pr = encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")
        pr = pr[0] if isinstance(pr, tuple) else pr
        ids = eng.tokenizer.encode(pr, add_special_tokens=False)
        out = []
        for burst in eng.generate(ids, max_tokens=int(os.environ.get("MAX_TOKENS", "200")), temperature=0.0):
            out += burst
        text = eng.tokenizer.decode(out)
        s, chk = grade_nest(text, d, leaf)
        scores.append(s)
        log(f"  depth={d:2d} score={s:.3f} {chk}")
        log(f"    {text[:160]!r}")
    log(f"ARM {arm} nesting macro {sum(scores) / len(scores):.3f}  {scores}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
