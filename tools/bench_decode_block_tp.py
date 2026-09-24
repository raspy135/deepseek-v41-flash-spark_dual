"""Decode throughput and draft acceptance across workloads, for comparing DSV41_BLOCK values.

DSV41_BLOCK sizes module-level buffers and graphs, so one process runs one block size; compare
separate gate runs that differ only in it (run_two_node_gate.sh GATE_ENV="DSV41_BLOCK=5").
Same frozen-ranking / prefix-off setup as the timeline bench. Four workloads, because
acceptance -- and so the best block -- depends on how predictable the text is:

  html     the timeline bench's coffee-shop page (structured, highly predictable)
  python   a small, specified program (code)
  explain  an expository answer (prose)
  story    open-ended fiction (least predictable); also run sampled at temperature 0.7

One warm-up pass over every workload (graph capture, caches) is discarded, then --passes
measured passes. Greedy outputs must repeat exactly across passes.

    GATE_ENV="DSV41_BLOCK=5" GATE_IMAGE=<id> GATE_LOG_DIR=results/<dir>/block5 \\
    bash tools/run_two_node_gate.sh bench_decode_block_tp.py --out /app/results/<dir>/block5
"""
import argparse
import json
import os
import statistics
import sys
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt, HTML_PROMPT
from engine.fastdecode import T_DRAFT

WORKLOADS = (
    ('html', HTML_PROMPT, 0.0),
    ('python', 'Write a Python module implementing an LRU cache class with get, put and a '
               'max-size eviction policy, plus five unittest test cases. Output only the code.', 0.0),
    ('explain', 'Explain in plain prose, for a curious non-specialist, why the sky is blue and '
                'why sunsets are red. Use about five paragraphs and no lists or headings.', 0.0),
    ('story', 'Write an original short story, about 400 words, about a lighthouse keeper who '
              'finds an unusual object washed ashore. Literary tone, no title.', 0.0),
    ('story_t07', 'Write an original short story, about 400 words, about a lighthouse keeper who '
                  'finds an unusual object washed ashore. Literary tone, no title.', 0.7),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--passes', type=int, default=2)
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    report = dict(config=e.config(), draft_tokens=T_DRAFT, max_tokens=args.max_tokens, runs=[])

    def emit(kind, item):
        print('DECODE_BLOCK_' + kind + ' ' + json.dumps(dict(rank=e.ep.rank, draft_tokens=T_DRAFT, **item)),
              flush=True)

    ids = {}
    for name, prompt, _ in WORKLOADS:
        _, ids[name], _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                               enc, tok, False, 75, e)
    outputs = {}
    for p in range(args.passes + 1):
        for name, _, temperature in WORKLOADS:
            out = []
            for burst in e.generate(ids[name], max_tokens=args.max_tokens, temperature=temperature,
                                    seed=42 + p, stop_token_ids={eos}):
                out.extend(burst)
            st = dict(e.last_stats)
            same = None
            if temperature == 0:
                same = outputs.setdefault(name, out) == out
            item = dict(workload=name, temperature=temperature, measured_pass=p if p else None,
                        completion_tokens=st.get('completion_tokens'), decode_s=st.get('decode_s'),
                        decode_tok_s=st.get('decode_tok_s'), steps=st.get('steps'),
                        accept_len_mean=st.get('accept_len_mean'), repeat_exact=same)
            if p:
                report['runs'].append(item)
            emit('RUN' if p else 'WARMUP', item)
    summary = {}
    for name, _, _ in WORKLOADS:
        rows = [r for r in report['runs'] if r['workload'] == name]
        summary[name] = dict(decode_tok_s=statistics.median(r['decode_tok_s'] for r in rows),
                             accept_len_mean=statistics.median(r['accept_len_mean'] for r in rows),
                             ms_per_step=statistics.median(1000 * r['decode_s'] / r['steps'] for r in rows),
                             tokens=statistics.median(r['completion_tokens'] for r in rows))
    report['summary'] = summary
    emit('SUMMARY', dict(summary=summary))
    path = f'{args.out}/block{T_DRAFT}-rank{e.ep.rank}.json'
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(report, f, indent=2)
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    emit('PASS', {})
    os._exit(0)


if __name__ == '__main__':
    main()
