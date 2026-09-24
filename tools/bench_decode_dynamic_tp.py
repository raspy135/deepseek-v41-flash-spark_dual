"""DSV41_BLOCK_DYNAMIC full-engine check and timing. Disposable gate only; never serving.

Run with DSV41_BLOCK_DYNAMIC=3,5 (and DSV41_BLOCK=5, or unset, since .env pins 3):

    GATE_ENV="DSV41_BLOCK=5 DSV41_BLOCK_DYNAMIC=3,5" GATE_IMAGE=<id> \\
    GATE_LOG_DIR=results/<dir> GATE_SOURCE_ROOT=<snapshot> \\
    bash tools/run_two_node_gate.sh bench_decode_dynamic_tp.py --out /app/results/<dir>

Per workload, in one process:
  alternate  switch depth EVERY step (worst case for buffer/graph switching and the rank-0
             depth broadcast); also the warm-up that captures both widths
  pin3/pin5  the policy pinned to one depth
  adaptive   the real policy (DSV41_BLOCK_DYNAMIC_TOKENS, default every 60 tokens)
Greedy outputs must be token-identical across all four -- verification accepts exactly the
target model's argmax, so the depth schedule may change speed, never text. The sampled story
is timed only (its random stream depends on the schedule). Peak GPU memory is reported: the
second width's graph set is the price of this feature.
"""
import argparse
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt
from bench_decode_block_tp import WORKLOADS
import engine.fastdecode as FD


class Alternate:
    """decide() hook: flips depth every step."""
    def __init__(self, pol):
        self.pol, self.next = pol, pol.lo

    def __call__(self):
        d, self.next = self.next, (self.pol.hi if self.next == self.pol.lo else self.pol.lo)
        self.pol.depth = d
        return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=512)
    args = ap.parse_args()
    assert FD.DYNAMIC_DEPTHS, 'set DSV41_BLOCK_DYNAMIC (e.g. 3,5) for this bench'
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    pol = e.depth_policy
    assert pol is not None
    real_decide = pol.decide
    report = dict(config=e.config(), depths=list(FD.DYNAMIC_DEPTHS), runs=[], mismatches=[])

    def emit(kind, item):
        print('DECODE_DYN_' + kind + ' ' + json.dumps(dict(rank=e.ep.rank, **item)), flush=True)

    def generate(ids, temperature, mode):
        pol.pinned = {'pin_lo': pol.lo, 'pin_hi': pol.hi}.get(mode)
        pol.decide = Alternate(pol) if mode == 'alternate' else real_decide
        out = []
        for burst in e.generate(ids, max_tokens=args.max_tokens, temperature=temperature, seed=42,
                                stop_token_ids={eos}):
            out.extend(burst)
        pol.pinned, pol.decide = None, real_decide
        return out

    for name, prompt, temperature in WORKLOADS:
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                         enc, tok, False, 75, e)
        modes = ('alternate', 'pin_lo', 'pin_hi', 'adaptive') if temperature == 0 else ('pin_lo', 'adaptive')
        reference = None
        for mode in modes:
            out = generate(ids, temperature, mode)
            st = dict(e.last_stats)
            exact = None
            if temperature == 0:
                reference = out if reference is None else reference
                exact = all(e.ep.gather_objects(out == reference))
                if not exact:
                    report['mismatches'].append(f'{name}/{mode}')
            item = dict(workload=name, mode=mode, temperature=temperature, exact_vs_alternate=exact,
                        tokens=st.get('completion_tokens'), decode_tok_s=st.get('decode_tok_s'),
                        steps=st.get('steps'), accept_len_mean=st.get('accept_len_mean'),
                        spec_depth=st.get('spec_depth'))
            report['runs'].append(item)
            emit('RUN', item)
    report['peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
    report['graph_keys'] = sorted(str(k) for k in e.fast.graphs)
    path = f'{args.out}/dynamic-rank{e.ep.rank}.json'
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(report, f, indent=2)
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    emit('SUMMARY', dict(peak_allocated_gb=report['peak_allocated_gb'], mismatches=report['mismatches'],
                         graph_keys=report['graph_keys']))
    emit('PASS' if not report['mismatches'] else 'FAIL', {})
    os._exit(0 if not report['mismatches'] else 1)


if __name__ == '__main__':
    main()
