"""Replay verify-width policies against logged DSpark confidence (tools/bench_spec_conf_tp.py).

Each logged step verified all five drafts, so for every step the number of leading accepts `a` is
known, and any smaller width w would have emitted min(a, w) + 1 tokens. Policies, per workload:

  fixed w      verify w drafts every step
  best fixed   the better fixed width for this workload: an upper bound on per-request adaptive
               depth (DSV41_BLOCK_DYNAMIC), which also pays for its first ~60 tokens
  confidence   DSpark's rule (tech report 2.4.3): survival s_k = prod_{j<=k} sigmoid(c_j),
               E[tokens | w] = 1 + sum_{k<=w} s_k, pick the w maximizing E / step_ms(w)
  conf+bias    the same with a logit offset tuned on THIS data -- in-sample, so optimistic
  oracle       knows `a`; the cheapest width that still collects it

Step costs are constants per width (--ms), from measured TP2 step times. Real step cost also
varies with how many distinct experts the drafts route to, so these are rates of a model, not
of the engine; a policy that wins here still has to win in a full-engine run.

    python tools/analyze_spec_conf.py results/<dir>/spec-conf-rank0.json [--widths 1,3,5]
"""
import argparse
import json
import math


def sig(x):
    return 1 / (1 + math.exp(-x))


def auc(scores, labels):
    """Rank AUC; None when one class is missing."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def rate(steps, choose, ms):
    tok = t = 0.0
    for a, conf in steps:
        w = choose(a, conf)
        tok += min(a, w) + 1
        t += ms[w]
    return tok / t * 1000


def conf_rule(widths, ms, bias=0.0):
    def choose(a, conf):
        best, best_r, s, e = widths[0], -1.0, 1.0, 1.0
        for k, c in enumerate(conf, 1):
            s *= sig(c + bias)
            e += s
            if k in widths and e / ms[k] > best_r:
                best, best_r = k, e / ms[k]
        return best
    return choose


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('report')
    ap.add_argument('--widths', default='3,5', help='verify depths a policy may choose (1 needs new graphs)')
    ap.add_argument('--ms', default='1:88,3:106,5:124', help='step ms per depth (docs/decode-dynamic-depth.md)')
    args = ap.parse_args()
    widths = sorted(int(x) for x in args.widths.split(','))
    ms = {int(k): float(v) for k, v in (kv.split(':') for kv in args.ms.split(','))}
    assert all(w in ms for w in widths), 'every width needs a step cost'
    rep = json.load(open(args.report))
    assert rep.get('spec_conf'), 'this report was collected without DSV41_SPEC_CONF=1'
    top = max(widths)
    print(f'widths {widths}, step ms {ms}\n')
    hdr = f"{'workload':10} {'steps':>5} " + ' '.join(f'{"fixed " + str(w):>8}' for w in widths)
    print(hdr + f" {'best fix':>8} {'conf':>7} {'conf+b':>7} {'oracle':>7}  conf vs best fixed   AUC per draft")
    for run in rep['runs']:
        log = [(d, a, c) for d, a, c in (run['steps_log'] or [])]
        steps = [(a, c) for d, a, c in log if d >= top]
        censored = len(log) - len(steps)
        fixed = {w: rate(steps, lambda a, c, w=w: w, ms) for w in widths}
        best = max(fixed.values())
        conf = rate(steps, conf_rule(widths, ms), ms)
        conf_b = max(rate(steps, conf_rule(widths, ms, b / 4), ms) for b in range(-16, 17))
        oracle = rate(steps, lambda a, c: next((w for w in widths if w >= a), top), ms)
        # conditional acceptance of draft k given drafts 1..k-1 accepted: the quantity the head predicts
        aucs = []
        for k in range(len(steps[0][1]) if steps else 0):
            sub = [(c[k], a > k) for a, c in steps if a >= k]
            aucs.append(auc([s for s, _ in sub], [l for _, l in sub]))
        auc_s = ' '.join('  -  ' if x is None else f'{x:.2f}' for x in aucs)
        print(f"{run['workload']:10} {len(steps):5d} " + ' '.join(f'{fixed[w]:8.1f}' for w in widths)
              + f' {best:8.1f} {conf:7.1f} {conf_b:7.1f} {oracle:7.1f}  {100 * (conf / best - 1):+17.1f}%   {auc_s}'
              + (f'  ({censored} censored steps skipped)' if censored else ''))


if __name__ == '__main__':
    main()
