"""Print one decode layer's kernel program from bench_decode_kernels_tp.py's sequence dump.

    python3 tools/decode_layer_program.py results/<dir>/kernels-seq-rank0.tsv [--layer 6]

A layer is what lies between two consecutive routed-expert down kernels (native CUDA `down` or the
Triton `_moe_down_kernel`), so layer i here starts at the tail of backbone layer i-1's MoE (routed
combine, shared expert, hc_post) and ends with layer i's routed experts. Also prints the kernel
count per layer across the window, to spot layers with extra work (indexer, compressor, engram).
"""
import argparse
import collections
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tsv')
    ap.add_argument('--layer', type=int, default=6)
    args = ap.parse_args()
    rows = [line.rstrip('\n').split('\t') for line in open(args.tsv)]
    rows = [(float(s), float(us), name) for s, us, name in rows]
    down = [i for i, (_, _, n) in enumerate(rows) if re.search(r'moe_down', n)]
    print(f'{len(rows)} kernels, {len(down)} routed-expert down kernels')
    counts = collections.Counter(down[i + 1] - down[i] for i in range(len(down) - 1))
    print('kernels between consecutive down kernels (count: occurrences):', dict(sorted(counts.items())))
    lo, hi = down[args.layer] + 1, down[args.layer + 1] + 1
    print(f'\nlayer program #{args.layer}: {hi - lo} kernels, {sum(r[1] for r in rows[lo:hi]):.0f} us')
    for _, us, name in rows[lo:hi]:
        print(f'{us:8.1f}  {re.sub(r"<.*", "", name.replace("void ", ""))[:110]}')


if __name__ == '__main__':
    main()
