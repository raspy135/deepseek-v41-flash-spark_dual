"""Summarize interval unions, not summed CUDA durations or inferred bandwidth."""
import argparse
import collections
import json


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def length(intervals):
    return sum(b-a for a, b in merged(intervals))


def intersection(a, b):
    a, b = merged(a), merged(b)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def analyze(trace, trim_edges=False):
    events = trace['traceEvents']
    windows = [(e['ts'], e['ts']+e['dur']) for e in events
               if e.get('cat') == 'decode_host' and e.get('name') == 'decode/window']
    if trim_edges:
        controls = sorted(e['ts'] for e in events if e.get('cat') == 'decode_host'
                          and e.get('name') == 'decode/control')
        if len(controls) < 4:
            raise ValueError('need at least four decode controls to trim window edges')
        windows = [(controls[1], controls[-1])]
    gpu = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    if not gpu:
        raise ValueError('no GPU events; this is not a GPU timing measurement')
    if not windows:
        windows = [(min(e['ts'] for e in gpu), max(e['ts']+e['dur'] for e in gpu))]
    groups = collections.defaultdict(list)
    kernels = collections.defaultdict(lambda: [0, 0.0])
    for event in gpu:
        name = event['name']
        interval = intersection([(event['ts'], event['ts']+event['dur'])], windows)
        if not interval:
            continue
        if event['cat'] != 'kernel':
            family = 'copy_or_memset'
        elif 'nccl' in name.lower():
            family = 'nccl'
        elif '_moe_up' in name or '_moe_down' in name:
            family = 'expert_projection'
        elif '_fp8_' in name:
            family = 'fp8_projection'
        elif 'gemmSN_TN' in name or 'sgemm' in name.lower():
            family = 'fp32_gemm'
        else:
            family = 'other_kernel'
        groups[family].extend(interval)
        kernels[name][0] += 1
        kernels[name][1] += length(interval)/1000
    busy = merged([p for intervals in groups.values() for p in intervals])
    idle = []
    for lo, hi in merged(windows):
        cursor = lo
        for start, end in intersection(busy, [(lo, hi)]):
            if cursor < start:
                idle.append((cursor, start))
            cursor = end
        if cursor < hi:
            idle.append((cursor, hi))
    host = collections.defaultdict(list)
    for event in events:
        if event.get('cat') == 'decode_host' and event['name'] != 'decode/window':
            host[event['name']].extend(intersection([(event['ts'], event['ts']+event['dur'])], windows))
    return dict(metadata=trace.get('decode_metadata', {}), edges_trimmed=trim_edges,
                window_ms=length(windows)/1000,
                gpu_busy_union_ms=length(busy)/1000, gpu_idle_ms=length(idle)/1000,
                gpu_family_union_ms={k:length(v)/1000 for k,v in groups.items()},
                host_spans={k:dict(union_ms=length(v)/1000,
                                  overlaps_gpu_idle_ms=length(intersection(v, idle))/1000)
                            for k,v in host.items()},
                largest_idle_gaps_ms=sorted(((b-a)/1000 for a,b in idle), reverse=True)[:10],
                top_kernels=[dict(name=k, calls=v[0], summed_ms=v[1]) for k,v in
                             sorted(kernels.items(), key=lambda kv:-kv[1][1])[:15]],
                caveats=['Family times overlap; do not sum them as exclusive costs.',
                         'Host/idle overlap indicates correlation, not the cause of a stall.',
                         'No DRAM counters: this trace does not measure achieved bandwidth.',
                         'Profiler overhead and any cold graph capture affect timings.',
                         'Profiled request stats include profiler start/stop overhead; use unprofiled stats for throughput.',
                         'Ranks use independent host clocks; do not align them by timestamp.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--trim-edges', action='store_true', help='Exclude first/last iterations using host control boundaries')
    args = parser.parse_args()
    with open(args.trace) as f:
        print(json.dumps(analyze(json.load(f), args.trim_edges), indent=2))
