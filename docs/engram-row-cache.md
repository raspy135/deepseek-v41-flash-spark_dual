# Packed Engram row cache

The mmap/native Engram reader can keep immutable 264-byte rows in a bounded host
cache. Each slot contains 264 payload bytes and one 8-byte row ID. This avoids
keeping an entire filesystem page for every useful row, while retaining the
existing parallel NVMe reader for misses. No weights are quantized or changed.

Configure total capacity per Spark in the head node's `.env`:

```sh
DSV41_ENGRAM_CACHE_MB=2048
# Optional: give the peer more memory; otherwise it inherits the head budget.
DSV41_ENGRAM_CACHE_MB_PEER=4096
```

Values are MiB, from 0 (off) through 4096. The budget includes row tags and payload
arrays, is divided equally between the model's Engram tables, and does not include
temporary gather outputs or the OS page cache. Payload pages become resident as
rows are inserted. The peer setting is forwarded to both nodes; rank 1 selects it.
Both settings and the cache version participate in the boot configuration guard.
Different effective capacities are safe: cache hits do not change model arithmetic,
collective shapes or which ranks reach a collective. The default remains off
pending measurement. Images must contain this implementation on both nodes.

The cache is direct mapped. Hash collisions evict rows; they cannot change results.
Hit copies and insertions hold a per-table lock, while miss reads run outside it
so concurrent prefill read-ahead can continue. Cache outputs own their bytes and
remain valid after later insertions. Capacity stays fixed across requests; request
statistics contain cumulative cache hit/miss counters, not a reset cache. Subtract
successive snapshots for per-request counts. Prefix reuse and row reuse are separate.
Filled-row and eviction counters show whether increasing capacity is likely to help.

This is a local cache on each node, not a shared remote cache. Large prefill gathers
already split reads and exchange rows over RoCE through the existing Engram path.
Decode does not consult the other node's cache.

## Measurement

`tools/bench_engram_cache_tp.py` checks greedy and sampled token equality across
both ranks and cache modes. Prose, HTML and Python use off/cold/cold/off timing,
with an additional exact-repeat warm-cache run. Cold here means an empty packed
row cache; the OS page cache is retained. Off arms release packed-cache memory;
on arms touch the full capacity before timing to include its memory pressure.
The warmed repeated-prompt result is a favorable case, not a general throughput
claim. Host future wait is measured separately from total throughput; it does not
directly measure exposed GPU idle time.

`tools/bench_engram_remote.py` is a measurement-only synchronous remote-hit prototype:
send IDs, copy them to the peer CPU, gather cached rows, copy the reply to GPU and
broadcast it back. It includes those transfers and synchronization, but assumes
the peer is waiting and has the rows. It has no remote-miss protocol, cache directory
or arbitration with inference traffic. A microbenchmark win alone does not justify
turning on remote reads in serving. Random local reads report page-fault counts;
no global page-cache flush is used.

## 2026-10-03 measurements

Both nodes used 2048 MiB, confidence scheduling, resident FP4 TP2 and the same
frozen expert ranking. Up to 512 greedy output tokens, two measured runs per
off/cold arm (ABBA), plus one exact-repeat warm run:

| Workload | Cache off | Empty cache | Warm repeat | Empty-cache hit rate |
|---|---:|---:|---:|---:|
| HTML, 512 tokens | 46.81 tok/s | 45.61 tok/s | 47.00 tok/s | 34.8% |
| Python, 482 tokens | 50.07 tok/s | 50.10 tok/s | 50.99 tok/s | 43.0% |
| Prose, 512 tokens | 25.06 tok/s | 24.49 tok/s | 24.73 tok/s | 12.3% |

The empty-cache changes were -2.6%, +0.1%, and -2.3%; warm-repeat changes were
+0.4%, +1.8%, and -1.3%. **No consistent throughput improvement was established.**
Even 99.5-99.9% hits on exact repeats did not translate into a large decode gain.
Confidence scheduling responds to observed wall time, so the selected depths and
step counts varied slightly between runs. Output tokens remained identical across
cache modes and both ranks, including the sampled fixed-depth check. Host waits
and throughput have run-to-run variation; these small sample counts are not proof
of a reliable percentage gain. A varied, long-running working set was not tested.

The synchronous remote-hit prototype gave these head-node medians (10 samples,
after two warmups per size):

| Rows | Local first read | Local OS-cache repeat | Remote primed-cache lookup |
|---|---:|---:|---:|
| 24 | 1.202 ms | 0.569 ms | 0.531 ms |
| 144 | 2.247 ms | 0.483 ms | 0.509 ms |
| 384 | 3.642 ms | 0.457 ms | 0.629 ms |

Local first reads incurred median 48.5, 293 and 781 major page faults respectively,
consistent with the NVMe path. Remote replies matched the local bytes exactly.
This is evidence that a warm remote lookup can beat a local disk miss, not evidence
that adding a remote lookup to every decode step would help. Cache ownership,
miss handling, scheduling and contention during inference remain unimplemented.

Evidence: `results/engram-cache-20261003/`, including both rank reports and
`summary.json`. The user requested leaving the feature enabled at 2048 MiB on the
head and 4096 MiB on the peer after a separate asymmetric-capacity qualification.
The library and templates remain off by default; the local serving configuration
selects those capacities explicitly. No Engram model data is copied or downloaded.

The final 2048/4096 MiB gate passed five requests per rank, including greedy
off/cold/warm parity and sampled off/cold parity with fixed draft depth. Both
payload arrays were fully touched before the cache-on runs. Evidence:
`results/engram-cache-2-4-20261003/`. This short gate validates correctness and
capacity, not a fresh throughput claim for the asymmetric configuration.

The cache was first deployed in `deepseek-v41-flash-spark:engram-cache`, identical on both
nodes (`sha256:c08fdd6a7c43422e7f071921d59cea9b76319bcd830cdc26ac0c6044ddc22f41`).
It retains confidence scheduling and includes the experimental prefill graphs
with `DSV41_PREFILL_GRAPHS=0`. To disable row caching on both nodes, set both cache
budget variables to `0` and restart with `scripts/dual-down.sh` followed by
`scripts/dual-up.sh`. Clearing only the head budget leaves an explicit peer override
active. The prior `confidence-depth` image remains available for rollback.

Live greedy and sampled API checks passed after deployment; the health response
reported model `deepseek`, context 524288, primary cache 2048 MiB and confidence
scheduling enabled. Peer startup confirmed 4096 MiB. Both running containers use
the image ID above. API cache counters showed actual insertions and hits. Saved
verification: `results/engram-cache-2-4-20261003/service-enabled.json`.
The later `urgent-adapt` image retains this cache implementation and the 2048/4096
MiB serving configuration; see [urgent expert loading](urgent-expert-loading.md).
