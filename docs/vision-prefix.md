# Vision prefix caching

Previously every request containing an image forced `prefix_start=0`, even when
the image was unchanged conversation history. That made every following turn pay
for a full prefill. Comparing token IDs alone is unsafe: different images use the
same placeholder IDs.

Each request now hashes the actual preprocessed patch tensors, their dtype/shape,
image token types, and ViT grid. The fingerprint is attached to the image's token
span. Both ranks agree on these identities before attempting reuse. RAM snapshots
and disk-prefix hashes include the image identity within that boundary.

- Same image inputs and text: reuse the longest matching prefix.
- Changed image, even at the same URL: reuse only boundaries before that image.
- New image appended to history: the earlier matching prefix remains eligible.
- Image removed: do not reuse snapshots that depended on it.
- Mid-image boundary: never restore there; an image must be prefetched whole.

Preprocessing and hashing still happen per request. A cached image span skips the
vision tower and model prefill. Post-response answer preparation still skips vision
requests; the input prefix is cached normally, so the next turn prefills the previous
answer and newly appended text rather than the whole conversation.

The disk format/compatibility version is now 2. Previous cache bundles are not used
by this build and are subject to the existing LRU budget; no manual deletion is
required. As before, cached model state can reveal private prompt/image information.

Validation scripts: `engine/test_prefix_media.py`, `engine/test_prefix_disk.py`, and
`tools/test_vision_prefix_cuda.py`. The GPU gate uses tiny synthetic red/blue images,
compares cached vs fresh greedy output, checks RAM and disk hits, and checks that
identical placeholder IDs cannot reuse a different image's state.

## Measured validation

TP2 native MXFP4, 768K allocation, packed KV, 92 GB arena, frozen keep=0.63;
eight output tokens maximum, synthetic 64x64 images preprocessed by the checkpoint.
All cached/fresh comparisons returned identical greedy output tokens.

| Case | Cached / prompt tokens | Prefill seconds |
| --- | --- | --- |
| Initial red image | 0 / 1463 | 4.632 |
| Same prompt, RAM | 1463 / 1463 | 0.200 |
| Same prompt, disk after unrelated request | 1463 / 1463 | 0.254 |
| Changed text suffix, red image | 1031 / 1466 | 2.511 |
| Fresh computation of that suffix | 0 / 1466 | 2.647 |
| Changed pixels to blue, same token IDs | 7 / 1466 | 1.971 |
| Fresh blue computation | 0 / 1466 | 1.821 |

The red requests answered “Red”; both blue requests answered “Blue”. The seven-token
reuse stops exactly before the changed image. This is a small functional probe,
not a general throughput benchmark: partial reuse helped little here, and the tiny
changed-image hit was slower than fresh. The first cold GPU attempt stopped before
inference because the test driver omitted the server's vision-enabled flag; the
corrected run passed. Logs: `results/vision-prefix-gate-2/rank{0,1}.log`.

CPU validation: 37 unit tests, four existing prefix-snapshot checks, and 16 mock HTTP
tests passed. `tools/test_vision_prefix_http.py` checks a real two-turn conversation
with unchanged image history, rather than repeating an identical request.

Live HTTP continuation passed with the user's current profile unchanged (512K
context, 90 GB arena, keep=0.61, adaptation enabled, concurrency=1). First-turn
prefill: 6.337 s for 1,462 tokens. The follow-up reused 1,462/1,477 tokens (99.0%),
prefilled only 15 new tokens, and took 1.061 s prefill / 2.363 s HTTP wall time.
This verifies image history no longer forces a full miss in the serving path.
Local log: `/tmp/dsv41-vision-prefix-http.log`.
