# Phase 4.4A: Target-only Fixed-shape Graph Experiment

## Scope and Setup

This is an isolated diagnostic prototype, not a production graph integration.
Frozen production code, historical tests, Scheduler and Phase 4.1 transaction
semantics remain unchanged. No Git commit. No draft batching, Adaptive K,
attention rewrite, custom GEMM/NCCL or kernel fusion.

- Qwen3-14B BF16, TP=2, dedicated EAGLE-3 persistent draft, K=3.
- Original Scheduler, eager engine; only explicitly prepared target shapes replay.
- Target revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Draft revision: `3d13517724e81cb409ddf1d4650772ec52f1e18e`.
- Torch 2.8.0+cu128, CUDA 12.8, NCCL 2.27.3, FlashAttention 2.8.3,
  Triton 3.4.0, Transformers 4.57.1.
- Two RTX 4090 24GB. Actual current topology is **SYS across NUMA nodes**,
  not the NODE topology reported in the project's initial environment.
- max_model_len=2048, token budget=2048, max_num_seqs=4, memory utilization=.70.

## Capture Boundary and Ownership

Only unchanged target tensor computation is captured: embedding, Transformer
layers, model TP all-reduces, LM-head gather and static output copies. Proposal,
acceptance, reservation, commit/rollback, request cursors, finite/status checks
and cleanup remain outside capture. A borrowed static feature output is cloned
before it becomes request-owned state. No graph retains a request dictionary.

`GraphKey` includes M, batch size, q lengths, exact max_q/max_k, rectangular
block-table width, block size, config identity, dtype/device class, model mode,
TP size and feature layers. M alone is not a key. Weights/KV/static input-output
addresses remain fixed locally. Dynamic IDs, positions, cumulative lengths,
slots and block-table values are copied into existing buffers. Both ranks agree
key, ordered layout, generation, mode and key availability before execution.
Any rank missing a key forces both to eager; unsupported q/ragged shapes also
fall back. This is a two-entry bounded prototype, not a general graph cache.

Each engine owns one capture stream. Graph memory pools are independent. Graph
destruction precedes buffer scrubbing and process-group teardown. See the
[architecture document](../../docs/EAGLE3_PHASE4_4A.md) and
[reproduction commands](commands.md).

## Correctness Gate

The standalone two-rank all_reduce/gather probe passed 20 mixed eager/graph
iterations on each rank. The full target gate then passed:

- M4/M8/M16, uniform context lengths 255/256/257/768/1024.
- Heterogeneous contexts [255,256] and [255,256,257,768].
- 17 paired generation cases, 41 paired requests, 24 output tokens/request.
- 35 audited verifications per rank, each replayed three times from restored
  identical pre-verification pages: 210 rank-level replay comparisons total.
- Bitwise equality of local logits on both ranks, full rank0 logits/target IDs,
  target features and every touched KV page, including committed prefix and
  unused/rejected rows. No numerical tolerance waiver.
- Identical proposal IDs, target verification IDs, accepted lengths, committed
  IDs, cursors, ordered rank status and final outputs across full generation.
- Independent feature ownership and stable static storage addresses.
- Graph -> synchronized eager veto -> graph with identical verification results.
- Intentional host acceptance exception after verification: normal frozen abort
  cleanup, followed by graph release and normal engine exit.

After release, all 17 cases have exactly the same allocated memory per rank:
19,574,825,984 / 16,778,348,544 bytes. Waiting/running/used blocks/transactions,
host requests/drafts and both ranks' target states are zero. Graph counts are
zero after release. No numerical disagreement or NCCL/CUDA error occurred in
the final gate. 69 frozen source/artifact file hashes were unchanged.

This validates **same-shape eager versus graph**, not universal q=1 versus q>1
BF16 serial token parity. The previously accepted numerical contract and its
historical strict serial-parity failures remain unchanged.

## Retained Development Failures

The first M4 generation passed, but a subsequent prototype control command
raced the frozen single-slot RPC mailbox, causing a Gloo timeout. Buffer scrub
also lacked inference-mode scope during exit. Both were fixed exclusively in
the new prototype: acknowledged control commands and inference-mode release.
This was not a model collective capture failure; no correctness check or NCCL
setting was removed to obtain successful capture.

A subsequent full numerical gate exposed 8,519,680 retained bytes per newly
created capture stream. A standalone GEMM control reproduced the exact increment
and showed a constant allocation when reusing one stream. This is consistent
with documented per-handle/stream [cuBLAS workspace retention](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cublas-workspaces).
The prototype now owns one shared capture stream, avoiding unbounded stream-key
growth without clearing backend caches. The complete gate was rerun. An earlier
performance process was terminated during startup before timing samples.
All earlier logs/artifacts are retained separately, not filtered or overwritten.

## Timing Method

Timing and correctness use separate fresh processes, with source-hash gating.
Two timing processes each execute all six cells (M4/M8/M16 x context256/768).
Per cell: four warmup calls/mode, five rotated repeats, ten calls/mode/repeat.
Modes are frozen eager, coordinated eager, and graph. Coordinated eager performs
the same rank-plan consensus before invoking the unchanged eager target.

Primary latency is host monotonic time around the complete verification RPC,
including packing, graph consensus/copies where applicable, and all frozen
finite/status checks. Rollback, proposal, capture and warmup are outside this
interval. Every timed call must reproduce identical target IDs/rank status.
Deferred CUDA events measure the full endpoint's stream span, **not pure kernel
compute time**. No samples are deleted. This is fixed-input target replay, not
an online serving throughput benchmark.

Separate minimal CPU/CUDA traces with two targets/mode/cell inspect CPU launch
API counts/time, GPU graph nodes, device-active interval unions/gaps and matched
collective start skew. They are not the speedup denominator. Raw endpoint entry
timestamps use the host-wide monotonic clock; GPU skew uses Kineto calibration.
NCCL residency is not interpreted as pure transport cost.

## Additional Storage-reuse Gate

A separate process reused one captured graph per M across three unrelated
requests/batches, changing token contents, request IDs, physical blocks and,
for M8/M16, per-sequence context lengths while preserving the same safe key.
Only three graphs/rank were captured for nine batches. All 54 additional
rank-level replay comparisons were bitwise identical to eager, with normal
cleanup and exit. Thus the evidence is not restricted to replaying unchanged
capture-time buffer contents.

## Measured Target Latency

All repeats are complete. Each table row pools **100 calls/mode** from two fresh
processes, each with five repeats of ten calls. Total primary samples: 1,800.
Numbers are mean full verification RPC milliseconds, not profiler durations.

| M | Context | Frozen eager | Coordinated eager | Graph | Saved ms | Target speedup | Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 54.747 | 55.801 | 19.610 | 35.137 | 2.792x | 64.18% |
| 4 | 768 | 64.523 | 64.795 | 21.101 | 43.422 | 3.058x | 67.30% |
| 8 | 256 | 68.425 | 66.691 | 21.547 | 46.879 | 3.176x | 68.51% |
| 8 | 768 | 68.135 | 65.823 | 22.132 | 46.003 | 3.079x | 67.52% |
| 16 | 256 | 66.955 | 67.315 | 23.506 | 43.449 | 2.848x | 64.89% |
| 16 | 768 | 67.541 | 65.747 | 24.191 | 43.350 | 2.792x | 64.18% |

All 60 paired repeat means improved: speedup range **2.422x to 3.564x**.
Process B cell speedups were 2.800-3.451x; process C, 2.559-2.942x. Variation
between processes is retained, not averaged away as a claim of universal gain.
Equally pooling the two contexts gives M4 **2.930x**, M8 **3.126x**, M16 **2.820x**.
M8/context256 has the largest pooled benefit. This is not a serving-weighted mix.

Coordinated eager is close to frozen eager and does not reproduce graph's gain.
The additional Gloo alignment alone is therefore insufficient to explain it.
Graph timings include that consensus and static-buffer update overhead.

| M | Context | CUDA-event span eager -> graph ms | RPC ms/row eager -> graph | RPC P95 eager -> graph ms | RPC P99 eager -> graph ms |
|---|---:|---:|---:|---:|---:|
| 4 | 256 | 54.669 -> 19.498 | 13.687 -> 4.902 | 69.484 -> 20.456 | 73.351 -> 21.344 |
| 4 | 768 | 64.417 -> 20.952 | 16.131 -> 5.275 | 89.312 -> 22.582 | 99.785 -> 24.078 |
| 8 | 256 | 68.309 -> 21.389 | 8.553 -> 2.693 | 87.600 -> 23.276 | 93.005 -> 23.681 |
| 8 | 768 | 68.009 -> 21.971 | 8.517 -> 2.767 | 92.402 -> 23.530 | 102.050 -> 24.021 |
| 16 | 256 | 66.833 -> 23.335 | 4.185 -> 1.469 | 85.937 -> 25.389 | 92.348 -> 25.793 |
| 16 | 768 | 67.419 -> 24.018 | 4.221 -> 1.512 | 82.150 -> 26.017 | 91.154 -> 26.392 |

P50 and all rank1 event statistics are also in `summary.json`. All raw samples,
including monotonic rank entry timestamps and repeat IDs, are in
`target-timings.csv`. These are target endpoint latencies, not request TTFT/ITL.

## Launch, Gap and Rank-arrival Evidence

Counts below are per verification on rank0. Model GPU work is not eliminated.

| M | Eager CPU kernel-launch calls | Graph CPU kernel-launch calls | Graph-launch calls | Remaining graph GPU kernel nodes |
|---|---:|---:|---:|---:|
| 4 | 640 | 15 | 1 | 640-645 |
| 8 | 641 | 16 | 1 | 641 |
| 16 | 643 | 18 | 1 | 643 |

Counting graph submission as one call, CPU launch submissions decrease by
**97.50%, 97.35%, 97.05%**, respectively. A few static-buffer copies appear as
`memcpy32_post` kernels in the first key and as device-copy events later; this
is not fusion or removal of Transformer work. Every traced endpoint on both
ranks still has **84 NCCL kernels in matching order**. All target GPU events
were attached through external-ID/launch correlation; no time-window fallback
was needed in the real traces.

**Launch count reduction is not launch-API CPU-time reduction.** Under minimal
tracing, eager summed launch-API time is 3.63-4.62 ms, versus graph's
10.49-15.86 ms. An additional unprofiled Python timer around rank0
`CUDAGraph.replay()` also observes means of 9.37/13.96/16.84 ms for M4/M8/M16
(nine samples each). M16 includes a retained 29.95 ms sample; its median is
15.05 ms. Therefore the longer graph-launch call cannot be dismissed as solely
profiler overhead. Its internal CUDA/NCCL breakdown is not identified here.

In traces, 9.51-14.12 ms of graph API duration overlaps active device work;
CPU API time must not be added to GPU time or interpreted as fully removable
idle overhead. The measured gain is from amortizing the repeated Python/model
dispatch and submission sequence and reducing inter-layer rank drift/gaps,
not from proving each launch API itself became faster.

| M | Context | Rank0 device gap eager -> graph ms | Matched collective start skew eager -> graph us | Unprofiled RPC entry gap eager -> graph us |
|---|---:|---:|---:|---:|
| 4 | 256 | 29.819 -> 1.546 | 502.52 -> 5.56 | 35.27 -> 33.02 |
| 4 | 768 | 38.746 -> 2.226 | 611.67 -> 6.72 | 37.33 -> 47.78 |
| 8 | 256 | 42.238 -> 2.272 | 767.26 -> 6.36 | 40.17 -> 46.08 |
| 8 | 768 | 19.986 -> 2.336 | 695.66 -> 7.60 | 45.98 -> 44.40 |
| 16 | 256 | 40.511 -> 2.527 | 679.60 -> 8.46 | 38.19 -> 60.70 |
| 16 | 768 | 38.461 -> 2.610 | 693.52 -> 9.35 | 36.71 -> 50.15 |

Device gaps and matched collective skew are minimal-trace mechanism evidence,
not the unprofiled savings denominator. A GPU waiting inside NCCL is active
residency, not an idle gap. Rank1 gaps and joint residency are retained in the
summary. **RPC entry skew did not consistently improve.** Internal collective
arrival coordination improved substantially; claiming that both quantities
improved would be incorrect.

## Memory and Regression

After request cleanup but before graph release, live graph allocation above the
stable base is 2.24-8.13 MiB on rank0 and 0.74-2.95 MiB on rank1 for one key.
This is incremental live allocation, not total reservation or capture-time
peak. The one-time per-stream cuBLAS workspace is already in the stable base.
After each of all 12 timing cells, graph release returns to exactly the same
allocated bytes as the correctness gate. Reserved bytes are recorded separately;
allocator reservation is not treated as a request leak.

All waiting/running/used blocks/transactions/draft/target states and graph counts
are zero after final cleanup in both fresh processes. All timed verification
IDs/status match. No NCCL/CUDA error, deadlock or stale-state failure in final
runs. After process exit, both GPUs show 1 MiB used and no compute process.

CPU regressions: pre focused **22/22**, pre complete **115/115**; post focused
**22/22**, post complete **122/122**, including seven new graph-key/consensus
tests. Historical tests were not edited. Offline analyzer synthetic checks
cover percentile interpolation, correlation/window ownership and interval union.

## Limitations

1. This is warmed, fixed-input target verification followed by rollback, not a
   growing-context closed-loop serving benchmark. No E2E serving speedup is claimed.
2. Capture and warmup cost are excluded. Practical hit rate and capture
   amortization remain unmeasured. Exact max_k changes as a real request grows;
   blindly reusing a graph with another maximum context scalar is not authorized.
3. Only uniform q=4 at M4/M8/M16 is supported. Ragged/end-of-request shapes use
   eager. Different contexts and physical layouts are safe only within the key.
4. Minimal profiling expands timings; API totals, gaps and collective residency
   are diagnostic. The unprofiled replay API probe itself has only nine samples/M.
5. Two fresh processes support reproducibility here, not across hardware,
   transport topology, revisions or stochastic decoding. Peak capture memory
   and long-duration graph-cache churn are not fully characterized.
6. The BF16 cross-shape contract is unchanged. No old serial-parity failure was
   relabeled. This phase compares the same computation shape.

## Decision and Bounded Phase 4.4B Design

**Recommend an approval-gated Phase 4.4B, but do not implement it this turn.**
All three shapes pass same-shape correctness, repeated collective ordering and
cleanup. Every paired repeat improves target latency well above 5%. CPU launch
count, GPU gaps and internal matched-collective skew support the direction of
that gain. RPC entry skew is explicitly not part of the claimed improvement.
The Phase 4.3 dispatch/coordination hypothesis therefore has direct target-only
evidence, rather than a ceiling inferred from profiler-expanded residual time.

The minimal next design is an opt-in, bounded `GraphCache[key]` attached to the
batched target runtime. Keep input/output storage and one capture stream owned
by the engine. Keep request state, proposals, transactions and all validation
outside. Construct/agree the safe key before dispatch; a common hit replays,
otherwise both ranks use eager. Prepare captures only at a coordinated safe
point; do not fall back after partially submitting incompatible collectives.
Eviction must synchronize, destroy the executable, release/scrub buffers and
avoid touching request-owned feature copies.

Before adopting any context buckets or padding, establish their FlashAttention
metadata/kernel and BF16 numerical contract separately. Start with exact keys
and measure hit rate, capture cost, bounded memory and complete serving E2E;
decide whether a validated bucket policy is necessary. Do not silently introduce
one to turn this fixed-shape result into a serving claim.

The next useful work is safe target replay coverage and its **measured serving
impact**, not draft batching, custom NCCL or a new kernel. Re-profile the remaining
draft/target/host breakdown after integration; MLP remains the largest known
compute category, but this experiment does not justify optimizing it yet.

## New Files and Artifacts

New benchmark modules: `eagle3_graph_probe.py`, `eagle3_graph_key.py`,
`eagle3_graph_target.py`, `eagle3_phase44a.py`, `eagle3_phase44a_reuse.py`,
`eagle3_phase44a_summary.py`, all under `benchmarks/serving/`.
New test: `tests/test_eagle3_graph_prototype.py`. New architecture doc:
`docs/EAGLE3_PHASE4_4A.md`. This report, summary, raw timing CSV, commands and
artifact indices live under `benchmarks/eagle3-phase4_4a/`.

Large traces, source snapshots, intermediate failures and raw rank records stay
on `/root/autodl-tmp/`; `artifacts.json` and the supplemental index provide paths
and hashes. Existing dirty production/test files belong to earlier work and
were left unchanged. No Git commit or Phase 4.4B integration was performed.
