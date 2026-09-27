# Phase 4.4B: Bounded Target Graph Serving Integration

Status: complete. All 420 primary/sensitivity trials and 90 default-runner
controls finished, five repeats per cell, with no outlier removal. Correctness
passed. The bounded exact-key integration is opt-in; **this capture policy did
not improve EAGLE serving performance in the tested cold-cache regimes**.
No next-phase optimization or Git commit was performed.

## A. Integration Architecture

Only concurrent EAGLE target verification has an opt-in graph path:

```text
frozen scheduler / serial draft proposal / KV reservation
  -> frozen verification layout
  -> exact GraphKey + ordered-layout/generation agreement on both ranks
  -> cache hit: replay; miss: eager; eligible second occurrence: bounded capture
  -> unchanged target validation / greedy acceptance / commit / rollback
```

The default constructor still selects `ConcurrentModelRunner`; the optional
`target_graph_config` selects `GraphConcurrentModelRunner`. Ordinary LLMEngine,
ModelRunner, Scheduler, Qwen3, Attention, sampling and persistent draft state
are unchanged. Production modules do not import benchmark modules.

`GraphEntry` owns executable/static buffers and references to engine-owned
model/KV storage, never Sequence, request, transaction or accepted-length
state. Target features are cloned out of static graph storage. One shared
capture stream per rank bounds the engine-lifetime cuBLAS workspace cache.

The exact Phase 4.4A key includes M, batch, q lengths, exact max-q/max-k,
block-table width/size, dtype/device class, model/config identity, TP size,
training-mode identity and feature-layer selection. Only uniform q=4 with
M4/M8/M16 is supported. No context bucket, padding or extra ragged shape was
introduced.

CPU Gloo consensus occurs before model/NCCL submission. Key or availability
vetoes on an otherwise identical layout cause both ranks to use eager. An
actual ordered-layout/generation mismatch fails closed: eager would not make
different collective shapes safe. Control RPCs acknowledge both workers
before the original single-slot transport can be reused.

On capture, eager verification runs first and supplies that step's result.
Only the affected KV slots are snapshotted; three core warmup forwards and
graph recording occur at the coordinated safe point; the slots are restored
before normal acceptance. Eviction synchronizes both ranks, destroys the
executable, scrubs/releases static storage and leaves request/KV ownership
alone. It does not recover from a CUDA/NCCL failure after submission.

## B. Correctness Protocol

The GPU gate precedes all measured serving trials. It uses exact bitwise
same-shape comparison, not a raw-logit tolerance or a BF16 parity waiver.
It compares complete proposal/verification/acceptance/commit traces, output
IDs, C/D cursors, target features and target KV. The per-call observer restores
the same initial pages before eager and graph paths and compares all touched
pages, including committed and tentative regions.

The eight paired cases contain 52 request pairs: c1 at prompt lengths
255/256/257/768; c2 and c4 uniform replacement waves that actually hit M8/M16
graphs; c2/c4 heterogeneous lengths and output limits. An additional cap-2
fixture covers eviction, graph/eager/graph transitions and rank-1 key and
availability vetoes. An acceptance exception after verification exercises
the frozen request/transaction cleanup, followed by graph release and exit.

The initial pilot is retained. Its trace comparison failed on different
physical block IDs allocated across consecutive runs, while output/state
checks passed. The new audit normalizes only owner IDs and physical allocation
labels, preserving the raw mapping and bitwise KV checks. The earlier full
gate is also retained; uniform replacement waves were then added because its
mixed c2/c4 cases exercised fallback but not actual replay. No historical test
or inference arithmetic was changed to pass these checks.

Final results: focused CPU tests 22/22 and complete CPU suite 130/130, including
eight new policy/key tests; the pre-change complete suite was 122/122. The GPU
gate has 182 rank-level bitwise verification comparisons in its eight paired
cases. Both-rank committed feature hashes, cursors and phases also match.
There are seven heterogeneous-acceptance batches in these cases. Real replay
is exercised at all three supported M values, even though subsequent serving
measurements do not necessarily reuse those shapes/keys often enough.

Post-benchmark, all 1,280 graph/eager request pairs have identical output IDs
and all 240 paired trials have identical step/acceptance structure. Those
same 1,280 pairs also match ordinary output IDs on these fixed inputs. The
480 default-runner control requests match the same-process eager outputs.
Every measured trial has zero request/KV/transaction/state leftovers, zero
prefix-cache hits and zero dropped graph events. All nine serving processes
exit normally, without observed NCCL/CUDA errors or deadlocks. This is not a
claim of universal BF16 serial-versus-parallel token parity.

## C. Experimental Setup and Workloads

Qwen3-14B BF16, TP=2, two RTX 4090 GPUs, original Scheduler, fixed K=3,
persistent draft state, eager ordinary/draft paths. The target graph opt-in is
the only graph use. Model length 1024, batched-token limit 2048, sequence limit
4 and memory utilization 0.70 are the frozen Phase 4.2 configuration.

Software: PyTorch 2.8.0+cu128, FlashAttention 2.8.3, Triton 3.4.0,
Transformers 4.57.1, NCCL 2.27.3. Actual recorded inter-GPU topology is SYS,
not the NODE topology described in the project's older environment notes.
Rank 0/1 use GPU 0/1. Git HEAD is
`0ce2c1e054035766f2ee27027ffc309d58ad8c0c`; the pre-existing dirty worktree is
preserved, so source hashes/snapshots, not HEAD alone, identify this experiment.

Target revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`.
Dedicated draft revision: `3d13517724e81cb409ddf1d4650772ec52f1e18e`.
The checkpoint's undisclosed exact target training revision remains a prior
limitation; this phase does not change the checkpoint or numerical contract.

| Workload | Prompt tokens | Output limits |
|---|---|---|
| short-short | 64 | 32 |
| short-long | 64 | 128 |
| long-short | 768 | 32 |
| long-long | 768 | 128 |
| mixed-prompt | 64/192/384/768 | 64 |
| mixed-output | 256 | 16/32/64/128 |

Closed-loop active concurrency is 1/2/4. There are four requests per c1/c2
trial and eight per c4 trial, with immediate replacement until the fixed
request count completes. The unmodified Phase 4.2 builder determines inputs,
seeds, prompt isolation, warmup and rotated/reversed workload order. This is
an in-process pretokenized benchmark, not network serving or open-loop load.

The primary systems are ordinary, eager EAGLE, graph-never, and graph-second-4.
The `never` mode measures agreement/cache-glue overhead without capture.
The `second` policy captures from the second occurrence, with at most 16
captures per cache epoch, deterministic LRU, 4096 observed-key records and
8192 event records. Capacity 4 was chosen before results. Capacities 2/8 are
only tested on the preregistered long-long/mixed-output sensitivity cells.

Five repeats per cell yield 420 primary/sensitivity trials. Mode order also
rotates within each speculative process. Ordinary and speculative systems
require separate fresh processes; cross-system process order is not fully
interleaved. An additional 90 trials use the literal default concurrent
runner through the unmodified Phase 4.2 CLI, distinguishing it from the
same-process graph-disabled eager comparator.

All trials start with empty graph cache, occurrence history and capture
allowance. Warmup does not populate graphs and is not exhaustive for every
exact key. First-use costs, capture, eviction and agreement remain inside
measurement. End-of-trial graph release is outside request-serving time and
reported separately. All outliers are retained.

Throughput is total committed tokens divided by serving wall time, never
sum(request E2E). TTFT/E2E/TPOT percentiles pool requests; ITL percentiles pool
committed-token intervals. Tokens committed together retain equal timestamps,
so speculative ITL can legitimately have zero-valued intervals.

## D. Graph Eligibility and Key Distribution

Primary capacity-4 results, pooled over six workloads and five repeats:

| c | Verification batches | Eligible | Eligible rate | Ineligible | Unique keys/trial mean | New keys differing only in exact context |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2466 | 2367 | 95.99% | 99 | 63.2 | 1856/1896 |
| 2 | 1303 | 1215 | 93.25% | 88 | 38.0 | 1090/1140 |
| 4 | 1364 | 1176 | 86.22% | 188 | 32.9 | 940/987 |

The final column holds all other key fields fixed within each trial. New
block-table-width families account for 10/5/0 additional distinct keys at
c1/c2/c4. Width changes create a different safe key, not an unsafe replay.
The exact records, including max-k, widths and M, are retained in raw events.

Ineligible batches: c1 has 99 output-limit clips; c2 has 88 clips, including
66 ragged batches; c4 has 142 clips, including 126 ragged batches, plus 46
uniform q4 M12 batches, which Phase 4.4A did not authorize. These categories
overlap as stated, not as an additive count. No KV-capacity clipping occurred.

At c2/c4 **no exact key occurs more than twice in a measured trial**. The
policy captures on occurrence two and only replays subsequently. Thus zero
hits are explained by observed key histories, not a replay implementation
failure. Four requests at c2 and eight at c4 provide only two full replacement
waves. Growing contexts, different acceptance progress, batch turnover and
short trial horizon jointly limit reuse. This is an important qualification
to the exact-context key-explosion finding.

## E. Hit, Miss, Capture and Eviction

| c | Hits/all verifications | Eligible misses | Eager fallbacks | Captures | Evictions | Replays/capture | Replay P50/P95 per capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56/2466 = 2.27% | 2311 | 2410 | 260 | 166 | 0.215 | 0 / 2 |
| 2 | 0/1303 = 0% | 1215 | 1303 | 75 | 20 | 0 | 0 / 0 |
| 4 | 0/1364 = 0% | 1176 | 1364 | 181 | 100 | 0 | 0 / 0 |

Eligible miss rates are 97.63%/100%/100%; overall eager-fallback rates are
97.73%/100%/100%. Captures return eager results and are included in misses.
At c1, 260 capture lifetimes correspond to 230 distinct captured keys;
replays per distinct captured key are 0.243, and captures/replay are 4.64.
Cache churn, evictions/captures, is 63.85%/26.67%/55.25%.

The highest primary hit rate is c1 long-short, 9.41%, still insufficient to
pay back capture. c1 short-short/mixed-output have 5.05%/5.84%; c1 short-long
has only 0.27%. All c2/c4 workloads have zero hits. Detailed counts and
fallback reasons are in [cache-table.md](cache-table.md) and `summary.json`.

### Capacity Sensitivity

Only long-long and mixed-output are pooled in this table; do not compare its
throughput directly with a six-workload aggregate.

| c | Entries | tok/s | Hits | Captures | Evictions | Peak live allocated delta, MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 35.57 | 6 | 143 | 123 | 4.85 |
| 1 | 4 | 38.26 | 28 | 128 | 88 | 9.39 |
| 1 | 8 | 42.28 | 42 | 116 | 40 | 18.76 |
| 2 | 2 | 69.07 | 0 | 44 | 29 | 10.38 |
| 2 | 4 | 73.59 | 0 | 44 | 18 | 20.23 |
| 2 | 8 | 72.79 | 0 | 44 | 9 | 34.34 |
| 4 | 2 | 91.18 | 0 | 58 | 48 | 17.40 |
| 4 | 4 | 91.45 | 0 | 58 | 38 | 32.66 |
| 4 | 8 | 93.17 | 0 | 58 | 22 | 64.60 |

Larger capacity reduces eviction, and helps retain some c1 keys, but cannot
create a third occurrence at c2/c4. No capacity is selected as a new default.

## F. Capture Amortization

Capture P50/P95 is 266.37/313.97 ms at c1, 268.04/333.72 ms at c2 and
322.41/345.94 ms at c4. Totals for primary capacity 4 are 260/75/181 captures.

Example: c1 short-short repeat 0, M4, exact max-k=68. Capture costs 328.14 ms;
same-key eager endpoint averages 55.91 ms, hits average 19.61 ms. The estimated
saving is 36.30 ms/hit, requiring `ceil(328.14/36.30) = 10` replays. Actual
replays: **2**. The capture does not pay back even before charging eviction.

Primary captures with an observed positive same-key saving require 6-18
replays, median 14; actual lifetime reuse never exceeds two. There are:

- 516 primary capture lifetimes: 0 profitable, 40 reused but unprofitable,
  476 never reused.
- 486 distinct primary captured keys within trials: 0 profitable, 40 reused
  but unprofitable, 446 never reused.
- Across all three capacities: 979 lifetimes, 0 profitable, 79 reused but
  unprofitable, 900 never reused.

The estimate uses noncapture eager and replay endpoints for the same exact
key, not a randomized same-input microbenchmark. It is observational, and
cannot isolate clock/RPC skew. Zero-reuse captures are unprofitable regardless
of an unidentified per-hit saving. End-of-trial clear costs are recorded in
lifetime metadata but excluded from serving time; measured evictions remain
inside serving time. See [capture-lifetimes.csv](capture-lifetimes.csv).

## G. Target Hit Latency: Layer A

Only c1 has actual serving hits. Matching same-key eager means to the 56
observed hits gives 44.81 ms eager versus 20.83 ms graph endpoint, **2.15x**.
This includes the integrated verification endpoint, not just GPU kernels.
Across the 40 reused lifetimes the unweighted speedup median is 1.97x.

There is no c2/c4 hit-only serving estimate: the denominator is zero. The
frozen Phase 4.4A M8/M16 microbenchmark speedups are not substituted for these
missing serving observations, nor relabeled as end-to-end gains.

## H. EAGLE Eager Versus Graph: Layer B

Primary six-workload aggregates use total committed tokens / total wall time.

| c | Ordinary tok/s | Eager EAGLE tok/s | Never tok/s | Graph-4 tok/s | Graph/eager | Graph/ordinary |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 31.36 | 72.63 | 69.51 | 40.17 | 0.553x | 1.281x |
| 2 | 55.61 | 103.62 | 100.61 | 79.12 | 0.764x | 1.423x |
| 4 | 102.82 | 142.17 | 129.77 | 85.33 | 0.600x | 0.830x |

Graph reduces EAGLE throughput by 44.7%/23.6%/40.0% in these aggregates.
**All 18 primary cells have lower aggregate throughput than eager EAGLE.**
Only three of the 90 paired trial ratios exceed one, in three different cells;
no cell has a repeatable graph win. All samples remain in the results.

Literal default-runner EAGLE reaches 74.83/109.13/142.50 tok/s. Graph/default
ratios are 0.537x/0.725x/0.599x, confirming the conclusion. Default versus
same-process disabled differences are not attributed solely to the wrapper:
process order, clock variation and allocator history, including previous
graph releases, are confounders.

## I. Ordinary Versus Graph EAGLE: Layer C

Graph EAGLE still exceeds ordinary in some regimes, but those gains are
already present, and larger, in eager EAGLE. Best graph/ordinary cell is c1
mixed-prompt, 2.289x; that cell has **no capture or replay**, so the gain cannot
be credited to CUDA Graph. Worst is c4 mixed-prompt, 0.570x.

Ordinary/eager/graph requests per second are respectively
0.424/0.981/0.543 at c1, 0.751/1.400/1.069 at c2 and 1.390/1.921/1.153 at c4.
Per-workload throughput is in [throughput-table.md](throughput-table.md);
per-trial durations and ratios remain in `trials.csv` and `summary.json`.

## J. Latency Distributions

P50 / P95 / P99 in milliseconds, pooling the six workloads. Full ordinary,
never and capacity-specific cells are in [latency-table.md](latency-table.md).

| c | System | TTFT | TPOT | ITL | E2E |
|---|---|---:|---:|---:|---:|
| 1 | eager | 56.30 / 130.17 / 132.93 | 12.81 / 15.21 / 16.12 | 0 / 46.48 / 59.01 | 793.78 / 1829.36 / 1912.96 |
| 1 | graph-4 | 56.00 / 130.06 / 133.87 | 18.40 / 52.98 / 60.77 | 0 / 61.17 / 427.39 | 1222.37 / 5170.66 / 6929.25 |
| 2 | eager | 68.70 / 252.26 / 253.66 | 16.98 / 23.47 / 24.63 | 0 / 66.54 / 72.66 | 1011.08 / 2658.53 / 2891.74 |
| 2 | graph-4 | 65.47 / 251.87 / 254.26 | 19.20 / 44.41 / 61.65 | 0 / 69.19 / 339.60 | 1301.94 / 3847.00 / 7344.08 |
| 4 | eager | 129.43 / 501.74 / 504.03 | 24.14 / 31.95 / 40.75 | 0 / 85.72 / 130.72 | 1539.85 / 3631.37 / 3939.23 |
| 4 | graph-4 | 129.24 / 502.07 / 503.09 | 30.41 / 117.92 / 120.53 | 0 / 129.70 / 557.18 | 2297.73 / 9088.76 / 10315.09 |

TTFT is similar because prefill is unchanged and new closed-loop requests
arrive after the previous step returns. E2E, TPOT and ITL tails deteriorate.
Zero ITL medians reflect multi-token commits, not free per-token service.
These pooled P99 values are observations from small synthetic workloads,
not precise population-tail estimates.

## K. Memory and Cache Behavior

After initial shared-stream workspace creation, release returns allocated
memory to exactly 19,574,825,984 bytes on rank 0 and 16,778,348,544 on rank 1.
Before first capture the values are 19,566,306,304 and 16,769,828,864: the one
8,519,680-byte increment on each rank is the bounded engine capture-stream
workspace, not growth per key. Post-capture release values do not drift.

Primary peak live cache allocated-delta sums are 17.27/20.23/33.90 MiB at
c1/c2/c4. The largest across all capacities is 64.60 MiB (cap8, c4). Across
the complete matrix, rank-0 allocated/reserved peaks are 18.840/19.541 GiB;
rank-1 observed process peaks are 16.182/16.775 GiB. Rank 1's allocator peaks
are not independently reset each trial by the frozen collector, so they are
not claimed as isolated cell peaks. Instantaneous and release samples exist
on both ranks.

Allocation deltas include first-use workspace; reserved deltas are process
allocator observations, not exact graph-private-pool ownership. They can be
negative when capture/eviction releases unrelated cached allocator blocks.
Raw before/after, capture peaks, each entry's release and both-rank snapshots
are preserved; no negative reservation delta is interpreted as negative graph
memory. End-of-trial clear is distinct from timed in-trial eviction.

KV capacity is 88 blocks; maximum used blocks are 4/8/16 at c1/c2/c4. The
benchmark's peak *additional physical tentative blocks* is zero: tentative
verification rows fit already allocated pages. Page-boundary correctness is
covered by the separate 255/256/257 gate. These serving workloads do not stress
KV exhaustion, and no concurrency above four was tested.

Every final release has zero graph entries, used blocks, transactions, waiting
and running requests, and draft/target states. Both GPUs show 1 MiB used after
all processes exit. There is no observed memory leak, OOM or stale-state use
in the exercised lifecycle; arbitrary distributed-init/capture-failure
recovery is not claimed.

## L. Updated Wall-clock Bottleneck

Disjoint percentages of complete graph-4 serving wall time:

| c | Target eager forward | Replay forward | Capture | Eviction | Key/agreement | Draft | Prefill | Remaining |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.58% | 0.48% | 31.96% | 8.56% | 0.97% | 6.99% | 4.25% | 2.22% |
| 2 | 54.49% | 0% | 18.57% | 2.02% | 1.02% | 13.38% | 7.75% | 2.76% |
| 4 | 39.54% | 0% | 28.01% | 5.49% | 0.70% | 14.35% | 7.83% | 4.08% |

Remaining includes verification endpoint work, acceptance/commit, Scheduler,
prefill controls, driver refill and residual host/sync. Individual values,
including nested draft catch-up, are in `summary.json`. Every disjoint
partition reconciles with measured wall time; catch-up is not added on top
of draft and cache work is not counted again as target verification.

The original eager endpoint spends 78.29%/70.57%/61.46% on target verification
and 12.47%/17.68%/23.91% on draft. After integration, most verifications are
still eager, and capture plus eviction adds 40.52%/20.60%/33.50% of total
graph serving time. The new limiting issue is **poor reuse/amortization plus
remaining eager target execution**, not a new serial-draft bottleneck.
Even on c1 hit steps alone, draft occupies 27.62% and verification 70.35%.
There are no c2/c4 hit steps from which to infer a post-replay draft bottleneck.
These are host wall endpoints, not isolated GPU-kernel or NCCL durations.

Acceptance is unchanged: accepted/proposed is 87.22%/87.19%/86.78%, accepted
tokens per batched verification 2.55/4.83/9.22, and committed decode outputs
per verification 3.55/6.72/12.84. Draft forwards remain 7216/7219/14487.
Useful target forwards including prefill are 2586/1386/1482; graph capture
adds 780/225/543 warmup forwards, giving 3366/1611/2025 executed forwards.
The 260/75/181 graph recordings are reported separately, not as extra executed
GPU model forwards. This also corrects the interpretation of the frozen
collector's legacy `target_forwards` field, which excludes speculative prefill.

## M. Limits and Files

This is a cold-cache, finite-horizon characterization, not a warmed long-lived
service. In particular, second-occurrence capture cannot pay back in a trial
where no key occurs three times. The data do not prove that exact keys can
never pay back after many more repeated requests. Larger capacity alone did
not solve the observed problem. The workloads are synthetic and acceptance
is high; no generalization to all natural tasks, stochastic sampling or c>4
is asserted. No new Nsight/kernel attribution is inferred from wall timers.

Distributed errors after submission are fatal; availability/key vetoes were
injected before submission. Use a validated `GraphCacheConfig`. This phase
does not establish arbitrary malformed-configuration, CUDA-OOM or corrupted
registry recovery. Model/KV allocation and execution mode stay fixed for an
engine lifetime. The opt-in is not production-wide graph capture.

Modified existing files: `nanovllm/speculative/concurrent_engine.py` and
`nanovllm/speculative/batched_runtime.py`, only runner selection and an optional
verification callback. Four new runtime modules are `graph_policy.py`,
`graph_entry.py`, `graph_cache.py`, `graph_runner.py`. Three new benchmark/audit/
analysis modules and `tests/test_eagle3_graph_cache.py` are isolated additions.
No historical test was edited. Of 280 pre-existing hashed files, the other
278 are byte-identical. The four pre-existing tracked dirty files are untouched.

See [integration-only.diff](integration-only.diff),
[source-preservation.json](source-preservation.json), [commands.md](commands.md),
[summary.json](summary.json), [trials.csv](trials.csv),
[artifact hashes](artifacts.json), and the tables linked
above. Raw requests, steps, events, keys, failed pilot, complete gate, manifests,
logs and source snapshots remain under
`/root/autodl-tmp/eagle3-phase4.4b-20260924`, not as committed model/raw artifacts.

## N. Decision

1. Keep graph integration **opt-in**, with ordinary and concurrent eager paths
   unchanged by default. Do not select capacity 8 merely from the sensitivity
   subset or present target-only speedup as serving improvement.
2. No workload has a repeatable net gain over eager EAGLE with this cold-cache
   policy. Highest observed reuse is c1 long-short, still not profitable.
   Larger concurrency has zero measured replay reuse, not a correctness fault.
3. Do **not** enter draft batching on this evidence. Target eager work and
   capture lifecycle dominate; serial draft is not the new first bottleneck.
4. A separate context-bucketing/padded-graph **design and correctness audit is
   worth considering**, but is not yet authorized or shown safe/profitable.
   First distinguish finite-horizon cold-start cost from long-lived exact-key
   reuse with a preregistered longer-horizon experiment. If insufficient reuse
   persists, audit padded max-k/kernel/mask/metadata and BF16 numerical behavior
   independently before any serving implementation. Do not silently relax the
   current exact-key contract.
5. Phase 4.4B ends here. No policy tuning, draft batching, kernel optimization,
   context bucketing or Git commit follows without confirmation.
