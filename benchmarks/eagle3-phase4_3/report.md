# EAGLE-3 Phase 4.3: Target Verification Diagnostic Profiling

Date: 2026-09-23. **Diagnostic-only; not a replacement for Phase 4.2.**

## 1. Executive Decision

The largest *compute* cost is MLP GEMM, approximately 11.6 ms per target
verification per rank. However, target wall time is not predominantly attention
or LM head, and a long NCCL kernel is often waiting for its peer's host-driven
execution. Target eager dispatch/launch and cross-rank arrival gaps are the
highest-priority next investigation/optimization boundary.

Do not start grouped draft batching first. Do not rewrite NCCL based on the
large, unstable NCCL residency percentages. Recommend a bounded, separately
approved **target-only fixed-shape launch/replay experiment** for Phase 4.4.
No Phase 4.4 implementation, production change, or Git commit was made.

## 2. Setup and Scope

- Qwen/Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Dedicated EAGLE draft revision `3d13517724e81cb409ddf1d4650772ec52f1e18e`;
  same tokenizer, reference implementation and persistent state as Phase 4.1.
- BF16, TP=2, 2 x RTX 4090, eager, original Scheduler, fixed K=3.
- max_model_len=1024, max_num_batched_tokens=2048, max_num_seqs=4,
  gpu_memory_utilization=0.70. Speculative prefix caching remains disabled.
- PyTorch 2.8.0+cu128; Triton 3.4.0; FlashAttention 2.8.3;
  Transformers 4.57.1; NCCL 2.27.3; driver 570.124.04.
- `nsys` and `ncu` are unavailable. Dual-rank PyTorch/Kineto CPU+CUDA traces
  provide attribution; no Nsight Compute hardware-counter measurements exist.
- Actual topology: **SYS, GPU0 NUMA0 / GPU1 NUMA1**, not the NODE topology in
  early project context. NCCL INIT/GRAPH logs show `SHM/direct/direct` channels.
  Socket bootstrap messages do not establish that GPU collectives use Ethernet.
- Git HEAD `0ce2c1e054035766f2ee27027ffc309d58ad8c0c`, plus the existing dirty
  working tree, frozen by file hashes and a source snapshot.

Ordinary and speculative systems each cover mixed-output and long-short at
c1/c2/c4: 12 cells total. Inputs reuse the requested Phase 4.2 representatives,
not new tuning data. Mixed-output uses 256-token prompts and limits
16/32/64/128; long-short uses 768-token prompts and limit 32. At c1/c2 only
the first 1/2 limits are active in the captured initial batch. This is a
full-active-batch diagnostic slice, **not** the closed-loop throughput matrix.

Each cell runs six same-input replays: warmup, plain control, target-event
timing, rich profiler, minimal profiler, and collective-event timing. Capture
the first two decode/verification steps after all requests are prefetched;
drain every request afterward. c4 long-short requires two original prefill
steps under the unchanged token budget. First-step target prefixes match
between systems; second-step cursors naturally differ with accepted progress.

Two fresh primary processes, one per system, produced 48 dual-rank Chrome trace
files and 72 complete replay trials. There are only two captured target steps
per cell/mode: these data support attribution, not statistical performance or
tail-latency conclusions.

## 3. Timing Contract and Observer Effects

`T_verify_host` is rank-local endpoint time using perf_counter_ns.
`T_verify_gpu` is a CUDA-event elapsed stream span: **includes waits and idle**.
`T_device_execution` below is the union of correlated kernel/copy intervals,
not the CUDA-event span. CPU and GPU times are nested/asynchronous, not additive.

`T_cpu_pack_dispatch` measures the unchanged pickle/shared-memory write/event
notification. `T_rpc_wait` measures rank1 read_shm, including time awaiting the
driver's next command. It includes serial draft/other host work and must not
be added again to verification latency. `T_rank_sync` is represented by
collective CPU calls, NCCL residency, stream dependency events and observed
rank arrival skew; pure network-transfer time is not independently identified.

Profiler start/stop barriers and explicit event synchronization are outside
target spans. CUDA events and hooks still perturb enqueue timing. Rich mode
records module ranges and shapes; minimal mode removes those hooks and shape
recording. Neither is a formal performance measurement.

Mean rank0 target times, averaged across both workloads and two steps:

| System | c | rows | event-only host ms | event-only GPU span ms | minimal profiler host ms | rich GPU span ms |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 1 | 1 | 41.46 | 41.43 | 53.20 | 87.12 |
| ordinary | 2 | 2 | 59.34 | 59.30 | 64.55 | 106.48 |
| ordinary | 4 | 4 | 70.75 | 70.72 | 64.48 | 103.29 |
| speculative | 1 | 4 | 54.39 | 54.33 | 62.37 | 102.66 |
| speculative | 2 | 8 | 44.10 | 44.05 | 51.07 | 84.92 |
| speculative | 4 | 16 | 52.20 | 52.15 | 68.16 | 107.39 |

The variation, including minimal c4 ordinary being faster than event-only, is
retained. It shows why mode ratios are not exact overhead corrections.
Speculative rich replay whole-step time is 1.45-2.08x its plain-control mean;
we do not apply rich NCCL percentages to Phase 4.2 wall time.

Across speculative target-event controls, pack/dispatch is 39.7-59.7 us and
rank1 target entry follows rank0 by 16.5-38.5 us. Rank1 wait/read spans are
6.84-46.08 ms, mostly awaiting proposal/driver work, not transport overhead.
Rich CUDA launch API CPU sums are about 3.51-3.58 ms/verification;
cudaStreamSynchronize CPU spans vary 0.19-39.14 ms and overlap queued GPU work.
These are not disjoint additional costs.

## 4. Kernel Attribution

Kernel correlation IDs map to enclosing CPU module ranges. NCCL overrides its
enclosing projection category. Copies/memsets are separate. Timeline analysis
uses interval unions and intersections, with offline tests for double counting.
Exact names, shapes, counts and individual events remain in raw artifacts.

Rank0 compute service time, rich traces, milliseconds per verification:

| Category | c1, 4 rows | c2, 8 rows | c4, 16 rows |
|---|---:|---:|---:|
| embedding | 0.012 | 0.014 | 0.017 |
| QKV projection | 1.714 | 1.711 | 1.727 |
| attention output projection | 1.221 | 1.221 | 1.234 |
| MLP gate/up | 7.621 | 7.636 | 7.676 |
| MLP down | 3.958 | 3.965 | 3.989 |
| FlashAttention | 0.807 | 0.814 | 0.817 |
| LM head including local concatenation, excluding gather | 0.822 | 0.827 | 0.834 |
| RMSNorm / activation | 0.356 | 0.362 | 0.373 |
| RoPE | 0.106 | 0.106 | 0.108 |
| KV store | 0.046 | 0.047 | 0.047 |
| other CUDA + copies/sets | 0.059 | 0.075 | 0.088 |

MLP is about **69% of non-NCCL GPU service time**; QKV/output projections
about 17.5%, attention about 4.8%, LM head about 4.9%. These percentages use
**compute service**, not verification wall time. Attention is about 0.53 ms
on mixed-output and 1.09-1.10 ms on long-short, so it is context-sensitive but
not the leading cost in this context range.

Minimal traces, rank0 timeline partition (percent of first-to-last GPU event
envelope; mutually exclusive in these observed traces):

| c | non-NCCL compute/copy | NCCL resident | no captured device work |
|---:|---:|---:|---:|
| 1 | 26.94% | 56.92% | 16.14% |
| 2 | 33.10% | 4.58% | 62.32% |
| 4 | 24.93% | 61.31% | 13.76% |

The NCCL/idle split depends heavily on which rank gets ahead. Neither column
alone identifies a transport or Python bottleneck. Both ranks must be read
together. No measured same-rank compute/NCCL overlap occurs in rich or minimal
target windows (intersection = 0 us at profiler resolution).

## 5. Actual TP Collectives and Rank Imbalance

Per speculative verification, **on each rank**:

| Location | API | Calls |
|---|---|---:|
| vocab-parallel embedding | all_reduce SUM BF16 | 1 |
| attention o_proj, one per layer | all_reduce SUM BF16 | 40 |
| MLP down_proj, one per layer | all_reduce SUM BF16 | 40 |
| finite-feature/logit check | all_reduce MIN int32 | 1 |
| vocab-parallel logits | gather to rank0 | 1 |
| transaction/rank status | all_gather int64 | 1 |
| **Total** | **82 all_reduce + 1 gather + 1 all_gather** | **84** |

Thus two collectives per transformer layer, plus four outside layers. Ordinary
decode has 81 all_reduce + 1 gather = 82. These are verification-only counts,
excluding proposal, commit, close, diagnostic barriers and cleanup.

Observed GPU symbols include AllReduce BF16 RING_LL, AllReduce u32 RING_LL,
SendRecv for gather, and AllGather RING_LL. The finite check's API is MIN even
though the shared observed u32 kernel symbol contains `Sum`; the kernel name
alone is not used to infer the reduction operation. This audit does not alter
the original collective arguments.

Projection all_reduce payloads per rank are 40,960 / 81,920 / 163,840 bytes at
c1/c2/c4. LM gather local logits are 607,744 / 1,215,488 / 2,430,976 bytes.
The finite check is four bytes; status is 64 / 128 / 256 bytes per rank.

Minimal trace means across the two workloads:

| c | rank0 NCCL ms (% timeline) | rank1 NCCL ms (% timeline) | rank0 idle ms | rank1 idle ms | matched NCCL joint residency ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.33 (56.92%) | 3.49 (5.63%) | 10.02 | 41.76 | 1.187 |
| 2 | 2.32 (4.58%) | 23.96 (47.30%) | 31.58 | 9.96 | 1.555 |
| 4 | 41.58 (61.31%) | 7.14 (10.56%) | 9.33 | 43.60 | 3.068 |

NCCL residency changes sides: rank0 is not intrinsically always slower/faster.
Matching collective kernel start skew averages 183-637 us across sampled
verification windows, while end skew averages only 0.9-3.1 us. This strongly
supports peer-arrival waiting. Joint residency is the intersection of matching
rank0/rank1 kernel intervals after aligning Kineto base epochs. It is about
1.9% / 3.1% / 4.5% of the corresponding target timeline, **not** a measured
pure-wire fraction or a transport-speedup guarantee.

Collective-event controls corroborate varying rank wait without profiler:
per-cell mean stream dependency brackets span 5.28-48.68 ms on rank0 and
7.04-26.65 ms on rank1. These brackets include host enqueue delays and stream
dependencies, not just NCCL kernel execution.

### Retained Outlier

Ordinary c4 long-short collective-event replay, second step: 311.94 ms host
step, 311.80 ms rank0 GPU-event span. Rank0 shows a **260.18 ms CPU gap** after
collective #6 (layer2 MLP down) and before #7 (layer3 attention output).
Rank1's #7 stream bracket is **257.29 ms**, despite its collective Python call
taking only 0.154 ms. No sample was removed. This is concrete evidence that
a long collective span can be caused by peer host progress. This mode has no
CPU stack trace, so GC versus OS scheduling versus dispatcher stall remains
unresolved; it is not evidence of a recurring production GC problem.

## 6. Shapes and Why Per-Row Time Falls

Each rank has 20 Q heads, 4 K/V heads, head_dim=128 and 40 layers. Per-rank
linear algebra uses `X[M,K] @ W[N,K].T`:

| Projection | K | N | calls/forward |
|---|---:|---:|---:|
| QKV | 5120 | 3584 | 40 |
| attention output | 2560 | 5120 | 40 |
| MLP gate/up | 5120 | 17408 | 40 |
| MLP down | 8704 | 5120 | 40 |
| LM head | 5120 | 75968 | 1 |

Ordinary M=1/2/4; speculative M=4/8/16. The actual M=1 ordinary kernel is
`internal::gemvx::kernel`. M=2/4/8/16 projection kernels use the observed
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8`
family. This is direct evidence of a GEMV-to-skinny-tensorop-GEMM transition,
not a claim that these become large, compute-saturated GEMMs.

The five projection categories together cost about 15.33 / 15.36 / 15.46 ms
at speculative c1/c2/c4. Four times the rows do not require four weight passes
or four copies of each launch. The 16-row tile family also stays unchanged.
The once-per-forward BF16 projection weight volume computed from shapes is
about 14.0 GB per rank. A weights-only, read-once arithmetic-intensity model
gives M FLOP/byte: 4 -> 8 -> 16. This is a model, **not measured DRAM traffic,
cache hit rate, achieved occupancy or FLOPs**. Nsight Compute would be needed
for those stronger claims.

Other observed amortization:

- 84 collectives/verification stay constant; collectives per query row become
  21 -> 10.5 -> 5.25. Joint-residency ms/row drops 0.297 -> 0.194 -> 0.192.
- Target GPU launch counts are 640 -> 641 -> 643, or 160 -> 80.1 -> 40.2 per
  row. Ordinary launches are 659/658/658 at M=1/2/4.
- One LM-head projection remains about 0.82-0.83 ms as rows increase.
- Attention remains one batched varlen invocation per layer, around
  0.81 ms/forward across the two sampled prompt lengths; KV stores remain 40.
- Shared-memory pack/dispatch is only around 0.04-0.06 ms and does not grow
  proportionally to query rows. It amortizes, but is too small to explain
  the full improvement by itself.

Event-only target host ms/row here is 13.60 -> 5.51 -> 3.26 for speculative,
versus 41.46 -> 29.67 -> 17.69 for ordinary. **Do not interpret these few-step
ratios as new serving speedups.** Phase 4.2's 12.33 -> 6.59 -> 4.01 ms/row
uses a much wider population, including smaller/ragged tail batches. The
diagnostics support amortization mechanisms, not exact reproduction of those
aggregate values.

Ordinary attention calls flash_attn_with_kvcache (q=1) and emits split-KV plus
combine kernels, 80 across 40 layers. Verification uses paged
flash_attn_varlen_func (q=4/request), 40 observed split-KV kernels without the
ordinary combine stage. It processes more rows per forward, but its absolute
attention time is slightly higher; attention kernel-count reduction is not
the dominant source of serving gains.

## 7. Optimization Decision Table

The bounds below are deliberately optimistic zero-cost/scheduling ceilings,
not predicted gains. Kernel-service savings are sample-local and may be
hidden by another critical path. Profiler-expanded waits cannot be transferred
as-is to formal serving. Candidate ceilings must not be added together.

| Candidate | Potential / evidence | Optimistic ceiling or identifiable bound | Complexity / correctness risk | Recommendation |
|---|---|---|---|---|
| target host dispatch / launch | High: hundreds of launches, large peer-arrival gaps; rank-wait changes sides | Minimal-trace noncompute residual is 33.9-50.9 ms/verify; overgenerous ceiling including real communication and observer overhead | Medium-high / medium-high | First bounded experiment; distinguish hot-path dispatch from RPC transport |
| CUDA Graph target replay | High hypothesis: can address repeated launch/dispatch, not matrix math | Same residual ceiling as above, not an additional benefit; uncapturable checks and real communication remain | High / high: TP order, addresses, dynamic layouts, feature ownership | Candidate for explicitly authorized Phase 4.4 prototype only |
| verification GEMM optimization | Medium: all projections ~15.3-15.5 ms; MLP ~11.6 ms | Zero all GEMMs saves at most their ~15.5 ms service under unchanged dependencies; 2x MLP gives at most ~5.8 ms | High / medium-high numerical risk | Get hardware counters before writing kernels; already uses library tensorop/GEMV paths |
| TP communication optimization | Medium investigation; 84 collectives, but large residency includes peer wait | Whole-target span is a loose ceiling; no useful transport-only upper bound identified. Joint 1.19-3.07 ms is a sensitivity proxy, not a rigorous wire-time bound | High / high | Do not infer link saturation from NCCL residency; separate rank-arrival effects first |
| compute/communication overlap | Not now: zero observed overlap, row-parallel results feed next operations | Mathematical overlap cannot exceed ~16.7-16.9 ms compute service, but dependencies invalidate most of that ceiling | High / high | No blind extra streams; requires a legitimate independent-work schedule |
| concurrent/grouped draft batching | Medium, secondary: Phase 4.2 draft 10.96/17.25/24.32% | Even free entire draft gives serving ceiling 1.123x/1.209x/1.321x; grouping cannot remove all draft work | High / high ownership and BF16 shape risk | Defer until target launch experiment is evaluated |
| attention optimization | Not now: ~0.53-1.10 ms/verify | At most ~1.10 ms if attention computation vanished | High / high mask/KV/numerical risk | Poor first target for these <=768 prompt samples |
| LM-head optimization | Not now as a standalone compute project: ~0.82-0.83 ms | At most ~0.84 ms local compute; gather is counted separately under TP | Medium-high / high token-selection sensitivity | Preserve exact vocab-parallel semantics; low immediate leverage |
| RPC packing rewrite alone | Not now: measured 0.040-0.060 ms | At most ~0.060 ms/verify in these controls | Medium / medium synchronization risk | Do not confuse rank1 waiting for draft with serialization cost |
| KV transaction optimization | Not now: Phase 4.2 reserve 0.02-0.04%, accept/commit/postprocess 0.81-1.03% | Even deleting both entire categories saves only ~0.83-1.07% serving time; KV-only share is smaller | Medium / high state safety risk | Keep frozen ownership/rollback semantics |

## 8. Correctness, Preservation and Artifacts

- Before: Phase 4.1 focused tests **22/22**; complete CPU suite **110/110**.
- After: focused **22/22**; complete CPU suite **115/115**, including five
  new offline attribution/interval-analysis tests. Historical tests unchanged.
- All six same-system replay modes produce identical complete outputs and
  identical output prefixes at the two captured steps in every cell.
- Ordinary versus speculative control outputs match **14/14 request pairs**.
  This does not broaden the accepted BF16 cross-shape numerical contract.
- All 168 primary replay requests finish; waiting/running/used blocks,
  transactions, host request/draft maps and target state on both ranks are zero.
- 56 hashed production/test/frozen-artifact files are identical before/after
  each primary process. No NCCL/CUDA/deadlock error in completed runs.
- Both GPUs returned to 1 MiB/no active compute after exit.
- No production files, frozen tests, Phase 4.1/4.2 results or algorithms changed.

New files are the Phase 4.3 diagnostic runner, offline analyzer, five-test
analysis test module, protocol doc, this report, summary JSON and target CSV.
Existing unrelated dirty files were preserved. No Git commit was made.

Primary raw data:

- ordinary: `/root/autodl-tmp/eagle3-phase4.3-final-20260923`
- speculative: `/root/autodl-tmp/eagle3-phase4.3-spec-final-20260923`
- compact results: `summary.json`; all individual target rows: `target-table.csv`
- complete event classifications, full metadata and detailed summary remain
  outside the repository; see `artifacts.json` and `commands.md`.

Retained supplemental runs:

- `eagle3-phase4.3-pilot`: completed c1 mixed-output prototype.
- `eagle3-phase4.3-raw-20260923`: early harness stopped on c4 long-short's
  incorrect single-prefill assertion. No production fix was required.
- speculative files under `eagle3-phase4.3-final-20260923` plus
  `eagle43-speculative-final.log`: completed verbose NCCL TUNING-log run.
  TUNING emitted per-collective log lines, so a new process repeated all six
  cells with INIT/GRAPH/NET/ENV only. The verbose measurements and their
  separate analysis are retained, not silently deleted or treated as outliers.

## 9. Limitations

Two early steps per cell, no repeated performance matrix; fixed replay-mode
order; profiler allocation/GC and CPU scheduling may perturb subsequent modes.
The event-control outlier remains unexplained below the host-gap localization.
Neither the profiler nor CUDA-event controls prove exact uninstrumented
NCCL/host fractions. No hardware counters or CPU stack sampling were collected.
GPU idle is an observed device gap, not a direct measurement of Python work.
Cross-rank timestamps use Kineto's calibrated shared host epoch and are not
sub-microsecond synchronization proofs. Only K=3, c<=4, early full batches and
256/768-token prompt contexts were sampled. No transport or clock setting was
tuned. These diagnostics cannot establish a future optimization's speedup.

## 10. Answers and Proposed Phase 4.4

1. **Where does the 61-81% verification stage go?** Compute is ~16.7-16.9 ms
   per sampled forward; the remaining diagnostic timeline includes eager
   dispatch/device gaps and peer-rank NCCL waits. There is no justified single
   pure-network or pure-Python percentage for unprofiled serving.
2. **Attention primary?** No, ~0.53-1.10 ms in these samples.
3. **MLP/GEMM primary?** MLP is the largest compute category, ~11.6 ms; that
   does not make it the largest removable wall-clock bottleneck.
4. **LM head significant?** Only ~0.83 ms local compute; gather is separate.
5. **NCCL fraction?** Minimal rank0 56.92%/4.58%/61.31%, rank1
   5.63%/47.30%/10.56%; joint residency about 1.9%/3.1%/4.5%.
   These are diagnostic residency, not pure communication bandwidth cost.
6. **Overlap?** No same-rank compute/NCCL overlap observed.
7. **Why does time/query fall?** Nearly fixed projection service, launches,
   weight footprint and collective count are shared by 4/8/16 rows. Actual
   GEMV/tensorop kernel selection and shape/count traces support this.
8. **Best next module?** Target execution's eager host launch/dispatch path
   and associated rank arrival coordination, not shared-memory packing alone.
9. **Postpone draft batching?** Yes. It is meaningful at c4 but still secondary.
10. **Minimum Phase 4.4?** After approval, prototype only fixed-shape target
    replay at M=4/8/16 with preallocated input/layout buffers and unchanged
    model math/TP order. A CUDA Graph version requires explicit authorization,
    capture-capability checks and dual-rank replay correctness first. Keep
    draft, Scheduler, acceptance, commit/rollback, finite/status checks and
    request ownership outside this experiment; do not introduce graph capture
    by changing their semantics. Measure both ordinary and speculative paths
    from fresh processes and retain eager fallback. First prove launch/gap
    reduction, then run repeated serving comparisons. If capture safety is not
    viable, stop and reassess rather than expanding into fused kernels or
    scheduler changes.

**Phase 4.3 complete. Wait for confirmation before Phase 4.4.**
