# Phase 5.1: Target MLP Region Replay

Date: 2026-09-25. **Decision: Keep Experimental.**

There is a repeatable steady-state benefit in the tested single-request regime,
but not enough evidence for general concurrent-serving adoption under the
preregistered gates. Default eager remains unchanged. No production file,
historical test or historical artifact was modified in this phase. No Git commit.

## 1. Choice and Scope

Phase5.0 proposed context-independent `o_proj -> residual/norm -> MLP` replay.
This experiment narrows that to the existing `Qwen3MLP.forward`: gate/up, existing
SiLU/multiply, down projection and its existing all_reduce. Attention, o_proj,
norms, feature extraction and LM head remain eager. No TP arithmetic, kernel,
acceptance, Scheduler, draft or KV-state change.

The MLP is already a self-contained module and was the largest non-NCCL compute
region in the earlier profile. This boundary avoids moving residual ownership
across modules and requires no production refactor. It is the smallest useful
boundary tested, **not proof that it is globally optimal**. The question remains
whether context-independent launch replay can help real serving without exact
context-key capture. Broadening the region was not necessary to answer it.

An explicit benchmark-only runner owns 40 layers x 3 exact row counts (M=4/8/16),
120 graph entries per rank. Uniform q=4 with batch1/2/4 is eligible; clipped,
mixed-q and batch3 cases fall back to unchanged eager. No context padding,
dynamic capture, LRU or captured request KV. Each replay copies an owned input
and returns its graph output to the next eager operator. Bindings are restored
on normal return and failure. All ranks initialize entries in the same order.

See [architecture](../../docs/EAGLE3_PHASE5_1.md) and
[decision log](decision-log.md) for alternatives and evidence-driven changes.

## 2. Setup and Evidence Inventory

- Qwen3-14B BF16, TP2, two RTX4090, SYS topology/no NVLink, K3 persistent draft,
  original Scheduler, speculative prefix cache disabled.
- Target revision `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Draft revision `3d13517724e81cb409ddf1d4650772ec52f1e18e`.
- Pinned EAGLE export `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` is not a Git
  working tree. Exact target training revision remains unpublished/unknown.
- Torch2.8/cu128, FlashAttention2.8.3, NCCL2.27.3+cuda12.9. Full versions, source
  snapshots, GPU topology, commands and asset hashes accompany the raw manifests.
- Current paired runs use max_model_len2048, versus historical Phase4.2's1024,
  to accommodate correctness at1024+output. Both current systems use identical
  config and historical workload vectors. Historical absolute throughput is
  **not** used as the current baseline denominator.
- Other shared engine settings: max_num_batched_tokens2048, max_num_seqs4,
  gpu_memory_utilization0.7; engine enforce_eager=True, with only this opt-in
  verification MLP callback using region graphs. No full-model graph mode.
- 54 fixed-input cells, 3,150 raw timing samples, 252 rank trace files.
- Six fresh serving processes, 7 workloads x 3 concurrencies x 2 systems x
  5 measured repeats = **210 trials / 105 pairs**; 1,120 completed requests and
  81,600 committed output tokens. No sample/outlier removed.

Serving covers six frozen Phase4.2 families: short-short, short-long,
long-short, long-long, mixed-prompt and mixed-output. The preregistered held-out
family uses archive/version/checksum text, prompt lengths127/383/895/511 and
output limits48/96/40/80. It was not selected using performance outcomes.
There are4 requests/trial at c1/2 and8 at c4, closed-loop refill, fixed greedy
inputs. Warmup covers all families with short outputs and is outside measurement.

Literal and optimized runs are separate processes, with system order reversed
at c2 and workload order rotated/reversed across repeats. Five repeats share a
process per system/concurrency: **not five independent process pairs**. No
profiler runs inside the measured serving trials. These are in-process,
pretokenized serving measurements, not network/API latency.

## 3. Eligible-Miss Diagnosis

Never-capture controls use direct eager, a disabled GraphCache wrapper, shadow
key computation, agreement-only and the actual enabled GraphCache with
`capture_policy=never`. Representative median host RPC times (ms):

| batch/context | literal eager | enabled never-capture miss | shadow |
|---|---:|---:|---:|
| 1/256 | 37.612 | 40.210 | 37.428 |
| 1/768 | 41.854 | 42.985 | 41.289 |
| 2/256 | 45.101 | 43.916 | 41.719 |
| 2/768 | 42.603 | 43.583 | 42.301 |
| 4/256 | 42.059 | 43.963 | 42.448 |
| 4/768 | 41.843 | 42.820 | 41.945 |

Fresh wrapper/metadata/agreement cost is small and variable. It does **not**
reproduce the full historical long-run eligible-miss penalty.

A separate lifecycle experiment captures one full-target graph, never replays
it, and measures before capture, with graph live, and after graph release.
NCCL's `everCaptured` condition adds ordering to subsequent eager collectives;
it survives graph destruction in the inspected version. This is supported by
the [version-pinned NCCL source](https://github.com/NVIDIA/nccl/blob/72d2432094d6ae36abd6e511c3a16a2d052dbf94/src/misc/strongstream.cc).

Across two verify/rollback pairs, cudaEventRecord increases510->680 and
cudaStreamWaitEvent340->510 after capture, persisting after release. A separate
`NCCL_GRAPH_MIXING_SUPPORT=0` **capture-with-zero-replays** control removes these
extra calls. Real replay and serving always retain default mixing support:
disabling it forbids outstanding graph/noncaptured operations even when stream
ordered, per the [NVIDIA contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support).

This establishes a persistent capture-induced ordering mechanism, **not the
complete wall-time cause of Phase4.4C**. Wall effects are modest/noisy here;
historical long-horizon capture churn was not reproduced. Consequently a
post-capture eager mode is not treated as a pristine serving control. Fresh
literal processes never capture anything. Default mixing costs remain included
in optimized serving, rather than being subtracted or disabled.

## 4. Prototype and Mechanism

One-layer replay had small/noisy total-target effects, insufficient on its own.
All-layer fixed-prefix results were consistent enough to proceed:

| batch/context | post-capture eager ms | MLP replay ms |
|---|---:|---:|
| 1/256 | 41.477 | 37.355 |
| 1/768 | 40.570 | 37.362 |
| 2/256 | 41.862 | 36.987 |
| 2/768 | 41.532 | 37.478 |
| 4/256 | 42.093 | 37.945 |
| 4/768 | 42.594 | 37.678 |

These are within-process medians,25 samples/mode/cell. They are **not** the
fresh-process serving speedup. All samples, including initial ~53ms one-layer
cells, remain in the raw data.

At context768, rank0 profiled verification has CPU launch API counts
640->520,641->521,643->523 for batch1/2/4: **120 fewer submissions**, roughly19%.
GPU kernel counts stay640/641/643; NCCL kernels stay84. This is launch replay,
not kernel fusion or fewer collectives. Forty extra input copies add about
37-39us of device copy time. Forty graph launches occupy about1.00-1.04ms of
CPU API time in the profiled windows; do not interpret profiler timings as pure
unprofiled graph overhead.

GPU idle-gap estimates for these windows decrease33.87->19.31ms and
34.99->27.02ms at batch1/2, but increase34.30->43.89ms at batch4. This profiled
stall is retained. Matched NCCL start-time rank skew is likewise not uniformly
reduced: batch4 median skew across the two windows is175.77/187.10us eager
versus362.48/201.52us replay. Thus the hypothesis of universal gap/skew reduction
is **not supported**. Collective residency includes waiting, not just wire time.
The defensible mechanism is reduced CPU submission with preserved compute and
collective count, whose net value must be measured in serving.

See [mechanism table](mechanism-table.md). The trace annotation names both
verify and rollback alike; offline analysis explicitly labels the two alternating
pairs and preserves both, rather than halving verification counts by averaging
them together.

## 5. Correctness and Preservation

- Complete CPU suite: **143 passed**, including4 new tests. Historical tests
  unchanged. Printed failing numerical fixture cases are deliberate unit-test
  inputs, not suite failures.
- Sixteen generation/limit cases cover c1/2/4, contexts255/256/257/1024 and
  heterogeneous batches with output clipping1/2/4/5; explicit EOS and an injected
  acceptance exception also pass.
-64 audit points per rank, each with three same-state repeated replays:
  byte-identical features and involved KV pages on both ranks, byte-identical
  logits on rank0, finite checks, proposal/acceptance/cursor/output agreement.
  Rank state/status agreement uses the existing coordinator; this is not a new
  independent global feature-replication proof.
- All105 serving pairs have identical output IDs and aggregate speculation
  statistics. Every trial ends with zero requests, drafts, target states, used
  blocks and transactions. Prefix-cache hits0. No observed NCCL/CUDA/deadlock
  error; every process exits normally.
-339 protected preexisting files have unchanged hashes. Runtime code tested by
  correctness and all serving processes has identical SHA256. The earlier
  prototype differs only by subsequently added lifecycle diagnostic controls
  and GraphCache teardown, not MLP replay math.

No arbitrary logit tolerance was used. Historical BF16 cross-shape serial-token
parity failures remain historical failures; this audit demands exact same-shape
parity and does not reclassify those results.

## 6. Serving Results

Throughput is committed output tokens / complete serving wall time, not summed
request E2E. Speedup is eager wall time / optimized wall time for identical
output work. The aggregate below is the geometric mean of35 paired ratios/c.

| concurrency | serving speedup | improving repeat aggregates | held-out median | capture startup | projected payback requests |
|---|---:|---:|---:|---:|---:|
| 1 | 1.0703x | 5/5 | 1.0752x | 11.62s | 170 |
| 2 | 1.0190x | 5/5 | 1.0282x | 11.77s | 910 |
| 4 | 1.0253x | 5/5 | 1.0273x | 11.27s | 834 |

The largest cell-median gain is c1 short-long,1.0811x; the smallest is c2
short-short,1.0012x. Some individual paired repeats regress, including c2
short-short0.9774x. All are retained. See the complete
[throughput/repeat table](serving-table.md) and
[TTFT/E2E/TPOT/ITL P50/P95/P99 table](latency-table.md).

TTFT is not optimized: prefill stays eager. Some TTFT tails regress, notably
c4 short-short pooled P95 TTFT66.22->88.34ms, and c1 short-long60.93->71.83ms.
There is no >10% **repeat-median P95 ITL** regression (the preregistered tail
gate), but this does not mean every latency percentile improves. c2
short-short E2E P99 worsens683.75->713.41ms. Burst-committed tokens retain zero
ITLs, so P50 ITL is0 in these cells; this is not zero decode service cost.

Accepted/proposed ranges0.758-0.992 across cells and is identical between paired
systems. Per-cell acceptance, accepted tokens/verification and target/draft
forward counts are in [summary.json](summary.json). These relatively high
acceptance workloads do not establish low-acceptance generalization.

## 7. Why Serving Gains Shrink

| c/mode | draft fraction | target verify fraction | prefill fraction | accept/commit fraction |
|---|---:|---:|---:|---:|
| 1 eager | 12.79% | 77.67% | 8.02% | 0.81% |
| 1 replay | 13.56% | 76.18% | 8.64% | 0.86% |
| 2 eager | 18.71% | 68.73% | 11.01% | 0.85% |
| 2 replay | 19.38% | 67.73% | 11.30% | 0.88% |
| 4 eager | 23.63% | 61.18% | 13.62% | 0.98% |
| 4 replay | 24.19% | 60.09% | 14.06% | 1.02% |

These are pooled wall fractions, not pure GPU kernel fractions. Draft includes
catch-up; remaining time includes reservation, scheduling, synchronization and
driver refill. Higher concurrency reduces target verification's share while
leaving more unoptimized work. Eligibility also falls: e.g. held-out c1 is95.5%
of decode steps, c4 is74.1%, due to batch3/mixed clipped q fallbacks. Those steps
are neither padded nor discarded. Copy/replay/collective ordering and40 graph
boundaries consume part of the opportunity. This is consistent with smaller net
gains, but does not isolate an exact causal percentage for each factor.

## 8. Memory and Startup

Full capture costs11.27-11.77s in the serving processes. One entry per
layer/shape yields120/rank, about30MiB additional allocated memory/rank and
roughly274MiB additional allocator reservation. Serving peaks:

| c | eager rank0 allocated GiB | replay rank0 allocated GiB | replay rank1 allocated GiB | replay rank0 reserved GiB |
|---|---:|---:|---:|---:|
| 1 | 18.683 | 18.712 | 16.104 | 19.197 |
| 2 | 18.711 | 18.740 | 16.128 | 19.324 |
| 4 | 18.769 | 18.799 | 16.180 | 19.395 |

Worst conservative physical headroom estimate during serving is2.927GiB on
rank0 and5.467GiB on rank1, above the1GiB gate. This subtracts peak-minus-current
allocated usage from post-trial free memory; it is not continuous physical
memory telemetry. Audit snapshots use additional diagnostic memory and are not
the serving-memory denominator. KV use peaks16 blocks out of88; request cleanup
is zero. Retained graph storage is registry-owned until engine teardown, not a
request leak.

Payback is `capture_seconds / pooled saved_seconds_per_request`, rounded up.
The170/910/834 requests above are **projections**, not observed long-horizon
amortization. No512-request optimized serving run was conducted. Common model
load/warmup are outside steady state; the added capture cost is reported rather
than hidden. Phase5.0's stronger long-horizon validation remains unfulfilled.

## 9. Adoption Gates and Decision

| preregistered gate | outcome |
|---|---|
| exact correctness/preservation/cleanup | pass in tested coverage |
| >=1.05x in at least2 concurrency groups | **fail: only c1** |
| >=4/5 improving repeats and held-out >1 in qualifying groups | pass for c1 |
| no >10% repeat-median P95 ITL regression | pass; other tails still reported |
| >=1GiB physical headroom per rank | pass, with stated estimation method |
| projected payback <=512 requests in qualifying groups | pass for c1 only; c2/4 do not meet512 |

**Keep Experimental, not Adopt and not Reject.** Real serving value exists in
the tested low-concurrency regime. General promotion is unjustified: concurrent
gains are small, startup payback is long, TTFT can regress, and only within-process
repeats were collected per system. No threshold was relaxed after results.

If this candidate is revisited, the one useful next validation is independent
fresh-process, long-horizon replication of the c1 benefit including startup.
That is a recommendation only, not a new experiment started here. There is no
justification in this phase to add padding, fusion, draft batching or a new
GraphCache policy to rescue the adoption result.

## 10. Final Answers and Reproducibility

1. **Chosen scheme:** opt-in context-independent MLP replay, exact M4/8/16.
2. **Change from Phase5.0:** narrower boundary; observed512-request amortization
   replaced with explicitly weaker projected payback in this bounded phase.
3. **Reason:** existing module isolation, measured MLP opportunity, fewer state
   ownership changes; no change to the research question or frozen semantics.
4. **Experiments:** never-capture controls, one/all-layer prototypes, lifecycle
   controls, same-state GPU audits, and210 fresh-mode serving trials.
5. **Confirmed:** removable CPU launch work; exact same-shape replay; modest
   real serving benefit; capture changes later eager NCCL ordering.
6. **Not confirmed:** universal gap/skew reduction, strong concurrent gains,
   or metadata alone explaining the historical miss penalty.
7. **Eligible-miss root cause:** partially localized, not fully closed.
8. **Real serving benefit:** yes, strongest at c1; insufficient general adoption.
9. **Correctness:** passed the stated CPU/GPU/serving coverage, not a universal
   guarantee over untested shapes, devices or graph concurrency.
10. **Cost:**120 entries/rank,30MiB allocated/rank, ~11-12s startup,40 copies and
    graph boundaries per eligible target forward, unchanged collective count.
11. **Decision:** Keep Experimental, default eager unchanged.
12. **Next direction:** no automatic next phase; only the bounded independent
    replication described above merits consideration before promotion.

New files: five `benchmarks/serving/eagle3_phase51*.py` modules, one
`tests/test_eagle3_phase51.py`, the design doc, decision log and this report's
JSON/Markdown tables. Production changes in this phase:0. Existing dirty Git
state is preserved and must not be mistaken for Phase5.1 changes.

Raw root: `/root/autodl-tmp/eagle3-phase5.1-20260925/`, including complete
per-request/per-step records, source snapshots, startup, cleanup, traces and
`analysis-final.json`. Full executed invocations: [commands.md](commands.md).
Local assets and external source provenance: [references.json](references.json).
The complete CPU suite was rerun after GPU work:143 passed. Stop here and await
review; no next-phase implementation or Git commit.
