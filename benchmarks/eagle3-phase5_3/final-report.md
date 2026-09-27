# Phase 5.3 Final Report

## Decision: Keep Experimental

Cross-request draft-step batching has a demonstrated generation-stage benefit
and repeatable c2/c4 serving gains in this corpus. It is **not promoted to the
default/main path**: strict proposal reproducibility and the c1 tail qualification
are not fully established. This is not a rejection of the measured mechanism,
nor a claim that every adoption gate passed. The default remains serial draft;
`draft_batching=True` is explicitly experimental. Stop here for review; no Git
commit, secondary transfer or additional optimization is performed.

## Design and Scope

An independent executor serially catches up each request from its confirmed
state, packs right-padded copies of KV with valid masks and logical positions,
then batches heads and autoregressive feedback across selected requests. The
first proposal comes from the catch-up hidden state: fixed K3 needs three heads
and **two**, not three, feedback forwards. Rows with clipped K leave the active
batch without changing owner order. A request's second proposal still depends
on its first; this is not intra-request parallel prediction or P-EAGLE.

Only independently allocated confirmed KV is published to each request, after
all group work and owner-generation checks succeed. Feedback KV is temporary
and discarded. Existing target verification, greedy acceptance and transactions
are unchanged. One eligible request takes the original serial path. There is no
waiting for future arrivals, Scheduler change, catch-up batching or MLP replay.

Compared with the Phase5.2 sketch, the changed boundary follows actual Phase2
code rather than inventing a third feedback forward. Scratch is temporary, not
a new persistent allocator. The full explanation and pinned source provenance
are in [EAGLE3_PHASE5_3.md](../../docs/EAGLE3_PHASE5_3.md).

Existing production changes: `concurrent_engine.py` (opt-in flag) and
`coordinator.py` (generation binding and optional grouped proposal). New production
code: `draft_batch.py`. New CPU controls, benchmark runners and numerical audits
are isolated additions. [phase5.3-only.diff](phase5.3-only.diff) compares against
the saved pre-edit dirty tree, not Git HEAD, which also contains earlier work.
Historical tests and frozen Scheduler/TP/target Attention/BlockManager/ModelRunner/
sampling/DraftState/transaction implementations remain unchanged.

## Measured Benefit

| Metric | c1 | c2 | c4 |
|---|---:|---:|---:|
| Isolated generation wall reduction, packing included | Not batched | 45.16% | 70.17% |
| Main serving paired geomean speedup | 0.9973x | 1.0944x | 1.2216x |
| Main improving pairs | 28/50 | 47/50 | 50/50 |
| Full-serving generation reduction | approximately 0% | 34.75% | 63.88% |
| Total draft model calls, serial -> batch | 18270 -> 18270 | 18154 -> 13477 | 36404 -> 19956 |

There are 300 completed primary trials and 60 independent fresh-process trials,
five repeats/configuration, rotated order. Independent heldout-mixed/routing-table
checks confirm positive c2/c4 direction, not universal superiority. The targeted
c1 tail rerun is reported separately, not substituted for negative main samples.
Whole-serving speedup uses wall time, never the sum of request E2E latencies.

Isolated pack+mask+gather host submission costs are about 5.96% (B2) and 7.44%
(B4) of batched generation. They include copy submission but are not isolated GPU
copy-service measurements. Draft-only peak allocation increases about 9.13 MiB
(B2) and 83.79 MiB (B4). Full-serving peak allocation stays around 18.80-18.91
GiB on rank0, with constant post-trial allocation and zero leaked request state.
Maximum observed used target blocks are 5/10/20 at c1/2/4; initial packed draft
KV scratch reaches approximately 8.98/17.95 MiB at c2/c4. See raw JSON for both
ranks, all percentiles, forwards, acceptance counts and memory measurements.

## Adoption Gates

| Gate | Evidence and status |
|---|---|
| State/transaction correctness | CPU suite, real TP2 boundary controls, EOS/limits, injected failure cleanup and rank checks pass within coverage. All primary/fresh paired final outputs match; actual greedy acceptance is independently recomputed. |
| Numerical qualification | Strict proposal/path parity is not universal; failures remain recorded. Initial same-state BF16 discrepancy is localized to shape-dependent arithmetic with matching FP32 proposals. Extended audit and cross-process limitations are in correctness-report; not a blanket BF16 waiver. |
| Generation benefit | Clear c2/c4 reductions, exceeding the 10% reference gate including packing. |
| Serving benefit | c2/c4 exceed the 5% reference gate in aggregate; c4 improves all 50 main pairs and independent key cells. Some c2 cells are weaker. |
| c1 regression | Aggregate approximately neutral, but main long-long is 0.9416x. A post-hoc clean repeat is0.9857x (4/5 improving); not claimed universally within 3%. |
| Tails | c1 long-long/copy-pattern main P95 ITL regressions retained. Clean follow-up P95 ratios are0.9821/1.0018 and do not retroactively erase them. Full tail gate remains qualified. |
| Memory | No progressive allocated growth or completed-trial KV/state leak. Only c<=4 tested. |

## Attribution and Limitations

The isolated experiment proves that batching generation pays after packing.
It does not prove that every serving gain comes from the same mechanism. Target
verification wall also falls despite unchanged target code and nearly identical
forward counts. Rank cadence, host waits and GPU execution state are not isolated
causes here. Do not call the entire 22% c4 result a draft-kernel speedup.

All paired primary/fresh final outputs agree, but 37 primary pairs have different
proposal/target/acceptance signatures. Across main/fresh processes, 25/60 matching
full signatures differ, including frozen serial cases. No committed-output or
acceptance-semantics failure is hidden. These observations still limit stronger
claims of deterministic proposal trajectories across execution histories.

The three new workloads' intended high/medium/low acceptance labels did not
predict measured ordering. Their fixed prompts remain untouched; actual strata
are disclosed. The complete corpus includes high acceptance from retained
families. No checkpoint training revision is invented. No measurements establish
behavior beyond this model, hardware, K3 or c4. Warmup is not exhaustive.

## Next Direction, Only After Review

One narrow follow-up: a controlled **post-batching target-verification/rank-cadence
qualification** with identical inputs, shapes and execution history. It should
explain the target-wall change and reproducibility/tail limitations before any
default promotion. This is profiling and qualification, not permission to add
another optimization. Do not implement catch-up batching, graphs or the Phase5.2
secondary in this phase.

## Evidence Index

- [Decision log](decision-log.md): prespecified design and preserved failures.
- [Correctness report](correctness-report.md): strict invariants and numerical limits.
- [Draft batching report](draft-batching-report.md): isolated mechanism and overhead.
- [Serving report](serving-report.md): setup, scaling, acceptance and attribution.
- [Throughput cells](throughput-table.md), [latency percentiles](latency-table.md),
  [runtime/memory](runtime-table.md), [summary JSON](summary.json).
- [Commands](commands.md); external raw root includes all manifests, tensor
  diagnostics, traces, dirty-tree snapshot and source archive. No raw tensor or
  model weight is added to repository reports.
