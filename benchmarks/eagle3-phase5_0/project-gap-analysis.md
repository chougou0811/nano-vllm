# Project Gap and Bottleneck Audit

Read after the external frontier map was written at 2026-09-24 13:29:45 UTC.
No production or historical-test change. HEAD is
`0ce2c1e054035766f2ee27027ffc309d58ad8c0c` plus the existing dirty/untracked
worktree; HEAD alone does not identify the implementation. File hashes are in
the source ledger. Evidence IDs below resolve there.

## Module-by-module assessment

Labels: A aligned with contemporary design; B correct/conservative; C implements
mature framework ideas; D behind current high-performance runtimes; E measured
bottleneck; F candidate boundary to replace; G retain; H distinctive project
engineering (not algorithmic novelty); I preserve negative result. Multiple
labels apply. D does not mean incorrect or obsolete for a teaching engine.

| Module / local symbol | Assessment | Evidence and action |
|---|---|---|
| Scheduler / `Scheduler.schedule`, `make_scheduler` | B,C,G,I | Original prefill-first remains default; static/V2 alternatives retained. C<=4 measurements do not identify selection as runtime bottleneck. Do not reopen tuning |
| Continuous batching / `ConcurrentLLMEngine.step` | A,B,C,G | Refill at step boundaries and batched verification are contemporary mechanisms, but not async execution or network serving |
| Request state / `SpeculativeRequestState`, `ProposalEntry` | A,B,G,H | Per-request IDs/generations/phases separate ownership from packed batch order; scoped, testable distributed reasoning |
| Paged KV / `BlockManager` | A,B,C,G | Per-rank pages and reference counts are mature mechanisms, not a newly invented PagedAttention |
| Prefix cache / `can_allocate`, speculative adapter | B,C,G | Ordinary hashed prefix reuse exists; speculative path intentionally disables it. No speculative prefix-sharing claim; no measured reuse pressure in isolated inputs |
| Speculative state / `DraftState` | A,B,G,H | Confirmed target-feature-conditioned cache persists; tentative autoregressive feedback is discarded. No cross-request cache sharing |
| EAGLE draft / `ReferenceDraft`, `DraftState.propose` | B,C,G | Existing dedicated checkpoint/reference architecture; rank0 serial per-request execution and explicit conditioning synchronization. Cost grows, but not largest measured component |
| Target verification / `batched_runtime._forward` | A,D,E,F | Ragged batched causal verification is sound; repeated eager layer launches and Python dispatch are behind current runners. Best bounded transfer boundary |
| Acceptance / `accept_greedy` | A,B,C,G | Greedy accepted-prefix semantics, not general stochastic rejection sampling. Explicit target logits remain authority |
| Transactional KV / `TransactionalBlockManager` | A,B,G,H | Reserve rows, own only newly allocated blocks, commit valid prefix, rollback suffix, keep inherited pages; safety value exceeds measured optimization opportunity |
| TP runtime / `ModelRunner.call`, row-parallel projections | B,C,G | Rank-matched model math, one-slot shared-memory RPC; local TP only. RPC packing alone is not slow enough to prioritize |
| NCCL / `RowParallelLinear.forward`, `_status` | B,E,G | 84 collectives/verification measured, but long residency includes peer waits; no proof of transport saturation. Keep order and safety reductions |
| Metadata / `prepare_prefill`, `verification_layout` | D,F | Rebuilds Python lists, descriptors, pinned tensors; layout also validated separately. Source-level duplication, not yet isolated as dominant measured time |
| Host/device sync / `_forward`, `_status`, `_zero` | B,D,G | `.item()`, `.tolist()`, rank status and zero checks are explicit safety barriers. Never remove them merely to make timings look better |
| Async execution | D,G | No overlapped request-control pipeline; next draft depends on accepted target results. Full overlap would change ownership and in-flight cancellation scope |
| CUDA Graph / `GraphCache.forward` | B,E,I | Fixed-shape replay is valid; exact-context dynamic cache loses serving. Keep historical path frozen and nondefault; miss regression remains unresolved |
| Attention / `Attention.forward` | A,B,C,G | FA2 paged varlen vs q=1 KV paths; backend change reopens numerical/mask contracts. Only 0.53-1.10 ms sampled service time |
| GEMM / QKV, MLP, LM head | B,E,G | Library skinny tensorop GEMM/GEMV. MLP ~69% non-NCCL GPU service. No counter evidence supporting a custom kernel or DRAM-saturation claim |
| Sampling / `Sampler.forward` | B,C,G | Ordinary compiled probabilistic sampler differs from speculative greedy argmax. No sampler change authorized or needed by primary |
| Memory management / cache allocation and cleanup | B,G,H | Resource ownership is tested; c4 peak 16/88 blocks in 4.2. Not a demonstrated KV-capacity bottleneck at tested horizons |
| Distributed-serving readiness | D | No routing, remote KV transport, multi-worker failure recovery, HTTP/SSE service or cluster admission. Two TP ranks are not a production distributed serving system |

Local symbol records: N1-N8. External comparisons: V1-V4, S1-S2, T1, F1,
L1-L2, M1-M2. This is a research engine with unusually explicit audit coverage,
not a feature-complete replacement for vLLM or SGLang.

## Reinterpret existing measurements, do not rewrite them

Sources are the frozen reports/JSON: [4.2](../eagle3-phase4_2/report.md),
[4.3](../eagle3-phase4_3/report.md), [4.4A](../eagle3-phase4_4a/report.md),
[4.4B](../eagle3-phase4_4b/report.md), [4.4C](../eagle3-phase4_4c/report.md).
Ledger P42-P44C identifies them. No new GPU measurements were run in Phase 5.0.

| Candidate bottleneck | Existing project evidence | External evidence | Missing evidence / conclusion |
|---|---|---|---|
| Serial draft | 4.2 fraction 10.96/17.25/24.32% at c1/2/4; target still 80.82/71.40/61.10% | DFlash/DSpark address drafting dependency (D1-D4, PDS) | New draft memory, accepted progress and cost on this target; secondary cost, not current first target |
| Target compute | 4.3 projection service ~15.3-15.5 ms; MLP ~11.6 ms | MPK, optimized GEMMs reduce operator work (MP1) | Hardware counters, fair library alternatives and critical-path attribution; nonzero compute floor remains after launch fixes |
| Host dispatch | ~640 target GPU launches; gaps and changing peer arrival | SGLang segmented replay, vLLM piecewise (S1,V3) | Exact savings from a small region, copy cost and number of boundaries; strongest primary hypothesis |
| Python metadata | Lists/descriptors/pinned tensors rebuilt each step (N2,N4) | MRV2 incremental state/gather (V1,V2) | No independent input-prep timing. Do not label all GPU idle as metadata overhead |
| Rank synchronization | Per-layer start skew 183-637 us; endpoint entry skew only 16.5-38.5 us | Overlap and stream-owned runtime designs (T1,V1) | Stack/counter attribution and timing perturbation; reduce dispatch skew, not remove correctness consensus |
| NCCL transport | 84 calls; residency changes sides, matched joint residency ~1.19-3.07 ms | DeepEP/overlap address different dependencies (DEP) | Pure transfer fraction unknown; SYS/SHM path, not an NVLink or EP benchmark |
| Attention | 0.53-1.10 ms, about 4.8% non-NCCL compute | FlashInfer plan/run, FA4 (F1,PFA4) | Long-context evidence beyond existing samples; poor immediate compute target, potentially useful execution contract later |
| MLP/GEMM | ~69% of non-NCCL compute, not 69% serving wall | MPK and autotuning (MP1,F2) | Occupancy/traffic counters unavailable; two-times faster MLP is not two-times faster serving |
| Graph eligible miss | 4.4C paired eligible-miss endpoint deltas +294.096/+394.189/+136.945 s over three runs per c | Modular execution dispatch separates regions/metadata (S1,V1) | Operator/stack root cause unresolved. Gloo, GC, clocks and altered launch timing are hypotheses, not findings |
| Cache residency | 512 requests yield natural reuse, but cap4 actual replay 3.39/3.01/1.02% | Frameworks commonly use bounded shape plans/segments (S1,S2,V3) | Selective admission and larger-cache serving gains untested; no evidence that frequency alone solves enabled-miss cost |
| Scheduler | 4.2 selection 0.01-0.02% wall; high-load fairness was a different stage | Async scheduler hides work, SuperInfer solves capacity pressure | Current small closed-loop regime is not Scheduler V2 overload; keep frozen |
| KV transactions | Reserve 0.02-0.04%; accept/commit/postprocess 0.81-1.03% | Modern frameworks implement ownership/cache managers | No need to trade state safety for tiny current cost |
| Memory movement | Low KV occupancy; no offload, isolated prefixes | LMCache/Mooncake/ATSInfer (L1,M1,PATS) | No repeated-prefix/offload bottleneck; GPU/CPU hierarchy addresses another regime |

4.2 establishes serving benefit of EAGLE versus ordinary batching: 2.332x /
2.313x / 1.755x, but token ITL tails worsen because commits are bursty. These
are not universal workload/hardware speedups. 4.3 rich profiler percentages
cannot be applied to unprofiled serving; its event span includes waits. 4.4A's
2.930x / 3.126x / 2.820x are warmed target-only, not end-to-end. 4.4C's
0.901x / 0.805x / 0.889x are actual graph-enabled serving comparisons in three
selected long-horizon cells, not a refutation of 4.2 or all graph strategies.

## Competing explanations and next diagnostic gate

The best-supported statement is **target execution wall time, including
eager dispatch and rank-arrival effects, is the largest measured opportunity**.
It is not established that metadata, NIC bandwidth or attention is dominant.
The 4.4C eligible-miss regression is a second, path-specific unresolved problem.

Before any future optimization benchmark, separate literal eager, disabled
graph-runner eager, shadow key calculation, agreement-only, and graph-enabled
no-capture eager on identical traces. Measure CPU ranges, CUDA API submission,
device intervals and both ranks; retain all samples. Never compare a new path
only against a deliberately slower enabled-miss baseline. The primary MVP must
bypass the old dynamic cache as a separate opt-in path, not silently fix it.

## Interview value: three defensible stories

1. **Transactional concurrent speculative serving.** Explain the pending token,
   accepted prefix, tentative pages, generation checks, per-request draft state,
   packed verification, TP rank agreement and failure cleanup. This is systems
   integration with tested invariants, not invention of EAGLE or paging.
2. **Numerical correctness engineering.** Show teacher-forced/operator controls,
   BF16 q=1/q>1 divergence and the accepted numerical contract. Historical
   strict parity failures remain failures; no arbitrary raw-logit threshold.
3. **Measurement-driven decisions and negative results.** Distinguish wall
   time, stream spans, kernel residency and peer waiting; explain why a valid
   fast fixed-shape graph lost in serving, and why Adaptive K was frozen.

Paging, TP linear partitioning, continuous batching, draft acceptance, ordinary
graphs and prefix hashing are mature ideas. Implementing/testing their
interaction is valuable even without algorithm novelty. The distinctive work
is ownership, diagnostics, reproducibility and falsifiable design, not feature
count. A successful bounded execution-stage transfer would add a clean
runtime-optimization story; a failed one remains useful only with a measured
bound and an explicit stop decision. Do not claim unimplemented network,
multi-node, stochastic, high-concurrency or production readiness.
