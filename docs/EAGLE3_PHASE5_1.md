# Phase 5.1: Bounded Target Runtime Experiment

Date: 2026-09-25. The final decision and completed measurements are recorded in
[report.md](../benchmarks/eagle3-phase5_1/report.md) and
[summary.json](../benchmarks/eagle3-phase5_1/summary.json).
The pre-result gates and subsequent evidence-driven adjustments are in
[decision-log.md](../benchmarks/eagle3-phase5_1/decision-log.md).

Status: **Keep Experimental**. All210 serving trials complete; geometric-mean
speedup c1/2/4 is1.0703/1.0190/1.0253. Only c1 meets the preregistered5% gate,
so no production promotion. Complete CPU suite143 passed; same-shape GPU and
105 paired serving comparisons pass. Capture amortization remains projected,
not long-horizon observed. Work stops here pending review.

## Scope and implementation

All production inference files and historical tests remain unchanged. An
explicit benchmark-only `Phase51Runner` uses the existing verification callback
boundary. Default mode calls the original `runtime.dispatch` with no callback.
No Scheduler, draft, acceptance, KV transaction, TP arithmetic, attention,
sampling or model-weight change. No dependency upgrade, model download or commit.

The Phase5.0 o_proj/residual/norm/MLP proposal was narrowed to the existing
`Qwen3MLP.forward` boundary: gate/up matmul, existing compiled SiLU/multiply,
down projection and its existing NCCL all_reduce. This is an execution change,
not fusion or a new MLP kernel. QKV, attention, o_proj, norms, feature hooks,
LM head and all state control remain eager.

Why narrow it: MLP is already a self-contained context-independent module.
The broader proposal would move code across attention and decoder modules and
introduce residual ownership changes before proving useful launch savings.
The smaller scope leaves the research question unchanged and retains a direct
frozen-eager denominator.

## Registry and lifetime

- Fixed K3; eligible verification requires uniform q4 and batch1/2/4. Other
  shapes, including mixed clipped proposals and batch3, run unchanged eager.
- Each layer owns separate exact-M4/8/16 entries. On Qwen3-14B this is120 entries
  per rank. Weights are the existing immutable TP shards, never recopied.
- Each entry owns one input and graph output/storage. Static input receives a
  D2D copy; replay returns output consumed by the next eager layer. No KV page,
  position, context length or block table is captured.
- One target verification is in flight. Graph inputs are not shared between
  requests as persistent state; target feature hooks create the existing owned
  feature tensors. Mutable static outputs are not installed as request state.
- All ranks capture the same layer/shape order before measurement. The existing
  verify payload determines eligibility identically, without new per-layer
  agreement. Existing rank status, finite checks and transaction rules remain.
- Capture uses a shared capture stream, distinct storage/pools per entry. Eager
  and replay execution remain on the existing stream. NCCL mixing support stays
  at its default for every real replay and serving run.
- The temporary MLP bindings are restored on normal return or exception. Missing
  registry entries are detected before partial binding. Entry release waits for
  device completion; graphs are reset before owned storage is released.
- No dynamic capture, LRU, eviction policy, context bucket, padding, new attention
  backend or low-margin fallback exists in this prototype.

## Diagnostic separation

Never-capture modes isolate literal eager, disabled GraphCache wrapper, shadow
key computation, agreement-only and actual GraphCache never-capture misses.
Lifecycle runs compare before capture, one live but never-replayed full-target
graph, and after release. They preserve every sample and distinguish context/
shape, run order and the fact that this is not the historical long service run.

NCCL2.27.3 source contains a persistent `everCaptured` condition that adds event
ordering to later eager operations. Its event counts can survive graph release.
A separate diagnostic disables mixing only when **no graph is ever replayed**;
it is not a serving configuration or an optimization proposal. Turning mixing
off during actual segmented graph/eager execution would violate NVIDIA's
documented outstanding-operation contract and is not done.

Sources:
[versioned NCCL implementation](https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/misc/strongstream.cc),
[NCCL mixing contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support).
Source evidence explains an event-ordering mechanism, not automatically the
full wall-time magnitude of Phase4.4C. Final attribution limits stay explicit.

## Correctness

Same-input/same-shape audit snapshots committed KV, runs frozen eager, restores
the same state and executes three repeated region replays. Compare features and
KV bytes on both ranks and logits on rank0, with finite/rank-state checks.
Compare full generation proposals, verification token IDs, accepted lengths,
committed outputs, cursor movement and termination. Cover255/256/257/1024,
concurrency1/2/4, short output clipping, injected EOS and an intentional
post-verification acceptance exception. The original CPU suite is unmodified;
new focused tests check eligibility and binding restoration.

This is stricter than borrowing the BF16 cross-shape waiver: unexplained
same-shape disagreement is a failure. No sampler/tie-break/threshold changes.
Cleanup checks waiting/running/used blocks/transactions/host requests/drafts and
rank target states; graph registry cleanup is separate from request cleanup.

## Measurement and reproducibility

Raw root: `/root/autodl-tmp/eagle3-phase5.1-20260925/`. Each process records exact
command, model revisions, environment, versions, source snapshot and hashes,
GPU topology, raw requests/steps, startup and cleanup. Historical results are
compared as evidence, not overwritten or used as the fresh serving denominator.

Mechanism traces are separate from throughput. The prototype's trace marker
names both verify and rollback `phase51.verify`; the offline parser retains
and labels alternating spans, because the driver records two explicit
verify/rollback pairs. Rollback spans must not be averaged into verification.
CPU API counts, device interval unions, copies and collective residence are
different measurements. NCCL residence is not pure transport time. Small
profiled windows do not establish universal gap/skew improvements.

Serving uses six frozen Phase4.2 workload families plus the preregistered
held-out mixed family, c1/2/4, five repeats per system/cell. Literal and optimized
systems run in separate processes so literal has never captured a graph.
Mode-process order is reversed at c2; workload order rotates/reverses per repeat.
This is not five independent processes per cell; within-process repeats and
between-process ordering remain limitations, reported rather than hidden.

Warmup is excluded from steady-state latency; extra capture cost is reported
separately and amortization projections are labeled. Throughput denominator is
whole serving wall time, not sum of request E2E. Preserve all outliers, burst
token timestamps and eager fallbacks. Compare paired outputs/acceptance exactly.

Implementation files are only `benchmarks/serving/eagle3_phase51*.py` and new
`tests/test_eagle3_phase51.py`. No automatic promotion into the default engine.
Stop after the reported decision and await review.
