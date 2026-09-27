# Phase 5.1 Decision Log

## D0: Preregister before GPU results, 2026-09-25

- Decision: start with actual GraphCache never-capture controls and an isolated
  MLP-only replay prototype. Keep every production and historical test file intact.
- Evidence: Qwen3MLP.forward is already a context-independent module; the original
  proposed o_proj/norm/MLP spans two modules and residual ownership. Phase4.3
  measured MLP as the largest non-NCCL compute component. GraphCache permits
  capture_policy=never, allowing a faithful enabled-miss control without captures.
- Alternative: move the broader Qwen decoder boundary immediately. Rejected
  because it changes more code before establishing a bounded opportunity.
- Impact: same research question (context-independent replay), smaller region,
  one captured collective/layer rather than two. Literal eager stays the direct
  frozen dispatch. No historical artifact or numerical contract changes.

## Preregistered gates

1. Correctness: byte-identical same-input/same-shape target logits, features and
   KV; rank/state agreement; end-to-end proposals/commits/outputs; repeated replay,
   eager fallback, boundaries, EOS/output clipping and intentional failure cleanup.
   Any unexplained new numerical mismatch stops serving adoption.
2. Prototype: diagnose c1/2/4 exact q4 with paired repeated fixed-prefix forwards,
   preserving every sample. Advance to serving only with consistent all-layer
   target improvement, not a single lucky microbenchmark. Start one layer, then
   all layers; no new attention/TP algorithm, no dynamic capture or padding.
3. Serving: all six historical families plus held-out `heldout-mixed` defined
   before results below; c1/2/4, five paired repeats, rotated eager/optimized order.
   Same prompts/proposals policy and output counts per pair. Raw per-step/request
   records retained; profiler runs separate from throughput. Fresh processes per c.
4. Adopt only if correctness passes, geometric-mean paired serving speedup >=1.05
   in at least two concurrency groups, at least four of five repeat aggregates
   improve in each qualifying group, and held-out median improves. Report all
   tails/regressions; >10% repeat-median P95 ITL regression prevents general adoption.
   Require >=1GiB physical headroom/rank and projected capture payback <=512
   requests, explicitly labeling projection if not observed over that horizon.
5. Stop on no consistent target benefit, memory/correctness failure or no serving
   benefit. No padding, new cache policy, fusion, attention/draft/scheduler changes
   to rescue results. Missing gates mean Keep Experimental, not Adopt.

Held-out fixed definition: prompts of 127/383/895/511 tokens, output limits
48/96/40/80, cycling text about archive checksums, versioned records and recovery.
Use a distinct first token per request; identical per-pair inputs; no tuning
after outcomes. This is deliberately not the Phase3.1 prompt set.

No performance number in this log is an observed result at preregistration.

## D1: After never-capture diagnostic and initial prototype

- Decision: add a capture/no-replay/release lifecycle control. Do not disable
  NCCL graph mixing in actual replay/serving. Require fresh never-captured eager
  serving controls, not only same-process eager after capture.
- Evidence: never-capture disabled/shadow controls are near eager. Source
  NCCL v2.27.3-1 `src/misc/strongstream.cc` lines163-168 and263-264 uses persistent
  `everCaptured` to add event wait/record on future noncaptured calls. Two-verify
  traces increase cudaEventRecord510->680 and cudaStreamWaitEvent340->510 after
  one-layer capture. This identifies a testable mechanism, not yet its wall cost.
- Alternative: blame Gloo/metadata for the full historical penalty. Rejected:
  fresh never-capture controls do not reproduce the large historical regression.
- Impact: source-based causal diagnostic; history unchanged. Diagnostic-only
  `NCCL_GRAPH_MIXING_SUPPORT=0` is allowed only with zero graph replays, fully
  synchronized capture and eager work. Real segmented replay keeps default1:
  disabling mixing forbids outstanding graph/noncaptured calls even if ordered
  on one stream according to NVIDIA's contract. This is not a proposed fix.
- Source: https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/misc/strongstream.cc
  and https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support

## D2: Serving comparison and scope accounting

- Decision: use six independent serving processes (literal/optimized at c1/2/4),
  five within-process repeats and rotated workload order; reverse system-process
  order at c2. No paired in-process eager result is used for adoption because
  an eager mode after capture is not an untouched communicator.
- Evidence: correctness completed with16 generation/limit cases plus EOS and
  exception cleanup;64 audit points each execute three same-state replays with
  exact features/KV and rank0 logits. All-layer fixed-prefix results justify a
  serving gate; one-layer total-target effects alone were too small/variable.
- Alternative: only compare modes in one graph-initialized engine. Rejected
  because `everCaptured` persists beyond registry release. Five fresh processes
  per cell would be stronger independence, but is outside this bounded first
  characterization. Process/time ordering remains an explicit limitation.
- Impact: max_model_len2048 in both new modes supports the1024-context
  correctness case; Phase4.2 used1024. Actual historical workload inputs/limits
  are preserved, but absolute old throughput numbers are not the denominator.
  Fixed target/draft revisions, precision, TP, K and policy are unchanged.
- Startup: Phase5.0 proposed observed512-request amortization. Phase5.1
  preregisters projected512-request payback instead for a bounded experiment;
  it must be labeled projection and cannot establish long-horizon stability.
  This changes evidence strength, not the underlying runtime research question.

## D3: Trace interpretation safeguard

- Decision: retain all raw profiler spans, labeling alternating verify/rollback
  spans explicitly in offline analysis.
- Evidence: prototype marker name was reused for rollback, which has one status
  collective; averaging all four scopes would halve the verification launch and
  collective counts. Driver source records exactly two verify/rollback pairs.
- Alternative: count all scopes as verification, or silently remove small ones.
  Rejected. Label by the deterministic driver sequence, preserve rollback
  records, and assert the scope count. No timing outlier is removed.
- Impact: measurement-label correction only, not runtime or historical edits.

## D4: Completed serving evidence and final decision

- Decision: Keep Experimental. Do not promote to production or change defaults.
- Evidence: all210 trials complete;105 paired outputs/speculation statistics
  agree exactly. Geometric-mean speedups c1/2/4 are1.0703/1.0190/1.0253.
  Only one group clears the preregistered1.05 gate; two were required.
  Projected startup payback is170/910/834 requests, not a measured long horizon.
  No >10% repeat-median P95 ITL regression, but TTFT and some E2E tails regress.
  Full143-test CPU suite and GPU audits pass;339 protected files unchanged.
- Alternative: adopt based on c1 or declare the all-layer microbenchmark a
  success. Rejected because this would weaken the pre-result concurrent-serving
  gate and ignore startup/evidence limitations. Rejecting the entire direction
  would also ignore a consistent tested c1 benefit.
- Impact: retain opt-in research code and raw data only. Historical results,
  numerical contract and production semantics stay frozen. Eligible-miss
  ordering mechanism is supported, but the full historical wall penalty remains
  partly unexplained. No padding/fusion/cache-policy follow-on work is started.
