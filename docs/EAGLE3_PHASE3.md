# Phase 3: Adaptive Speculative Length

Phase 2 checkpoint: `0ce2c1e`. DraftState, target runtime, acceptance, reference
draft model, Scheduler, TP and sampling implementations remain byte-identical.
Only the choice of proposal count and controller telemetry change in the session.

## Controller (Frozen Before GPU Pilot)

Each request creates a fresh controller; no workload labels, test results or
cross-request history are inputs. Candidate K is 1..6. Initial K is 3. Runtime
context is bucketed in 512-token bands, with independent estimates per band.

For each observed K maintain EWMA effective output tokens, proposal latency and
verification latency (alpha 0.25). Score is EWMA outputs divided by the sum of
the two EWMA costs. This uses accepted-prefix length through actual committed
progress, rather than maximizing the accepted/proposed fraction.

After initial prefix setup, try each unmeasured K once. Thereafter select the
highest measured rate only if it exceeds the incumbent by 10%. Every 12 learned
observations, probe one K in round-robin order. Probes do not change incumbent.
This bounded exploration/hysteresis is the whole controller; early rejection
does not trigger a separate override. Rejections reduce measured progress.

The last 16 iterations also record acceptance ratio, accepted/verification,
effective outputs/verification and consecutive zero-accept rejections. Per-K
costs, score, reason, context and choice are included in every decision record.

Initial full-prefix conditioning and terminal/output-clipped iterations remain
in raw reports and performance totals but do not train steady-state cost
estimates. Actual K is min(chosen K, remaining output tokens - 1); zero is only
the unchanged final target-only drain, not a controller candidate or hidden
low-margin fallback. Selection and observe never edit KV or token IDs.

## Semantics

Proposal, verification, greedy acceptance and draft/target commit/rollback are
the Phase 2 implementation. All ranks receive the identical proposal list via
the existing RPC, so they agree on query length. Only rank 0 selects K.

Measured time is inherently noisy. Repeated adaptive requests may select
different shapes. Determinism is therefore checked by replaying fixed K
decisions/proposals, not by claiming wall-clock-driven choices are identical.
The BF16 numerical contract is unchanged; new output disagreements are not
automatically accepted because they came from adaptive selection.

## Experiment Plan

Use a K=3 pilot over repetition, sky explanation, arithmetic, hexadecimal IDs,
code and translation at contexts 255/1024. Select three acceptance regimes
based on the measured pilot, without adjusting controller constants.
Run K=1,2,3,4,5,6 and adaptive with three repetitions per family/context.
Freeze/save the plan before trials, rotate policy order, retain every sample.
Each request owns fresh persistent state and a zeroed exclusive target lease.
Shape warmups are recorded separately; all modes use identical eager/audit flags.

Transaction correctness uses a scripted 1 -> 6 -> 1 sequence with fixed-proposal
replay against full reconstruction at 255/256/257/1088 context, plus rejection,
acceptance coverage, EOS, output limits and adaptive decision replay. Shadow
and state-copy runs are diagnostics, not performance samples.

No SLO coupling, serving concurrency, CUDA Graphs, kernels or scheduler edits.
Initial performance results cannot establish a universal adaptive advantage.

## Results

The pilot selected repetition (high), arithmetic (medium) and sky explanation
(low). All 126 planned GPU trials completed. All fixed-K repeats and all
cross-K output comparisons against K=3 agree in this dataset. Dynamic state
replay and the supplemented full-accept=6 test pass; 79 unit tests pass.

The adaptive controller does NOT beat the best measured fixed K in any of the
six workload/context cells. It has genuine exploitation reversals on arithmetic
and sky, not just intentional probes. See `benchmarks/eagle3-phase3/report.md`.
Do not tune constants on these results and re-label the same inputs held-out.
Adaptive remains experimental/opt-in, not a performance-recommended default.
