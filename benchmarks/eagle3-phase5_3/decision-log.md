# Phase 5.3 Decision Log

## D0: Before implementation and timing

Freeze the current dirty tree in an external source archive; no Git commit.
Primary comparison is serial vs batched draft, target eager, fixed K3.
No MLP replay, catch-up batching, new checkpoints or scheduler changes.

Source fact: DraftState.propose uses the conditioning result for the first head
prediction and performs only K-1 feedback forwards. Batch heads and feedback
forwards, not three additional model forwards. Keep all conditioning serial.
Use right-padded transient KV, explicit logical positions and valid masks.
Publish only each request's independent confirmed KV; never retain feedback KV.

First measure correctness and generation wall including scratch packing/masks.
Stop without serving performance runs if the c2/c4 stage gate is clearly unmet
or if new numerical/state disagreement remains unexplained. Do not rescue the
experiment with a second optimization. Timing order rotates, outliers retained.

If isolated gate passes: retain the seven Phase5.1 families, plus fixed new
copy-pattern, prose-completion and structured-calculation prompts as intended
high/medium/low-acceptance strata. Their actual strata must be measured, not
asserted. c1/2/4, five repeats, fresh-process checks, no post-result prompt tuning.

## D1: Isolated gate and first integration attempt

Nine real target-state groups (B2/B3/B4, three successive iterations) retain
identical proposals and exact confirmed KV against the frozen serial oracle.
Five paired generation repeats/cell show about45% B2 and71% B4 median reduction
including temporary padding/masks. FP32 shape controls preserve proposal IDs.
Proceed to opt-in serving integration; retain default/c1 serial execution.

The first correctness run failed in new benchmark instrumentation: a late-bound
closure referred to the RPC wrapper instead of executor.propose, attempting to
serialize draft inputs into shared memory. Fix by binding wrapper originals in
default arguments. Production algorithms were not changed for this failure.
Retain correctness/manifest.json and its error log; rerun as correctness-r2.

## D2: First real proposal disagreement

Serving-main stopped at short-short/c2/repeat0. Final outputs for all four
requests agree, but one iteration's second proposal changes87856 ->25351,
changing accepted length3 ->1. The strict signature failure is retained.
Same-state replay reproduces it. Confirmed hidden/KV are identical at entry;
serial native vs identical padding is exact in BF16; first differences occur at
LM-head and feedback Q/K/V projections. At the affected head serial margin is
0.0625. FP32 both shapes select87856, with margin0.0229969. Teacher-forced operator
diagnostics prevent attributing later different-token states to the first cause.

Continue with an explicit benchmark-only record-draft-variation flag. Strict
signature remains false, acceptance is independently checked against the frozen
greedy function, and every differing cell remains pending diagnostics. Final
output disagreement still stops. No sampler, tolerance, tie-break or production
fallback was changed. Do not mark the whole correctness matrix passed merely
because this first numerical cause is explained.

The new offline acceptance checker initially compared a live tuple against a
list. Serving-main-r2 retained its two completed c1 trials and stopped on this
Python container-type mismatch. Normalize only the checker comparison and add
a tuple/list unit regression; no production acceptance change. Resume in a new
directory, retaining the failed process and its measurements.

## D3: Disconnect recovery and fresh-process checks

Main-r3 completed all300 trials/150 pairs: final outputs and independently
recomputed greedy acceptance agree;37 pairs retain strict-signature variation.
The connection interruption terminated serving-fresh during initialization,
before any measured trial. No running inference process survived the disconnect.
Its orphan `/dev/shm/nanovllm` prevented fresh-r2 initialization. Terminate only
that failed process group, confirm no remaining inference workers, unlink the
one orphan shared-memory segment, and retry as fresh-r3. Preserve both failures;
do not patch the frozen ModelRunner initialization path in this phase.

## D4: Full matrix, independent checks and bounded qualification

Main-r3 completes300 measured trials and fresh-r3 completes60. No paired final
output disagreement; actual greedy acceptance is checked independently. Main
paired geomean speedups are0.9973/1.0944/1.2216 at c1/2/4. Do not promote on
aggregate throughput alone: c1 long-long/copy-pattern show tail regressions,
and target verification wall changes without target code changes. Define a
post-hoc independent c1 tail recheck, retaining original negative observations.
No production tuning is performed after seeing these results.

Replay all37 differing main cells with an identical-state serial oracle and
full KV audit. Completed3023 groups capture87 local differences in32 cells.
Final outputs, actual acceptance and cleanup match; five originally different
cells do not locally reproduce. Across main/fresh,25/60 full signatures differ,
including frozen serial; all final outputs match. Do not infer cross-history
determinism from output equality. Apply teacher-forced native/padded/batched and
FP32 controls to every captured local difference, but preserve any residual
diagnostic limitation. Final adoption class: Keep Experimental, default false.
