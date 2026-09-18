# Scheduler V2: Progress Balance

## Preservation and Scope

Before editing, the complete dirty V1 worktree including .git and untracked files
was archived in `/root/autodl-tmp/benchmarks/scheduler-v2/pre-v2/worktree.tar.gz`.
SHA256: `7dc2025e75310b2c6e46ba90a6e3a7c1fb423ad03fa4e82770f288f6f33f776e`.
V1 artifacts are read-only inputs to the analysis. No commit, reset, clean or
default-policy switch is performed. `original` stays default; `slo-aware` remains
V1 and `slo-v1` is its alias. `slo-v2` is opt-in. TP, Attention, BlockManager,
ModelRunner, Sequence and sampling implementation files are not changed.

## V1 Failure Mechanism

V1 asks whether prefill fits the earliest decode slack after reserving decode
time. When no budget fits, even an expired TTFT only wins the minimum 128 budget.
The class-progress override also grants only 128. Thus waiting age can select
prefill but cannot increase its progress quantum. Pressure normalization by
2000 ms TTFT versus 100 ms token deadline further favors small decode overruns.

The single prefill EWMA divides complete step time by tokens. Fixed launch/host/
communication overhead is repeatedly charged as a per-token slope at small
chunks, then extrapolated to large chunks. A long prompt estimated as one large
prefill also omits intervening decode turns. Batch/context effects are mixed.
Under infeasible aggregate decode load, retrying the earliest deadline does not
make it feasible; it suppresses prefill and grows its queue. These are hypotheses
grounded in source and decision traces, not isolated causal proof without ablations.

## V2 Policy

V2 derives from the existing opt-in policy, sharing reservation, allocation,
commit, release, FIFO waiting and round-robin decode with static. It does not
construct mixed prefill/decode kernel batches. Only arbitration and chunk choice
are new.

When both classes are feasible, one prefill turn is followed by one decode turn.
An expired decode deadline does not cancel the prefill turn. This separates class
progress from deadline feasibility. A memory-infeasible prefill still falls back
to decode, draining residents under the unchanged conservative reservation rule.

For each candidate chunk (doubling from minimum to maximum), predict the head's
completion cost as `ceil(remaining/chunk) * (prefill_cost + decode_turn_cost)`.
Choose the smallest chunk that can finish within `TTFT_target - waiting_age`,
or maximum when none can. Remaining work includes actual partial-prefill progress.
The final chunk may have fewer actual tokens than the budget.

The new EWMA is keyed by phase, power-of-two total scheduled tokens, batch size,
and mean effective context length. Missing shapes use a nearest observed bucket
with a fixed-cost component and weak context/batch scaling; this is a predictor,
not an asserted hardware model. Cold predictions are configurable through the
existing initial-step seed. Outlier observations are retained and update EWMA.

Overload is detected when estimated decode round plus minimum prefill exceeds
the token interval target, or draining waiting-token backlog at the minimum chunk
exceeds the oldest TTFT slack. Enter immediately, exit after three healthy checks.
Under overload, use at least an efficiency chunk (default 512 desired tokens),
still alternating classes. A soft single-prefill cost cap of `max(TPOT,2*decode)`
can lower the budget, but cannot remove its turn. Defaults: minimum 256, maximum
1024. Tuning explores minimum 256 versus 512 with all other parameters frozen.

This is overload fallback, not overload admission rejection, SLO guarantee, or an
optimal controller. It never drops late requests. Choosing a larger progress
quantum can increase ITL; the experiment measures that trade-off explicitly.

## Progress Bounds

Under contention and feasible memory, each class receives a turn in at most two
engine steps. Decode rotation serves a resident within at most two times the
number of decode batches covering resident requests; new residents append behind
existing ones. FIFO head prefill advances at least the configured minimum budget
or its final remainder each prefill turn. Bounds on wall-clock time additionally
require bounded engine-step duration. Reservation backpressure excludes a waiting
request from the feasible set until finite-output residents drain. No bounded
arrival-to-finish guarantee is possible for arbitrary unbounded offered overload.

## Numerical Acceptance

The rules are saved before capture in `acceptance-rules.json`. Raw max absolute
BF16 error is reported but not a standalone pass/fail criterion.

1. Original repeat must be bit-identical for the same teacher-forced inputs.
2. Every expected output position is captured once; four forced tokens really
   commit; all values are finite; all KV references are released. CPU tests also
   exercise cache reuse, block boundaries, reservation and finite draining.
3. Independent HF eager reference: BF16 for 14B; FP32 for the small model with
   TF32 disabled. Small-model HF FP32 full and 256-chunk execution must agree
   within max absolute 1e-3 and total variation 1e-4.
4. Original full/chunk256/chunk512 comparisons to HF calibrate each position's
   total variation and centered-logit relative L2. Candidate limits are twice
   the largest control error, floored at 0.01 and capped at 0.10. A control
   exceeding the hard cap invalidates acceptance rather than widening the cap.
5. Candidate top-1 must be among HF's top ten and within a tie band determined by
   control errors at those candidates and two BF16 representable steps. Exact
   top-1 counts and margins remain reported. Candidates never set their limits.

These are finite diagnostic acceptance criteria, not a theorem of equivalence,
quality evaluation, or an excuse to discard mismatches. CPU negative controls
reject NaN, a strongly incorrect distribution and original-repeat perturbation.
The normal sampler is restored after test-only teacher forcing.

### Metric Revision Before Independent Holdout

The first equal-weight centered-L2 gate failed even original controls on very
confident inputs: large low-probability-tail differences while TV was around
1e-6. Its failed `acceptance.json` is retained, not relabeled as passed.
Probability rules revision 2 replaces the *gating metric* with
`sqrt(sum(p_ref * (delta - sum(p_ref*delta))**2))`, an offset-invariant residual
in nats weighted by the independent reference probabilities. Raw errors and
unweighted relative-L2 remain reported. The new weighted RMS floor is 0.05,
hard cap 0.30, same 2x original-control envelope. TV/tie/structural/high-precision
rules are unchanged. A new storage-audit input corpus, different from the inputs
used to discover this issue, is captured only after these rules are declared.
The frozen tuning candidate is also checked against the independent reference
before held-out timing. No gate is silently bypassed to run the experiment.

The first small-model BF16 comparison had three TV failures despite a passing
FP32 full/chunk control. Original full/chunk controls mostly shared decode
batch=4 and did not cover the candidate's smaller batches. The calibration is
therefore extended with unchanged-original batch=1/2/3 executions (`--shape-controls`)
on a fresh small-model corpus. Multipliers, floors and hard caps remain unchanged;
the earlier insufficient-control failure is retained. This is additional measured
control coverage, not setting thresholds from candidate errors.

Per-policy outcomes remain separate. On the new small corpus static and V2 pass;
V1 has one calibrated-envelope failure with exact HF top-1 and errors below hard
caps. The experiment requires strict static/V2 acceptance, original parity,
finite values, coverage and the FP32 control. V1 is a warning-marked historical
comparator only when its failed rows still have exact reference top-1, valid
controls and all hard caps satisfied. Its aggregate `passed=false` is retained;
no thresholds are changed to certify it. Any other failure blocks timing.

## Experimental Protocol

The model is Qwen3-14B revision 40c069824f4251a91eefaf281ebe4c544efd3e18,
TP=2, BF16 eager, model length/token budget 2048, batch cap 4, memory 0.85.
Scoring targets stay TTFT=2000 ms, request TPOT=100 ms. All arrivals are open-loop,
with step-boundary submission lag separately reported from engine waiting.

- Tuning: six (128 input,64 output)/(1536 input,16 output) pairs, 350/100 ms
  arrivals, two repeats per minimum-chunk candidate. Choose highest pooled goodput;
  within 5%, choose smaller minimum. Write frozen-config.json before held-out.
- Held-out: three sets of (64,96)/(384,32)/(1024,48)/(1792,8), seeded shuffled order,
  different input tokens, arrivals 800/350/200/100/0 ms, three repeats, four policies.
  Order rotates per repeat. Burst (0 ms) is not described as a finite offered rate.
- Ablation: paired static/V2 references plus V2 no-overload, V2 scalar predictor,
  static512, static non-rotating decode, static without reservation. Two rates,
  two repeats. Reservation removal is test-only and gated by aggregate worst-case
  KV demand fitting the entire workload; it cannot be used as an unsafe runtime mode.
- Sustained finite backlog: 48 held-out requests at 150 ms arrivals for all four
  policies. One repeat is diagnostic, not a statistical long-run stability claim.

Every trial resets scheduler/prefix metadata while retaining the GPU allocation.
Warmup tokens have isolated first tokens and are excluded from measurement. Exact
paired inputs, raw request/token/step data, source snapshots, revision/checksums,
environment, tuning decisions and failures are saved in a new V2 directory.
No slow samples are removed. Baseline/static/V1/V2 all run in this experiment;
old V1 timings are not silently substituted as contemporaneous controls.
