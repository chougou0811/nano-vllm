# Phase 3.1 Adaptive K Finalization: Preregistered Design

Final status: correctness accepted within the existing numerical contract;
performance superiority not demonstrated. Freeze as a negative-result research
path, with no default change or further tuning on these workloads. See
`benchmarks/eagle3-phase3.1/report.md` for all 378 held-out trials and limitations.

Phase 2 state semantics and the existing Phase 3 controller remain frozen.
Phase 3's dirty source is archived under the external Phase 3.1 artifact root;
no new Git commit is authorized or made by this stage.

## One Model, No Online Probes

Learn six conditional acceptance probabilities from censored prefixes. Accept a
of K gives successes at depths 1..a and, if a<K, one failure at depth a+1.
Unverified suffix positions are not failures. Context buckets are 512 tokens.
Independent calibration supplies Beta priors with effective count 16 per depth,
and cost priors with count 16. Request-local cumulative posterior updates replace
winner selection from isolated per-K reward samples. Priors never learn from
held-out requests or previous held-out repeats.

For K, expected outputs are 1 + sum over depths j<=K of product(p_1..p_j).
Subtract measured draft conditioning from current proposal cost. Model next
catch-up cost by its row count 1+a, and add its expectation under the predicted
accepted-length distribution. The score is expected outputs divided by current
proposal work + verification + expected next catch-up. This assigns lagged
catch-up cost to the action that generated the new rows, not the next action.
Initial full-prefix setup remains in E2E but not incremental cost learning.

Choose once at request start, then only every 64 generated output tokens.
There is no rate threshold, hysteresis, per-request arm calibration or active
probe. A 64-token request holds one K; a 512-token request has at most eight
decision epochs. Fixed K=1..6 all remain baselines. Parameters are frozen before
calibration/held-out measurements, not tuned from their performance results.

## Split and Experiment

Independent calibration prompts: structured status logs, a recycling explanation,
and random-looking numeric identifiers. Contexts 255/768/1024, fixed K=1..6,
96 outputs. These observations build one pooled, label-free prior by context.
Missing catch-up row shapes use the nearest observed row shape, explicitly
recorded. No model weights, kernel or Scheduler changes are made.

Held-out prompts: a new orchard prose continuation, a detailed fermentation
explanation, and UUID generation. They do not reuse Phase 3's scientific notes,
17x23 arithmetic or sky explanation, and do not appear in prior construction.
The intended high/medium/low labels must be verified from actual acceptance;
if the long-horizon regimes shift, report that rather than relabeling performance.

Held-out grid: 3 families x context 255/1024 x output 64/256/512 x 7 policies
(fixed 1..6 and prior adaptive) x 3 repeats = 378 requests / 104,832 output tokens.
Save the plan and prior hash before trials. Rotate policy order, keep every
sample, and record warmup separately. Requests use independent state and priors.
Timing runs use audit=False uniformly; dedicated correctness uses audit=True.
Target finite/rank transaction checks still execute in both modes. No comparison
of absolute performance to Phase 3's audit=True timings is claimed.

Report all latency/progress metrics, per-K posterior/prior estimates, chosen K,
switch/run returns, offline calibration time and zero online probe overhead.
No assertion of universal BF16 cross-shape token identity is introduced. New
disagreements remain subject to diagnostics, never arbitrary thresholds.

## Stopping Rule

If held-out gains are not stable against strong fixed K, freeze this as a
negative-result research path. Do not retune on these prompts, add thresholds,
or start SLO coupling. Concurrent serving/final integration can be a separately
authorized next task; it is not implemented in this finalization stage.

## Opt-In API

Load a schema-1 prior produced by the independent calibration stage. With an
already initialized eager engine:

```python
import json

with open("benchmarks/eagle3-phase3.1/prior.json") as handle:
    prior = json.load(handle)
result = engine.generate_eagle3(
    prompt,
    draft_path="/root/autodl-tmp/models/Qwen3-14B_eagle3",
    reference_path="/root/autodl-tmp/references/eagle-pinned",
    speculative_length=3,
    draft_state_mode="persistent",
    adaptive_k=True,
    adaptive_prior=prior,
    max_tokens=256,
)
```

`speculative_length=3` enables the existing draft path; the controller then
chooses requested K in 1..6. Without `adaptive_prior`, the original Phase 3
controller remains available. Fixed-K callers and defaults are unchanged.
The prior is deep-copied per request, not updated as a shared mutable cache.
Runtime latency observations may change choices between repetitions; replaying
the same decisions/proposals is the deterministic correctness control.
