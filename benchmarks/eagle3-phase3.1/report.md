# EAGLE-3 Phase 3.1: Adaptive K Finalization

## Decision

**Freeze Adaptive K as a negative-result research path.** Correctness and the
preregistered 378-request experiment pass. The low-exploration design bounds
switching, but does not demonstrate a stable advantage over strong fixed K.
Do not tune again on these held-out inputs, promote adaptive to the default,
or start SLO-aware K. Keep fixed-K paths; concurrent speculative serving/final
integration is the recommended separately scoped next task, not implemented here.
Controller and calibration prior stayed frozen throughout all measurements.

## Method

The new opt-in `PriorAdaptiveK` consumes an independently calibrated prior in
512-token context buckets. It learns conditional acceptance probabilities from
censored accepted prefixes: an accepted prefix contributes successes, its first
rejection contributes one failure, and the unverified suffix contributes nothing.
Beta prior strength and cost prior count are both 16. Each request starts from
the same immutable prior; held-out requests never train later requests.

For each K in 1..6 the estimated progress is
`1 + sum_j<=K product_d<=j p_d`. Its cost is current proposal work, verification,
and the expected *next* catch-up cost for `1 + accepted` rows. Current conditioning
is subtracted from proposal time and learned separately by row count. Thus work
caused by the previous accepted prefix is not charged to the current K arm.
Full-prefix initialization remains in measured latency but does not train the
incremental cost model.

The controller chooses at request start and every 64 committed output tokens.
There is no per-request K=1..6 calibration, active probe, threshold, or hysteresis.
Short 64-token requests hold one requested K. A 512-token request has at most
eight decision epochs. Requested K and output-budget-clipped actual K are
reported separately; the existing K=0 terminal drain is not a candidate.

This is a factorized predictive model, not measured rewards for every possible
action at the current state. Unchosen-K costs remain prior estimates. The model
assumes conditional acceptance generalizes across K within a context bucket;
changing content, censored deep positions, and stale costs can violate that
approximation. Epoch limits bound switching frequency but do not prove optimality
or eliminate slower A->B->A returns. Crossing context buckets can change the prior
and posterior used for decisions. The score is not horizon-optimal near termination.
Commit RPC, controller and Python bookkeeping are included in measured E2E but
not explicitly modeled by the proposal/verification/catch-up score.

## Frozen Components

Phase 2 persistent state, greedy acceptance, draft/target commit and rollback,
Scheduler, BlockManager, ModelRunner, TP, attention and sampling are unchanged.
The Phase 3 controller is also retained unchanged. New selection occurs before
proposal and observation after the existing commit. Original Scheduler remains
the default. No concurrent requests, SLO coupling, CUDA Graph or kernel changes.

Changed production surface: new `nanovllm/speculative/prior_controller.py` and
optional `adaptive_prior` forwarding/telemetry in `speculative/session.py` and
`engine/llm_engine.py`. New tests are in `tests/test_eagle3_prior.py`; experiment
and summary entrypoints are `benchmarks/serving/eagle3_phase31.py` and
`eagle3_phase31_summary.py`. Design: `docs/EAGLE3_PHASE31.md`.
Phase 3's uncommitted source was archived before editing. No new commit, reset,
default change, or modifications to user-owned `AGENTS.md` were made.

## Experimental Protocol

Qwen3-14B BF16, TP=2, 2 x RTX 4090, eager, dedicated EAGLE-3, persistent state,
single-request greedy execution. Model length and token budget are 2048,
max sequences 4, GPU memory utilization 0.70.
Target revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`.
Draft revision: `3d13517724e81cb409ddf1d4650772ec52f1e18e`.
The checkpoint's exact target training revision remains unpublished.

Calibration uses status logs, recycling prose and numeric identifiers at
255/768/1024 context, 96 outputs, K=1..6: 54 independent requests. Their session
E2E sums to 113.756 seconds, excluding engine startup and separately saved warmup.
All catch-up row counts 1..7 were observed in every bucket: no nearest-row
substitution was needed. Initial choices are K=4 at 255/1024 and K=5 at 768.

Held-out inputs are orchard continuation, fermentation explanation and UUID
generation, not Phase 3 prompts and not prior-building prompts. Exact contexts
are 255/1024; forced output horizons are 64/256/512. Every cell runs fixed K=1..6
and prior-adaptive, with three repeats and rotated policy order: 378 requests,
104,832 output tokens. Prompt IDs, plan and prior are saved before measurement.
No tuning follows held-out results. These are synthetic inputs, not a broad
production distribution. `ignore_eos=True` fixes experiment lengths; EOS handling
is validated separately.

Warmup is excluded and saved separately. Timing uses audit=False for *all*
policies; correctness uses audit=True. Rank/finite/state checks still execute.
No absolute speed comparison to Phase 3's audit=True measurements is valid.
Fresh request state and exclusive zeroed target pages prevent prefix-cache reuse.
Monotonic host timing measures the end-to-end execution path, not GPU kernel time.
Session E2E ends at the last committed output; API E2E additionally includes
controller setup and cleanup. TPOT is (last-first output time)/(outputs-1).
Speculative tokens committed together share a timestamp.

No outlier is removed. All repeats, min/max and step records are retained.
The best fixed K per cell is selected post hoc and is an optimistic comparator;
a single strong fixed K over the entire grid is also required. Three repeats
do not establish statistical significance or universal generalization.

## Correctness

The complete CPU suite passes 88 tests, including nine prior-controller tests.
Source comparison to Phase 2 checkpoint `0ce2c1e` finds no changes in 18 frozen
production files. Calibration/correctness/held-out controller hashes agree.

Qwen3-14B TP=2 checks pass at context 255/256/257/1088: replaying selected K and
proposals gives exact committed outputs, verification top-1 IDs, accepted lengths
and rank states. Additional checks pass max_tokens=1/2/7, forced rejection,
initial EOS, and context 1024 with 512 output tokens and bounded decision epochs.
Prior Phase 2/3 full/partial acceptance, exception, suffix, cursor and KV-hash
checks remain applicable to frozen logic; they are not claimed as newly rerun
GPU hash diagnostics in this stage.

The accepted BF16 cross-shape contract remains unchanged. Historical strict serial
parity failures remain failures. New output disagreements require diagnostics;
no arbitrary tolerance or silent K=0 fallback is allowed.

Final held-out audit: **378/378 requests, 104,832 output tokens, zero new output
disagreements against same-cell/repeat K=3**. All 108 fixed-K cells have identical
outputs and proposals across their three repeats. Ten GPU correctness check groups,
all request cleanup flags, prior equality and frozen source hashes pass. No observed
NCCL/CUDA error, deadlock or retained-state growth. Rank-0 allocated memory after
every measured request is 19,566,306,816 bytes; both GPUs return to 1 MiB after exit.
This does not assert bitwise BF16 tensor identity across different K.

## Results

Each policy generated 14,976 tokens over 54 requests. Aggregate tok/s below is
tokens divided by summed single-request session E2E, not concurrent serving
throughput; model startup, offline calibration and between-trial artifact writes
are excluded. API E2E includes setup/cleanup and gives the same conclusion.

| Policy | Summed E2E s | Summed API E2E s | Aggregate output tok/s |
|---|---:|---:|---:|
| Fixed 1 | 406.468 | 406.582 | 36.844 |
| Fixed 2 | 350.340 | 350.457 | 42.747 |
| Fixed 3 | 335.053 | 335.169 | 44.697 |
| Fixed 4 | 335.062 | 335.177 | 44.696 |
| Fixed 5 | 351.602 | 351.717 | 42.594 |
| Fixed 6 | 365.416 | 365.530 | 40.983 |
| Prior adaptive | 343.932 | 344.059 | 43.543 |

Fixed 3 and 4 are effectively tied on this aggregate, not a meaningful 9 ms win.
Adaptive is 2.65% slower than fixed 3 in summed E2E. Summing cell medians instead
favors fixed 4 (109.003 s versus adaptive 113.348 s, about 3.99% slower).
Adaptive is slower than both K=3 and K=4 in each of the three whole-grid repeats.
No significance or causal speed-regression claim is made from three repetitions.

| Output horizon | Best uniform fixed K for that horizon | Fixed total s | Adaptive total s | Adaptive difference |
|---|---:|---:|---:|---:|
| 64 | 2 | 27.842 | 31.582 | +13.43% |
| 256 | 4 | 104.993 | 112.654 | +7.30% |
| 512 | 3 | 199.497 | 199.697 | +0.10% |

Long horizons narrow the aggregate gap, but near equality at 512 is not a stable
win, nor proof that exploration amortization alone caused the change: the output
content and acceptance regime also change. There are no active online probes to
amortize; the first 64-token commitment and prior adaptation are still costs.

Cell results below are medians over all three retained samples. Positive means
adaptive slower. The full 126-row `trial-table.md` and `summary.json` include
TPOT, tok/s, acceptance, forwards, decomposition, distributions and repeat ranges.

| Family | Context | Output | Best fixed K | Fixed E2E ms | Adaptive E2E ms | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Orchard | 255 | 64 | 3 | 919.80 | 955.87 | +3.92% |
| Orchard | 255 | 256 | 4 | 3585.55 | 3560.10 | -0.71% |
| Orchard | 255 | 512 | 3 | 7105.24 | 8341.11 | +17.39% |
| Orchard | 1024 | 64 | 3 | 1053.48 | 1405.33 | +33.40% |
| Orchard | 1024 | 256 | 3 | 3699.77 | 4360.14 | +17.85% |
| Orchard | 1024 | 512 | 3 | 7371.91 | 7609.38 | +3.22% |
| Fermentation | 255 | 64 | 1 | 2020.01 | 2267.48 | +12.25% |
| Fermentation | 255 | 256 | 2 | 8738.22 | 9828.04 | +12.47% |
| Fermentation | 255 | 512 | 2 | 16228.16 | 17790.84 | +9.63% |
| Fermentation | 1024 | 64 | 2 | 2172.84 | 3001.60 | +38.14% |
| Fermentation | 1024 | 256 | 1 | 8200.39 | 10870.30 | +32.56% |
| Fermentation | 1024 | 512 | 2 | 16597.64 | 16894.37 | +1.79% |
| UUID | 255 | 64 | 3 | 1490.91 | 1796.65 | +20.51% |
| UUID | 255 | 256 | 4 | 5278.89 | 6173.62 | +16.95% |
| UUID | 255 | 512 | 5 | 9776.60 | 10284.22 | +5.19% |
| UUID | 1024 | 64 | 6 | 754.69 | 858.69 | +13.78% |
| UUID | 1024 | 256 | 6 | 2323.73 | 2502.72 | +7.70% |
| UUID | 1024 | 512 | 6 | 4480.72 | 4847.16 | +8.18% |

Adaptive beats K=3 median E2E in 6/18 cells and the best measured fixed K in
only 1/18, by just 0.71%. That small isolated advantage is insufficient evidence
of a generalizable performance benefit.

### Acceptance and Stability

Actual K=3 accepted/proposed ranges across output horizons:

| Family | Context 255 | Context 1024 |
|---|---:|---:|
| Orchard | 0.734-0.741 | 0.729-0.733 |
| Fermentation | 0.099-0.140 | 0.113-0.142 |
| UUID | 0.306-0.457 | 0.849-0.947 |

Thus high/medium/low acceptance are covered, but UUID is not uniformly a
medium-acceptance workload: context and generated content change its regime.
Low-acceptance fermentation favors K=1/2. Orchard favors K=3/4 despite relatively
high acceptance; larger K is not automatically worthwhile. UUID/1024 benefits
from K=6, while UUID/255 shifts from K=3 toward K=5 as the horizon increases.

Across 54 adaptive requests: 50 switches, six compressed-run A->B->A returns,
maximum five switches in one request. Horizon 64 has zero switches/returns;
256 has 20/1; 512 has 30/5. Decision epochs are exactly 1/4/8 by horizon.
Requested-K step counts are K1=0, K2=91, K3=954, K4=3848, K5=1344, K6=422.
Online calibration/probe steps are zero. This is bounded switching, not zero
oscillation and not a proof of an optimal controller.

Total measured selection work is 174.697 ms over 343.932 seconds of adaptive
session E2E (about 0.051%). Offline prior calibration costs 113.756 seconds and
must not be described as free. Adaptive proposal time totals 55.038 s, comprising
43.047 s proposal work, 11.675 s incremental catch-up and 0.316 s initial
conditioning. Verification totals 279.061 s. Target prefill, commit and other
bookkeeping account for the remaining session time. Full per-K forecasts and posterior probabilities
at every decision epoch are retained in `summary.json`; every step is in raw JSON.

### Interpretation and Limits

The attribution model is improved structurally: previous accepted-length cost is
separated from current K work. But that alone does not produce a reliable K
selector. A pooled, label-free prior chooses K=4 for both tested initial contexts.
It cannot distinguish the short low-acceptance cases before generating tokens.
No probes also means unchosen alternatives keep stale cost estimates. Holding
K for 64 outputs intentionally trades reaction speed for stability.

Read-only selected-action diagnostics compare forecasts to actual full-K,
noninitial steps with a measured next catch-up. Across horizons/repeats, predicted
progress exceeds realized progress by about 28.9%/17.9% for fermentation at
255/1024, but underestimates UUID/1024 by about 23.2%. Predicted cost totals are
about 0.95-0.99 of measured costs. These are observations on selected actions,
not counterfactual estimates for unchosen K and not a causal decomposition of the
overall loss. No parameters were changed in response.

Timing noise is material. All 18 short adaptive requests choose only K=4, and
their proposal IDs, target top-1 IDs, accepted lengths and committed IDs match
same-repeat fixed K=4 exactly. Yet their total E2E is 31.582 s versus 28.120 s
for fixed K=4; measured short-request selection work totals only 14.905 ms.
This is evidence against attributing that entire 3.462-second gap to selection
overhead. Run-order/host/GPU variability remains a confounder; its exact cause
was not isolated. Three rotated repeats cannot remove it. It also prevents
interpreting the single 0.71% winning cell as convincing adaptive improvement.

Largest verification sample: **294.945 ms**, orchard/1024/512/repeat2/fixed1,
step 236. The earlier 250.211 ms adaptive sample is also retained. No outlier
filtering, failed-sample deletion or threshold-based correctness waiver occurred.

## Finalization

The limited experiment answers the stopping question: longer horizons help
approach a strong fixed baseline, but there is no stable measured superiority.
Freeze both Phase 3 and this prior-based variant as research results. Keep
fixed-K=1..6 available; K=3/4 are strong general comparators on this grid, not
universal defaults, and no default is changed. Do not spend another iteration
adding thresholds or fitting these workloads. Do not enter SLO-aware K on this
evidence. The next useful project stage is separately designed concurrent
speculative serving/final integration, preserving the accepted state contract.

## Artifacts and Reproduction

Raw root: `/root/autodl-tmp/benchmarks/eagle3-phase3.1/`.
Each stage stores source, environment/config manifest, warmup and per-request
JSON with decisions, posterior forecasts, per-K cost estimates and all step
latencies. Large/raw artifacts are outside Git. Small report artifacts go here.

From the repository, using the existing baseline virtualenv and offline model
cache, run each stage with a fresh output directory:

```bash
python -m benchmarks.serving.eagle3_phase31 --stage calibration --output "$RAW/calibration"
python -m benchmarks.serving.eagle3_phase31 --stage correctness --prior "$RAW/calibration/prior.json" --output "$RAW/correctness"
python -m benchmarks.serving.eagle3_phase31 --stage heldout --prior "$RAW/calibration/prior.json" --repeats 3 --output "$RAW/heldout"
python -m benchmarks.serving.eagle3_phase31_summary --raw "$RAW" --output benchmarks/eagle3-phase3.1
python -m unittest discover -s tests -q
```

`RAW=/root/autodl-tmp/benchmarks/eagle3-phase3.1`; environment values and exact
stage arguments are saved in each manifest. The original directories are kept,
so use a different RAW to rerun rather than overwrite them.
