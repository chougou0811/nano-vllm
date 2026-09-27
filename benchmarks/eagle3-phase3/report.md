# EAGLE-3 Phase 3: Adaptive Speculative Length

## Decision

Implementation, scoped correctness validation and the 126-trial experiment are
complete. **The performance objective is not achieved:** this controller does
not beat a well-selected fixed K. Do not promote adaptive to the default or add
SLO coupling on the strength of these results. No parameters were retuned after
seeing the pilot or final trial results. Phase 3 changes are not committed.

Phase 2 was accepted and checkpointed as **`0ce2c1e`**, excluding raw artifacts,
weights, caches and the pre-existing user-owned `AGENTS.md` changes.

## Controller

`AdaptiveK` is request-local, with K in {1,2,3,4,5,6}. Initial K=3. A 512-token
context bucket has independent per-K EWMA estimates (alpha=0.25) of committed
output count, proposal time and verification time. Score is:

`EWMA(effective outputs) / (EWMA(proposal ns) + EWMA(verification ns))`.

After initial prefix setup, each arm is calibrated once. Exploitation changes
the incumbent only for >10% estimated rate advantage. One round-robin probe
runs every 12 learned observations. This is the entire policy; it has no SLO,
scheduler, acceptance-only or low-logit-margin override.

A 16-step history records accepted/proposed, accepted/verification, effective
outputs/verification and consecutive zero-accept rejections. Every decision
records per-K costs, sample counts, scores, context, reason and incumbent.
First-prefix setup, terminal and output-clipped observations are retained in
all raw/performance data but do not train incremental cost estimates.
Actual K is clipped to remaining-1. K=0 occurs only for the existing final
target-only drain, never as a hidden controller fallback.

## Frozen State Semantics

No edits to `draft_state.py`, `runtime.py`, `acceptance.py`, `draft.py`, the
Scheduler implementations, BlockManager, ModelRunner, attention, TP or sampler.
The session changes only choose K before proposal and observe after commit.
All ranks receive the same proposal list through the existing RPC. Target and
draft cursor rules, suffix discard and pending fallback/bonus remain Phase 2.

Production changes: new `nanovllm/speculative/controller.py`; optional arguments
and telemetry in `speculative/session.py` and `engine/llm_engine.py`.
Tests: new `tests/test_eagle3_adaptive.py`, extended persistent-state controls.
Tooling: `benchmarks/serving/eagle3_phase3.py` and `eagle3_phase3_summary.py`.
Design/API documentation: `docs/EAGLE3_PHASE3.md`.

Opt-in example with an already initialized eager engine:

```python
result = engine.generate_eagle3(
    prompt,
    draft_path="/root/autodl-tmp/models/Qwen3-14B_eagle3",
    reference_path="/root/autodl-tmp/references/eagle-pinned",
    speculative_length=3,
    draft_state_mode="persistent",
    adaptive_k=True,
    max_tokens=64,
)
```

Existing API defaults are unchanged; benchmark fixed-K baselines explicitly
use persistent state. Speculation-off model/sampling paths remain unchanged.

## Correctness

- **79 unit tests pass**, including immutable-source checks against Phase 2,
  rate-vs-acceptance objective, cold/terminal censoring, context isolation,
  rejection history, hysteresis and deterministic observation replay.
- Exact feature-dependent toy controls and pinned-reference small FP32 controls
  cover variable 1 -> 6 -> 1 proposal lengths and incrementally retained KV.
- BF16 Qwen3-14B TP=2: scripted 1 -> 6 -> 1 at 255/256/257/1088 contexts
  matches full-reconstruction fixed-proposal replay. Output IDs, verification
  top-1 IDs, accepted lengths and rank status agree. Rank-0 valid target KV and
  target-feature hashes/cursors agree at recorded iteration boundaries; both
  ranks check feature replication, suffix zeroing and logical state agreement.
- Repeated rejection at each context, accepted 0/1/partial/full=6, initial EOS,
  accepted EOS, fallback EOS, max_tokens 1/2/7 and repeated cleanup pass.
- Actual adaptive K decisions/proposals replay exactly; wall-clock-driven
  choices themselves are not promised to be identical between repetitions.
- Initial `correctness/` failed its **coverage** gate: natural proposals never
  accepted six tokens. Every transaction check passed. `correctness-final/`
  adds a test-only causal target oracle, verifies full acceptance at K=6 and
  passes all 18 check groups. The original failed coverage record is retained.
  Oracle forwards are logged separately and never enter performance trials.
- **126/126 planned trials, 8,064 measured output tokens**, normal process exit.
  Every fixed-K repeat has identical outputs/proposals. Every trial's output
  equals its same-workload/context/repeat K=3 reference: **zero new output
  disagreements**. This does not erase Phase 1 strict serial-parity failures,
  prove bitwise raw KV identity, or promise universal cross-shape token parity.
- No observed NCCL/CUDA errors, deadlock or cleanup failure. Rank-0 allocated
  memory after every measured request is 19,566,306,816 bytes. Both GPUs return
  to 1 MiB after engine exit. No observed request-to-request KV/state growth.

## Workload and Method

Qwen3-14B BF16, TP=2, two RTX 4090, eager, original Scheduler unchanged,
dedicated EAGLE-3 and persistent draft state. Limits: model length/batched
tokens=2048, max sequences=4, memory utilization=0.70. Single request only.

Target revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`.
Draft revision: `3d13517724e81cb409ddf1d4650772ec52f1e18e`.
The draft's exact target training revision is still unpublished.

The K=3 pilot covered six input families at 255 and 1024 tokens. Selected:

| Regime | Family | Pilot acceptance, 255 / 1024 |
|---|---|---|
| High | Repeated scientific-note text | 0.920 / 0.865 |
| Medium | 17 x 23 arithmetic explanation | 0.415 / 0.330 |
| Low | Why the sky is blue | 0.195 / 0.223 |

Non-repetition prompts use identical repeated-text padding before a chat-form
question to control exact context length. These are synthetic, narrow inputs,
not representative production traffic. Trials force 64 output tokens with
ignore_eos=True; real EOS semantics are tested separately. Pilot-selected
families are not described as held-out validation.

Three families x two contexts x seven policies x three repeats = 126 trials.
Plan/prompts were saved before measurement. Policy order is rotated across
the three repeats, not a fully balanced seven-order Latin square. Fresh
request state and zeroed exclusive target pages prevent prefix-cache reuse.
Warmup for each fixed K and context is saved separately. Target auditing is
enabled identically in all modes. No state-hash copies or oracle forwards run
inside performance trials. Times are monotonic host latency, not kernel time.

No samples removed. The largest measured verification step is **191.505 ms**
(`trial-repetition-255-2-adaptive`, step 1), retained in the raw data and totals.
GPU/host variability is material; three repeats are not enough for a strong
causal performance claim. Medians below summarize all three retained samples.

## Results

Full 42-row fixed-K/adaptive comparison is in `trial-table.md`. It includes
acceptance, accepted/verification, effective outputs/verification, proposal and
verification latency, TPOT, E2E, output tok/s and target/draft forward counts.
`summary.json` retains all repeat metrics, min/max, per-K costs and distributions.

Best fixed K selected **post hoc by median E2E** within each cell:

| Workload | Context | Best fixed K | Fixed E2E ms | Adaptive E2E ms | Adaptive difference |
|---|---:|---:|---:|---:|---:|
| Repetition | 255 | 4 | 695.86 | 925.96 | +33.1% |
| Repetition | 1024 | 5 | 847.46 | 912.02 | +7.6% |
| Arithmetic | 255 | 4 | 1348.20 | 1574.39 | +16.8% |
| Arithmetic | 1024 | 3 | 1580.09 | 1666.47 | +5.5% |
| Sky | 255 | 2 | 1739.08 | 1838.39 | +5.7% |
| Sky | 1024 | 2 | 1780.48 | 2103.64 | +18.2% |

Adaptive beats K=3 median E2E in only 2/6 cells, and beats the best measured
fixed K in **0/6**. Across all 18 requests per policy, summed measured E2E is
25.855 s for fixed K=4, 26.778 s for K=3, and 27.803 s for adaptive. Thus even
against a single post-hoc-selected global fixed K=4, adaptive is about **7.5%
slower** on this equal-count mix. Summing cell medians instead favors K=5;
this aggregation sensitivity is another reason not to declare a universal K.

High-acceptance repetition benefits from K=4/5; K=6 adds draft work without
enough extra accepted progress. Medium arithmetic favors K=3/4. Low-acceptance
sky favors K=2, not necessarily K=1: K=2 saves verification steps while K>=3
adds many draft forwards for little additional progress. Acceptance fraction
alone would incorrectly favor K=1 even on the high-acceptance workload.

## Stability and Failure Analysis

Requested-K distributions across three repeats (tail clipping is separately
recorded as actual_k; K=0 final drain is not a candidate):

| Workload/context | K1 | K2 | K3 | K4 | K5 | K6 |
|---|---:|---:|---:|---:|---:|---:|
| Repetition/255 | 6 | 3 | 6 | 3 | 27 | 3 |
| Repetition/1024 | 6 | 3 | 6 | 3 | 27 | 3 |
| Arithmetic/255 | 57 | 35 | 6 | 3 | 3 | 3 |
| Arithmetic/1024 | 19 | 44 | 6 | 3 | 3 | 27 |
| Sky/255 | 29 | 34 | 9 | 30 | 17 | 3 |
| Sky/1024 | 28 | 6 | 9 | 37 | 31 | 3 |

Repetition exploitation stays at K=5 in all six requests. This is stable but
not always optimal. Across all 18 adaptive requests there are **63 exploitation
switches and 21 A->B->A run returns**, after excluding calibration/probes and
compressing repeated choices. Arithmetic/1024 and sky genuinely oscillate.
The raw `exploit_reversals` field counts only adjacent-step ABA; use summary
`stability.run_returns` for returns across multi-step dwell periods.

Evidence-supported causes, not a claim of complete causal attribution:

1. **Sparse, position-dependent reward.** Each K is initially tried on a
   different token prefix. A favorable prefix can look like a favorable K.
   Repetition/255 repeat 0 initially estimates K4 at 89.17 tok/s after accepting
   3, versus K5 at 124.92 tok/s after accepting 5. K4 never gets another sample
   before the request ends, although fixed K4 wins that cell overall.
2. **Short-horizon calibration overhead.** High-acceptance adaptive runs have
   only 16 verifications; six are calibration plus one later probe. Exploring
   all arms consumes much of a 64-token request and is included in E2E.
3. **Stale alternatives and nonstationary acceptance.** On arithmetic/1024,
   exploitation cycles between K2 and K6; on sky it cycles among K1/K4/K5.
   A 10% margin cannot remove reward variation caused by changing prefixes.
4. **Costs are not fully attributable to current K.** Incremental draft
   catch-up depends on the previous iteration's accepted length. The controller
   buckets context and K, not this lagged state. The objective also excludes
   commit/control overhead, which E2E includes. Timing variability is retained,
   not attributed automatically to the controller or kernels.

No V1-style minimum-prefill lock-in is relevant here: Scheduler is untouched.
But stable selection is not proof of convergence to the best K, and this adaptive
controller must not be described as robustly converged.

## Next Step

Keep fixed K=3 persistent state as the existing comparison baseline and expose
other fixed K values for measured workloads. Adaptive remains opt-in research
code, not a recommended performance default. Do not add more thresholds merely
to fit these inputs.

**SLO-aware K is not justified yet.** First use separate calibration/held-out
inputs and longer output horizons to evaluate sparse-sample/credit-assignment
alternatives, retaining these negative results. This report does not implement
that followup. **Concurrent speculative serving is also not ready**: current
target KV still requires an idle engine and exclusive lease; request-local
controller ownership alone is not concurrent target/draft state isolation.

## Reproduction

Raw data: `/root/autodl-tmp/benchmarks/eagle3-phase3/` with `pilot/`, retained
`correctness/`, passing `correctness-final/`, and `experiment/`.
Logs: `/root/autodl-tmp/eagle-phase3-{pilot,correctness,correctness-final,experiment}.log`.
Unit log: `/root/autodl-tmp/eagle-phase3-tests.log`.
Each run saves manifest, source hashes/snapshots and every request trace.

```bash
env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN TMPDIR=/root/autodl-tmp/tmp \
 TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/tmp/torchinductor \
 TRITON_CACHE_DIR=/root/autodl-tmp/tmp/triton \
 /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase3 \
 --stage experiment --families repetition arithmetic sky --repeats 3 \
 --output /path/to/new-run
# Same environment: --stage pilot or --stage correctness, each with a fresh output directory.
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest discover -s tests -q
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase3_summary \
 --raw /root/autodl-tmp/benchmarks/eagle3-phase3 --output benchmarks/eagle3-phase3
```
