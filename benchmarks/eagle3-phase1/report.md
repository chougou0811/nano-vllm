# EAGLE-3 Phase 1 Baseline Integration

Date: 2026-09-20. Frozen Scheduler checkpoint: `6695617`.
Scope: isolated, single-request, greedy, eager, fixed linear-chain K=3.
This is not concurrent speculative serving or a performance-success claim.

**Correctness gate remains open:** the natural-task run exposes2/3 K=3 sequences
diverging from the original greedy sequence (first differences at zero-based
positions20 and10). K=0 matches all three. These failures are retained; passing
synthetic/boundary tests is not treated as overall correctness approval.

## Compatibility Audit

| Component | Pinned version |
|---|---|
| Qwen/Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| AngelSlim/Qwen3-14B_eagle3 | `3d13517724e81cb409ddf1d4650772ec52f1e18e` |
| SafeAILab/EAGLE reference | `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` |

The dedicated checkpoint is feature-conditioned EAGLE-3, not a small Qwen draft.
Target/draft hidden size5120, full vocabulary151936, head dimension128,
query/KV heads40/8, intermediate17408 and RoPE theta1e6 match. Target has40
layers; draft intentionally has1 layer, a 15360-to-5120 feature projection,
10240-wide attention input, and reduced vocabulary32000. `d2t` contains offsets,
not absolute IDs; its range, uniqueness and agreement with `t2d` are checked.
The target tokenizer and target embedding are reused. No embedding is included
in the draft state dict; that is the only permitted missing weight.

The published checkpoint does NOT identify the exact training target revision.
Thus architecture and vocabulary compatibility are verified, while exact
training-revision compatibility remains unverified. Config says BF16; actual
draft weights are FP16 and explicitly converted to BF16. The complete downloaded
weight SHA256 is `12d9f436fc08e6fffc580c5e4a71ecefcb44bef450e500df3fac698e9f530cbf`.
Core reference files are also digest checked before loading.

Sources: [dedicated checkpoint](https://huggingface.co/AngelSlim/Qwen3-14B_eagle3),
[reference draft](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/cnets.py),
[greedy acceptance](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/utils.py),
[Qwen3 features](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/modeling_qwen3_kv.py).

## Execution Chain

`LLMEngine.generate_eagle3 -> speculative.session.generate -> private KV lease
-> target prefill -> first target token -> reference draft proposal
-> TP target verification -> greedy accepted-prefix calculation
-> discard rejected suffix -> commit accepted outputs and target fallback/bonus
-> repeat -> clear pages/features -> release lease`.

1. Scoped target pre-hooks collect the residual stream before layers2/20/37.
   Fused residuals are reconstructed in float32 and cast back to BF16.
2. Hidden features are replicated at these TP boundaries. Both ranks execute
   target forward; only rank0 runs the unsharded reference draft. Target KV and
   attention heads remain sharded; each target rank has20 Q and4 K/V heads.
3. Verification inputs are the known pending target token plus K proposed tokens.
   Existing causal paged prefill attention evaluates all inputs in one forward.
   The external adapter retains all position logits instead of just the last row.
4. Accept the longest prefix equal to target argmax. At rejection emit target
   argmax for the rejected position; on full acceptance emit the bonus target
   token. Clip at EOS/output capacity. No stochastic acceptance is implemented.
5. Commit through the existing Sequence.append_token only after acceptance;
   proposal timestamps are never reported as output timestamps.

The original sampler does not implement temperature-zero greedy sampling.
The benchmark temporarily substitutes argmax on the original path and restores
the original sampler. Separate positive-temperature fixed-seed tests verify
the unmodified sampling path before/after speculative requests.

## KV and State

Reserve exclusive pages for prompt plus output capacity using existing
BlockManager APIs. Do not publish their hashes. The valid target KV cursor
normally trails committed token IDs by one pending token. After verification,
retain KV/features for pending plus accepted proposal inputs, zero the rejected
suffix on both ranks, and leave the new fallback/bonus token pending.

Draft KV is rebuilt on every proposal from committed target features paired with
shifted next-token embeddings. Tentative draft KV is discarded, never committed
as target state. This is intentionally a simple rollback baseline, not efficient
persistent draft caching. Finalization clears every leased page, drops features
and tentative state, then releases all block references. Host-side exceptions
are covered; process crashes or fatal NCCL errors are not recoverable transactions.

The API rejects non-idle engines. There is no scheduler admission, preemption,
prefix sharing or concurrent speculative batching. Scheduler code/defaults are
unchanged; original generate remains the speculation-off API.

## Correctness Evidence

- 65 unit tests pass, including existing Scheduler tests, exhaustive acceptance
  positions for K=1..6 vs reference, EOS/output truncation, dirty rank negative
  controls, noncontiguous physical slot clearing and repeated private leases.
- Frozen module byte comparisons and original Engine/Runner method AST parity
  pass against6695617. No TP/Attention/BlockManager/Qwen3/sampling edits.
- Trial03: original greedy, isolated K=0 and K=3 agree on all12 output IDs for
  short and255/256/257/511/512-token prompts. Trials01/02 retain earlier results.
- Trial03 proposal chain agrees with upstream topK_genrate configured as one
  branch; actual verification acceptance/fallback agrees with evaluate_posterior.
- Forced rejection, including crossing a256-token boundary, preserves original
  output. GPU checks cover EOS on initial target token, accepted proposal and
  fallback, max_tokens1/2/5, and intentional host exception cleanup.
- Every audited forward has finite logits/features; both ranks have exactly
  replicated feature tensors and matching valid cursor/token digest. Rejected
  slots and final leased pages are checked zero. No used blocks remain on exit.
- Independent HF BF16/SDPA check on the fixed short prefix: top1=12095 in both;
  feature mean cosine0.999950, mean absolute difference0.0425105, max absolute
  difference4.0. No raw max-abs threshold is used to declare correctness. This is
  one feature-prefix sanity check, not a full HF equivalence certification.

### Natural-Task Numerical Failure Audit

The test-only numerical runner snapshots the request's physical KV pages,
evaluates each proposed input with the original serial decode path, restores
the snapshot, then lets parallel verification run. It does not change production
code, accepted tokens or the request's KV state. Full logits are retained outside
Git under `numerical-audit/*-logits.pt`; JSON includes teacher-forced comparisons.

| First differing output (zero based) | Original serial | Serial with same speculative-prefix KV | Parallel verification | HF teacher-forced |
|---|---|---|---|---|
| sky /20 |comma ID11,46.75; and ID323,46.75|same tie, selects11|ID323=47.0, ID11=46.75|both47.0, selects11|
| arithmetic /10 |multiply ID24768=53.0; plus ID488=52.75|both52.75, selects488|both52.75, selects488|24768=53.0,488=52.75|

For sky, the same-prefix-KV intervention directly reproduces the argmax change
from serial vs parallel execution. Max abs serial/parallel logit delta is0.25,
cosine0.999944. This is a shape/path-dependent BF16 ranking change at a tie, not
evidence that0.25 should become a correctness threshold.

For arithmetic, current-step serial execution on the speculative prefix does
NOT recover the original winner. Earlier prefix numerical differences also
matter; attributing it solely to the current verification batch shape would be
incorrect. Original/parallel max abs delta is0.25; the winning margin is one
BF16 step. The exact earlier origin is not localized to a single layer/kernel.
Metadata and suffix-clearing checks did not expose logical KV corruption, but
those checks alone cannot prove all numerical prefix values equivalent.

Both K=3 outputs reproduce exactly in an independent engine run and an immediate
repeat, including with the reversible serial probes. Thus deterministic repeat
passes while original-sequence parity fails. Independent HF at both exact common
prefixes agrees with original top1 (HF/native max abs0.4375 and0.3125). No failed
case is filtered or reclassified as passed. The benchmark now saves all results
and raises a parity failure when summary.all_equal is false; the earlier trial04
exit code0 is not a pass, its persisted all_equal=false is the recorded result.

No heuristic near-tie threshold, special token tie-break, silent K=0 fallback or
sampling change has been added. A strict byte-for-byte serial contract would
need a separately designed validation/fallback path or reproducible numerical
execution; silently making every verification serial would defeat the stated
parallel-verification baseline. That contract remains an explicit open gate.

## Measurements

Raw data: `/root/autodl-tmp/benchmarks/eagle3-phase1/trial-01` through `trial-04`.
Small raw artifacts and logs are also archived under this report's `artifacts/`
directory; `summary.json` explicitly records the open correctness gate.
Trial03 is the final boundary/reference audit; trial04 contains natural chat
tasks. Every proposal/target/commit, rank status, token timestamp, forward count,
rejection/discard and latency remains in request JSON. Nothing is filtered.

Natural-task results are recorded in trial-04/summary.json. Boundary results are not representative acceptance
measurements: they deliberately repeat text. On trial03 the short capital prompt
accepted2/21 proposals (9.52%); boundary cases accepted8/8 or8/9 proposals.

Natural tasks: chat-0 sky explanation, chat-1 multiplication, chat-2 Python code;
target chat template, thinking disabled,32 outputs, ignore_eos=True. EOS handling
is tested independently. Each row is one request, not a repeated performance
estimate. Sequences differ for the first two tasks, so their timings must not be
interpreted as an equivalent-output speed comparison.

| Task | Path | TTFT ms | TPOT ms | ITL P50/P95/P99 ms | Output tok/s | Original IDs equal |
|---|---|---:|---:|---|---:|---|
| chat-0 | original greedy |84.54|46.90|43.12 /65.95 /84.62|20.80|control|
| chat-0 | K=3 |70.03|42.59|59.30 /65.50 /102.35|23.02|no|
| chat-1 | original greedy |53.97|43.61|42.95 /47.05 /47.37|22.76|control|
| chat-1 | K=3 |57.87|27.00|0 /61.63 /62.29|35.75|no|
| chat-2 | original greedy |59.80|35.46|31.00 /43.31 /46.32|27.61|control|
| chat-2 | K=3 |39.42|22.34|42.40 /43.91 /45.19|43.72|yes|

| Task | Proposed / accepted | Acceptance | Accepted / verification | Effective outputs / verification | Target / draft forwards | Rejections / discarded | Proposal / verification ms |
|---|---|---:|---:|---:|---|---|---|
| chat-0 |59 /10|16.95%|0.476|1.476|22 /59|20 /49|166.00 /1139.08|
| chat-1 |41 /17|41.46%|1.214|2.214|15 /41|10 /24|83.48 /745.70|
| chat-2 |46 /15|32.61%|0.938|1.938|17 /46|14 /31|91.62 /593.38|

Aggregate accepted/proposed=42/146 (28.77%), accepted/verification=42/51
(0.824). These are observed proposal statistics, not evidence of passed strict
original greedy parity. Instrumented K=0 timings and all outliers remain in the
JSON summary; chat-0 K=0 TTFT397.66ms is retained.

TTFT is internal token-ready time, not network/streaming TTFT: the API returns
only after completion. Multi-token commits share a timestamp, so within-burst
ITL=0. TTFT/TPOT/ITL include instrumentation and host/communication overhead,
not isolated kernel timing. Loading is outside request timing; cleanup has a
separate timestamp. K=0 is an instrumented target-only control, NOT the original
Scheduler path. Original-path measurements are reported separately.

Settings: BF16, TP2, eager, max_model_len2048, max_num_batched_tokens2048,
max_num_seqs4, gpu_memory_utilization0.75, original Scheduler default. Rank0 uses
GPU0; rank1 uses GPU1. Trial03 has148 KV blocks; end-of-run nvidia-smi reports
21031MiB on GPU0 and18355MiB on GPU1, including allocator reservations and the
resident rank0 draft/embedding. CUDA12.8, torch2.8.0+cu128, transformers4.57.1,
FlashAttention2.8.3; runtime reports NCCL2.27.3+cuda12.9.

First-run and shape-specific compilation overhead is retained. Trial01 short
K=0 TTFT397.07ms is not excluded. These runs are not randomized, warmed,
statistically powered serving benchmarks; no speedup conclusion follows.

## Files and Reproduction

- Existing production files: llm_engine.py adds the opt-in API (8 lines);
  model_runner.py adds a rank-RPC delegate (4 lines). Existing methods unchanged.
- New production module: nanovllm/speculative/{acceptance,draft,runtime,session}.py
  plus package initializer.
- Tests: tests/test_eagle3.py.
- Runners: benchmarks/serving/eagle3_phase1.py, eagle3_hf_features.py and
  eagle3_numerical_audit.py (test-only reversible serial probes).
- Architecture/API contract: docs/EAGLE3_PHASE1.md; this report and raw summaries.

From the repository, use `/root/autodl-tmp/venvs/nano-baseline/bin/python`:

```bash
python -m unittest discover -s tests -v
python -m benchmarks.serving.eagle3_phase1 --reference-probe --output NEW_AUDIT_DIR
python -m benchmarks.serving.eagle3_hf_features NEW_AUDIT_DIR/target-features.pt
python -m benchmarks.serving.eagle3_phase1 --natural-only --output NEW_CHAT_DIR
python -m benchmarks.serving.eagle3_numerical_audit --output NEW_NUMERICAL_DIR
python -m benchmarks.serving.eagle3_hf_features NEW_AUDIT_DIR/target-features.pt --numerical-audit NEW_NUMERICAL_DIR
```

Run HF separately after the TP engine has exited. The run environment is
PYTHONPATH=., CUDA_VISIBLE_DEVICES=0,1, NCCL_DEBUG=WARN, HF_HUB_OFFLINE=1,
TRANSFORMERS_OFFLINE=1, TMPDIR=/root/autodl-tmp/tmp,
TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/tmp/torchinductor,
TRITON_CACHE_DIR=/root/autodl-tmp/tmp/triton. Trial03/04 retain manifests and source
snapshots, not just the pre-change Git hash. Weights/reference downloads reside
on the data disk. No commit was made; pre-existing AGENTS.md edits are untouched.

## Limits and Next Stage

This establishes a bounded greedy integration, not complete reference tree or
stochastic sampling support. Only TP2/Qwen3-14B is GPU tested here. Long contexts,
multi-request ownership, preemption, prefix sharing, process failure recovery and
distributional stochastic correctness remain unverified. No SLO/adaptive K,
speculative CUDA Graph or kernel optimization was added.

Before the next engineering stage, resolve or explicitly bound the observed
serial/parallel numerical parity failure with teacher-forced diagnostics. Do not
declare Phase1 fully correct by relaxing a raw-logit threshold. After that gate,
persistent speculative KV/state management is the appropriate next step.
Do not start adaptive K yet: first remove full-prefix draft reconstruction and establish ownership,
rollback and longer/multi-request correctness without changing frozen Scheduler
policy. Checkpoint training-revision provenance also remains an open limitation.
