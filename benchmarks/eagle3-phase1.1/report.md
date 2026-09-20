# EAGLE-3 Phase 1.1 Numerical Equivalence Audit

## Accepted Status (2026-09-20)

The user adopted `docs/EAGLE3_NUMERICAL_CONTRACT.md`: **numerical semantics
accepted under the documented BF16 cross-shape contract; strict serial parity
not guaranteed.** Historical strict parity failures below remain failures.
The following investigation records the original pre-acceptance decision;
its references to an open gate are superseded only for numerical semantics.

## Decision

Outcome **2**, within the two replayed failure cases: the first divergence is
BF16 shape-dependent **layer 0 QKV linear**, not a demonstrated attention-mask or
speculative-state bug. Q=1 uses a GEMV kernel; Q=4 uses a WMMA GEMM kernel.
Their small rounding differences propagate through normalization and subsequent
layers, and accepted KV legitimately retains these different numerical values.

No production fix was made. **Strict original greedy token parity is not
restored. The existing strict-parity gate remains open.** A proposed, explicitly
different numerical contract is given below; adopting it would permit numerical
semantics sign-off, not retroactively turn the old strict-parity failures into
passes. No persistent speculative KV, adaptive K, Scheduler, sampling or CUDA
optimization work was started.

## Experiment

Replay the exact Phase 1 sky and arithmetic proposal/commit traces through the
first output divergence. Hold tokens and candidates fixed, rather than rerunning
or tuning the draft. Compare five paths on private page snapshots:

| Name | Query execution | Initial committed KV |
|---|---|---|
| serial_s | original q=1 decode | speculative prefix S |
| varlen1_s | q=1 paged prefill | identical S |
| parallel_s | q=4 paged verification | identical S |
| serial_n | original q=1 decode | serial teacher-forced prefix N |
| parallel_n | q=4 paged verification | identical N |

Each intervention restores the same physical pages before the next path. N is a
test-only reference snapshot following the same committed tokens; it is not a
new production cache implementation. The real verification/commit is subsequently
replayed and checked against the captured parallel result.

Sky covers iterations 0..12; arithmetic covers 0..5. Both ranks are inspected:
19 iterations, 38 rank-iterations, and 76 query positions per model control.
The 0.6B control teacher-forces these same token IDs; it does NOT load the 14B
speculator or claim that its predicted tokens should match the 14B model.

Captures include each layer's input residual, normalized attention input, QKV,
Q/K norm, post-RoPE Q/K and positions, attention core/output, post-attention
residual, MLP input/gate-up/output, effective layer output, final norm, LM-head
input and logits. Summaries retain max/mean absolute error, RMS, relative L2,
unequal count, finite checks and per-query max error. First differing operators,
largest absolute-error operators, layer-0 intermediates and logits are saved as
full tensors on the data disk. No numerical pass threshold selects these tensors.

## Layer Localization

All **38 BF16 rank-iterations** first differ at `00.qkv`. Layer-0 input residual
and attention input are exactly equal. The same-input standalone linear control
reproduces the difference without attention, KV writes, acceptance or the draft.

| First iteration, rank 0 | Sky max abs | Arithmetic max abs |
|---|---:|---:|
| layer 0 input residual / attention input |0 / 0|0 / 0|
| layer 0 QKV projection |0.00048828125|0.000244140625|
| layer 0 Q norm |0.03125|0.03125|
| layer 0 K norm |0.015625|0.00390625|

Thus the first nonzero error is the projection; its absolute magnitude increases
at Q/K normalization before entering attention. These operators have different
scales, so the table is not a claimed relative-error amplification bound.
Full per-layer progression is in `layer-operator-summary.csv` and raw JSON.

Profiler evidence for the actual QKV controls includes:

- q=1: `internal::gemvx::kernel<...__nv_bfloat16...float...>`.
- q=4: `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8`.

On eight local controls (two ranks, first/failing iterations of both cases),
turning off BF16 reduced-precision reduction does not remove the mismatch.
FP32 serial/parallel dot-product max difference is 1.4901161e-7; FP32 versus
FP64 reference max difference is 1.1920929e-7. The native BF16 differences range
from 0.000244140625 to 0.001953125 for these controls. These are observations,
not acceptance thresholds.

Example from identical BF16 inputs/weights: serial gives 0.013427734375,
parallel gives 0.01336669921875, while the FP64 dot product is
0.013397241446156727, close to their rounding boundary. Ordinary accumulation
order differences can therefore change the final BF16 rounded value.

## Attention Isolation

Original decode calls `flash_attn_with_kvcache`; verification calls
`flash_attn_varlen_func` with a block table. Both are causal. For committed cursor
C and query row i, position is C+i and visible context ends at C+i+1. Verification
uses right-aligned causal masking over a total key length C+q, not a top-left
q-by-k triangular mask.

The unmodified prepare methods reproduce identical serial/parallel position IDs
and physical slots on all **19 metadata controls**. Block tables, cu_seqlens,
query/key lengths and per-row serial context lengths are recorded in
`metadata-controls.json`. Actual per-layer position captures also agree.

For each BF16 layer/rank/iteration, hold post-RoPE Q and paged K/V fixed and
compare q=1 cache attention, q=1 varlen, q=4 varlen and q=4 cache attention:
**all three pairwise comparisons are bitwise equal in all 1,520 fixtures**.
Whole-target serial decode versus q=1 varlen also agrees exactly at every captured
operator. Thus changing this attention API alone does not explain these cases.

The profiler shows `flash_fwd_splitkv_kernel` specializations and a
`flash_fwd_splitkv_combine_kernel`; symbol parameters differ between calls.
Different symbols do not by themselves establish different numerical outputs.
The pinned FlashAttention traits use float accumulation with BF16 elements;
BF16 input/output still does not imply FP64-equivalent attention. Local FP64
explicit masked attention is retained as a reference: both native attention
paths have the same error relative to it, not zero error.

Sources: [FlashAttention interface, v2.8.3](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py),
[kernel accumulation traits](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/kernel_traits.h).
PyTorch also explicitly distinguishes batched/sliced numerical results and
documents the reduction flags; this supports the interpretation but does not
replace the experiments. [PyTorch 2.8 numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html).

## Sky Root Cause

At iteration 12, output position 20 (zero based), on speculative prefix S:
serial_s and varlen1_s select comma ID11; parallel_s selects and ID323.
On serial prefix N, both serial_n and parallel_n select ID11.

There are therefore two numerical contributors: current-query shape differences
and the numerical prefix retained from previous parallel steps. The earlier
same-KV experiment correctly detected a shape effect, but attributing that effect
specifically to FlashAttention was too narrow. Layer-0 linear isolation identifies
an earlier source, and the fixed-QKV experiment excludes an attention-path
difference in the measured fixtures. HF teacher-forced at this common prefix
also selects ID11, as recorded in Phase 1.

## Arithmetic Backward Trace

The divergence starts before output position 10:

1. Before iteration 0, S and N have identical committed KV and features.
2. Iteration 0 already differs at layer-0 QKV, then Q/K norm and subsequent layers.
3. At iteration 1 entry, layer-0 retained KV differs by max 0.00390625 on rank0;
   layer1 differs by 0.0625. Feature-prefix max difference is 1.0. This is before
   the first output-token disagreement, not an effect of different generated text.
4. By iteration 5, serial_s and parallel_s both select plus ID488. Replacing only
   the prefix with N makes both serial_n and parallel_n select multiply ID24768.
   HF teacher-forced also selects ID24768.

This intervention localizes the decisive difference to accumulated numerical
prefix values. The prefix is not populated with the wrong accepted tokens:
the retained values match the parallel computation exactly. Current-step serial
execution cannot undo earlier numerical drift. The first source of that drift
is already visible in iteration0's projection, not an unexplained late KV event.

## KV and State Checks

All 38 BF16 and 38 valid FP32 rank-level commit records pass:

- old committed KV prefix is bitwise unchanged by verification;
- retained KV equals the corresponding audited parallel rows exactly;
- rejected suffix slots are zero;
- retained target features equal old features plus the accepted computed rows;
- cursor advances by pending-input plus accepted-input count;
- committed token count is valid KV length plus one pending token.

Proposals are fixed from the old trace, so draft feature drift cannot explain the
target comparisons by secretly selecting different candidates. No logical error
in acceptance retention, fallback pending token, feature slicing, suffix clearing
or cursor/position handling was found in this coverage. This is not a proof for
untested concurrency, preemption or arbitrary contexts. Phase 1 boundary/EOS/
forced-rejection tests remain separate, unchanged evidence.

## High-Precision Controls and Probe Corrections

The valid Qwen3-0.6B TP2 FP32 control has **76/76 query top1 agreement across all
five paths**. Serial/parallel logits max abs is 3.6716461e-5, with max relative L2
8.4682745e-7. Independent HF FP32 eager teacher forcing agrees **16/16** at selected
initial/failing-iteration queries; max abs is 4.4941902e-5. No raw-logit cutoff was
used to reclassify a mismatch. Full 14B FP32 execution was not run: high-precision
14B evidence is local QKV FP32/FP64 and fixed-input FP64 attention, not a full-model
FP64 certificate. Published model weights are upcast for these arithmetic controls.

Two invalid probe/control attempts are retained and explicitly excluded from
sign-off, not silently deleted:

- Initial `bf16` registered layer-specific hooks on a shared cached RoPE module.
  The probe was corrected with active-layer gating and a unit test; `bf16-final`
  contains the valid RoPE captures. This was an instrumentation issue.
- Initial `fp32-small` reused production RMSNorm's BF16-oriented in-place code.
  `.float()` and `.to(float32)` alias FP32 storage: normalizing the temporary also
  overwrites the returned residual. Its apparent self-consistency was invalid;
  HF rejected it 0/16. The test-only control now uses functional, non-in-place
  RMSNorm, with an input/residual preservation test. Production RMSNorm remains
  unchanged. This FP32 aliasing issue does not explain the BF16 target mismatch,
  where the dtype conversions allocate separate storage.

The FP32 control also uses explicit high-precision causal attention because the
BF16 FlashAttention path is not a full FP32 target implementation. It therefore
validates the mathematical/state control, not a nonexistent FP32 FlashAttention
kernel. All failed/invalid control artifacts remain on disk with their labels.

## Proposed Numerical Contract

1. Speculation off preserves original behavior and default Scheduler/sampler.
2. Fixed-K greedy acceptance is exact with respect to the actual parallel target
   verification logits; only accepted proposals and target fallback/bonus outputs
   are committed, with exact transaction invariants.
3. Fixed inputs, weights, shapes and execution mode must reproduce outputs.
4. Cross-shape BF16 token equality is not universally promised. Every observed
   disagreement remains a regression fixture requiring same-prefix/state checks,
   operator localization and independent precision controls. It is not waived
   merely because a chosen absolute error threshold is large enough.
5. Greedy-only evidence does not establish stochastic distributional equivalence,
   semantic answer equivalence, universal 14B FP32 equivalence or performance gain.

For diagnostics, if serial top1 margin m exceeds twice the measured paired-logit
infinity difference delta, argmax agreement is mathematically guaranteed. Failure
of that sufficient condition does not automatically authorize a mismatch. Delta
is an observed pair difference, not an invented universal model error budget;
the localization/state/high-precision requirements above still apply.

These cases support this narrower contract. **The existing strict-token parity
gate is not automatically closed or weakened by this report.** Numerical root
cause investigation can end as outcome2; formal Phase1 sign-off requires explicit
adoption of the stated contract, or additional work if exact serial output is
still required. No serial verification replacement, tie-break hack, low-margin
K=0 fallback or sampler change was introduced.

## Files, Artifacts and Reproduction

New test-only files: `benchmarks/serving/eagle3_equivalence.py`,
`eagle3_precision_controls.py`, `eagle3_equivalence_summary.py`, and
`tests/test_eagle3_equivalence.py`. Production source hashes match the Phase1.1
entry snapshot, including previously uncommitted Phase1 changes. Scheduler and
all production modules are frozen; existing user changes are untouched.

`summary.json`, `layer-operator-summary.csv`, precision/metadata JSON and
`frozen-source.json` are next to this report. Full tensors and per-rank/per-layer
JSON live at `/root/autodl-tmp/benchmarks/eagle3-phase1.1/`. Valid result directories
are `bf16-final`, `fp32-small-final` and `precision-controls-final`. Invalid earlier
attempts remain in `bf16`, `fp32-small` and `precision-controls`.

Target revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`.
Small-model revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
Dedicated draft revision and its unpublished training-target revision limitation
are unchanged from Phase1; no draft weights or K were tuned here.

From the repository with the nano-baseline venv Python, run serially:

```bash
python -m benchmarks.serving.eagle3_equivalence --out NEW_BF16_DIR
EAGLE_AUDIT_FP32=1 python -m benchmarks.serving.eagle3_equivalence --small --out NEW_FP32_DIR
python -m benchmarks.serving.eagle3_precision_controls --bf16 NEW_BF16_DIR --fp32 NEW_FP32_DIR --out NEW_CONTROL_DIR
python -m unittest discover -s tests -v
```

Environment: CUDA_VISIBLE_DEVICES=0,1; HF_HUB_OFFLINE=1;
TRANSFORMERS_OFFLINE=1; PYTHONPATH=.; TMPDIR=/root/autodl-tmp/tmp;
TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/tmp/torchinductor;
TRITON_CACHE_DIR=/root/autodl-tmp/tmp/triton; NCCL_DEBUG=WARN.
BF16 target uses eager TP2, max length/batched tokens256, max sequences1,
memory utilization0.75; small FP32 uses0.2. These are diagnostic executions,
not serving latency or throughput measurements. GPU processes exit normally.
**69 unit tests passed. No Git commit was made.**
