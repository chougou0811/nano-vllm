# Correctness and Numerical Diagnostics

## Completed Controls

- Frozen serial DraftState oracle vs new executor: nine real-state cells, B2/3/4,
  three successive iterations. Proposal IDs and confirmed KV match exactly.
- FP64 CPU causal/masked control checks every feedback hidden row and valid KV
  against serial, including ragged masks, row removal, positions and order.
- CPU ownership tests cover stale request generation, stale state generation,
  foreign/duplicate owners, changed conditioning prefixes, repeated reject,
  accept0/1/K progress, reordering, cleanup and injected exceptions.
- TP2 boundary integration pairs cover255/256/257/511/512/513/1025 and mixed
  output limits. Raw per-step proposals, target IDs, accepts, cursors, outputs,
  confirmed draft KV and full valid target-KV SHA256 on each rank match.
- GPU limits1/2/3/4/5/9, injected EOS, repeated requests and a batched-generation
  exception pass. Existing engine abort releases all host/target/draft/transaction
  state. Rank statuses agree; no device/NCCL failure observed in these controls.

These controls do not establish universal BF16 proposal equality. The full
serving corpus is a separate gate; see [serving-report.md](serving-report.md).

## Three-Path Operator Audit

Compare native-length serial, identically padded serial and batched execution,
with teacher-forced feedback so later different tokens cannot masquerade as an
earlier arithmetic error. BF16 differences first occur at the LM head and
feedback Q/K/V projections. Padding-only differences are zero in the B2/B3
control groups and at most0.0000305 across inspected B4 operators. FP32 controls
on the same checkpoint/state preserve proposal IDs in all nine initial cells.
No arbitrary raw-logit threshold is used as a passing criterion.

The FP32 control isolates arithmetic on the represented checkpoint and captured
state; it is not an independently trained FP32 checkpoint or full FP32 target
history. The FP64 toy control independently tests layout/mask/cursor logic.

## Preserved Serving Disagreement

The first serving run stops at short-short/c2/repeat0. All four final outputs
agree, but one proposal changes from `[323,87856,1817]` to `[323,25351,1817]`.
Target correctly accepts3 vs1 candidates on the respective actual verification
paths. This is retained as **strict proposal/acceptance parity failure**.

Same-state replay reproduces the discrepancy. Entry confirmed KV and hidden are
identical; padding-only serial is exact in BF16. The second-token serial BF16
margin is0.0625. FP32 serial and batch both select87856, margin0.0229969; the
feedback hidden discrepancy contracts from2.0 to about0.0000305. Later unforced
hidden differences follow different token inputs and are not treated as evidence
of a first-cause KV bug. Teacher-forced operator data is retained separately.

This explains this particular draft change, not every future disagreement.
The benchmark records strict signature failures separately, re-evaluates every
acceptance with the frozen greedy function and stops on final-output mismatch.
New differing cells remain pending diagnosis until explicitly audited. No
production tie-break, sampling, low-margin fallback or tolerance was introduced.

## Harness Errors Retained

The first integration process failed due to a late-bound benchmark wrapper
calling the RPC wrapper, not the drafter. A later checker compared tuple tokens
to a list and stopped despite equal values. Both logs/manifests are retained;
the latter has a new tuple/list CPU regression. Neither is a production
acceptance fix, a removed timing outlier or a passing experiment.

## Completed Corpus and Same-State Replay

The complete CPU suite passes **151 tests**. Frozen source/test hashes match the
pre-edit archive except the two explicitly changed integration files. Three
initial GPU boundary pairs have exact proposal/accept/cursor and valid KV hashes.
All 150 main and 30 fresh-process paired final outputs match; every measured
acceptance is independently recomputed. There is no claim of universal strict
proposal parity: 37 main pairs retain a different full trajectory signature.

All 37 differing main cells were replayed with full target/draft state audit and
a local frozen serial oracle from the identical live confirmed state. Across
3,023 groups this captured 87 local proposal differences in 32 cells. Every
replayed final output matches its main-run output; acceptance semantics and
cleanup checks pass. TP statuses remain consistent. Five original differing
cells do not reproduce a local mismatch in this replay. They are **not** relabeled
as exact parity or explained merely by extrapolating the first BF16 example.

Main-vs-fresh comparison retains another limitation: 25 of 60 matching full
trajectory signatures differ, all in routing-table, including frozen serial
c1 cases. All 60 final outputs match. Different execution histories may matter,
but their causal role has not been isolated. Fixed-input/shape repeated isolated
calls are deterministic; that narrower result is not cross-process trajectory
determinism. No new output/state disagreement is hidden behind a tolerance.

## Extended High-Precision Result

All 87 captured local groups completed native-serial / padded-serial / batched
teacher-forced controls in BF16 and FP32 (174 controls total). BF16 proposal
differences reproduce in87/87; FP32 proposals agree in87/87. All captured tensors
and operator summaries are finite. At the first differing BF16 top1, observed
serial margins are0 in44 groups,0.0625 in30,0.125 in7 and0.03125 in6. No margin
threshold is used to authorize a proposal, change sampling or trigger fallback.

The first recorded differing batch-shape module is `lm_head` in86 groups and
`norm` in1. Padding-only BF16 differences occur in45 groups, first observed at
`midlayer.self_attn.o_proj`;42 groups have exact native-vs-padded module outputs.
The largest teacher-forced module differences are0.625 for padding and2.0 for
batch shape in BF16, versus about0.0000458 and0.0000916 in FP32. These are
descriptive maxima, not correctness tolerances. An output hook at o_proj cannot
separate upstream attention reduction from projection arithmetic; the evidence
does not prove GEMM alone caused every change.

Captured inputs and confirmed state are identical between each comparison;
logical positions and valid masks are explicit, with independent FP64 layout
controls. Together with agreeing FP32 proposals, this supports a BF16
shape/padding-path numerical explanation for these87 *local* differences.
It does not localize the five non-reproducing original cells or establish why
fresh frozen-serial trajectories differ. No full FP32 target-history replay or
new HF validation was performed in this draft-only audit.

Raw operator summaries are indexed in `summary.json` under
`numerical_diagnostics`. The adoption conclusion stays **Keep Experimental**:
qualified state/output correctness does not close the stronger cross-history
reproduction and tail gates. Historical strict failures remain failures.
