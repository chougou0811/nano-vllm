# Phase 5.2: Post-Optimization Bottleneck Reprofiling

2026-09-25. **Complete: profiling, source audit and design only.**

Primary recommendation: **cross-request EAGLE draft-step batching**, initially
keeping catch-up serial. Secondary: **mode-aware BF16 target MLP GEMM tactic
selection**. Neither feature was implemented or benchmarked. MLP replay remains
experimental opt-in; default eager is unchanged. No Git commit.

## A. New Measurements and Protection

This phase reran the current worktree; it did not infer new bottlenecks solely
from Phase4 or5.1 numbers. Qwen3-14B BF16 TP2, fixed K3, persistent dedicated
EAGLE draft, original Scheduler, two4090s, prefix reuse disabled. Target/draft
revisions and engine config are unchanged from Phase5.1: max_model_len2048,
max_num_batched_tokens2048, max_num_seqs4, gpu_memory_utilization0.7. Eager engine
mode is retained; only the frozen opt-in MLP callback replays region graphs.

- Two fresh processes: literal eager and frozen MLP replay. c1/2/4 and all seven
  Phase5.1 families, three new measured repeats each: **126 trials,63 pairs**.
  Same fixed inputs/output limits, closed-loop refill. Workload order rotates;
  concurrency order reverses between systems. All warmups are outside measurement.
- All unprofiled trials finish before profiler installation.36 diagnostic runs
  compare event-only, minimal and rich traces over two steady verification steps
  at mixed-output/long-short. Additional short closed-loop control/profile pairs
  include prefill, replacement waves and clipped verification.60 rank traces.
- **143 CPU tests pass before and after GPU work.** Profiling preserves outputs;
  all63 system pairs have identical output IDs and speculation statistics.
  All cleanup counters/rank target states return to zero. No observed NCCL/CUDA/
  deadlock errors. Both GPU processes exit normally.
-354 preexisting protected files retain their hashes, including Phase5.1 runtime,
  production code and historical tests/artifacts. Existing dirty/untracked work
  is preserved. Only new Phase5.2 diagnostic/analysis/docs files were added.

Limit: repeats share a process per system; this is not three independent process
pairs/cell. It is a new characterization, not a new production-adoption trial.
No outliers or negative paired results were filtered.

## B. Largest Wall-Time Components

These are **unprofiled serving wall fractions**, pooled across all21 trials/c.
Catch-up is nested within draft and must not be added to it a second time.

| MLP replay enabled | c1 | c2 | c4 |
|---|---:|---:|---:|
| Target verification | 76.08% | 67.49% | 60.28% |
| Draft total | 13.39% | 19.29% | 24.22% |
| Draft catch-up subset | 4.20% | 5.96% | 7.43% |
| Draft generation excluding catch-up | 9.18% | 13.33% | 16.79% |
| Prefill | 8.88% | 11.59% | 13.85% |
| Acceptance/commit | 0.86% | 0.90% | 1.04% |
| Scheduler | 0.016% | 0.015% | 0.012% |
| Graph-ineligible decode steps | 3.97% | 6.35% | 15.50% |
| Mean target verification ms/step | 34.66 | 39.03 | 53.36 |
| Mean total draft ms/step | 6.10 | 11.16 | 21.44 |

**Target verification is still the largest component at every concurrency.**
Draft has not overtaken it. But serial per-request draft cost scales nearly
linearly; target execution amortizes across requests more effectively. Prefill
also becomes a larger fraction and is completely outside MLP verification replay.
Acceptance, scheduler and explicit metadata are not the leading wall components.

Literal eager fractions and pooled throughput: [wall-table.md](wall-table.md).
Remaining wall time contains reservation, control/synchronization, postprocessing
and driver refill. Do not add nested RPC/metadata/profile intervals to these
exclusive serving buckets.

## C. Why Replay Gains Shrink

| c | Phase5.1 historical geomean | new Phase5.2 geomean | new eager / replay pooled tok/s |
|---|---:|---:|---:|
| 1 | 1.0703x | 1.0710x | 73.33 / 79.63 |
| 2 | 1.0190x | 1.0532x | 109.19 / 115.78 |
| 4 | 1.0253x | 1.0249x | 140.90 / 144.77 |

The new c2 result is **not2%**. Cross-run variation matters; no new optimization
caused this difference. Do not treat the earlier c2 percentage as a fixed law or
retroactively change Phase5.1's failed adoption gate. Ratio of pooled throughput
and geometric mean of paired wall ratios have different weighting.

Exact matched-step accounting gives a more concrete explanation than fractions
alone (seconds summed over21 trials/c):

| c | eligible verify time saved | fallback verify time saved | total serving time saved |
|---|---:|---:|---:|
| 1 | 6.9350 | 0.0339 | 6.6067 |
| 2 | 3.2832 | -0.0738 | 3.1875 |
| 4 | 2.8954 | -0.3466 | 2.3263 |

Eligible verification itself saves about11.03/8.96/6.36% of its eager cost at
c1/2/4. The eligible portion of eager serving wall time falls from75.35% to
65.35% to52.44%. Thus both coverage/share and per-eligible-forward benefit shrink.
At c4, fallback costs an additional0.3466s and draft0.2210s across the paired
runs, offsetting part of the eligible saving. These are measured differences,
not proof that capture ordering alone caused every offset. Full signed accounting:
[paired-table.md](paired-table.md).

Fallback is not a third independent operation: it is a subset of verification
due to mixed/clipped q or unsupported batch size. At c4 it is material; it does
not justify padding/context bucketing without a separate safety/cost argument.

## D. Inside Target Verification

New rich traces attribute kernels by CPU/external-id/launch correlation,
including graph replay's outer MLP scope. Captured leaves do not execute Python
hooks again; zero leaf-hook counts do not mean GEMMs disappeared.

Rank0 verification GPU service, averaged over two rich windows/cell:

- MLP replay region: **11.68-11.85ms**, approximately69.6% of non-NCCL kernel
  service at each c. Its arithmetic work remains; replay does not fuse GEMMs.
- Attention: **0.53ms** for short-context mixed-output and **1.10ms** for
  long-short. Forty attention calls remain; it is not the leading compute cost
  for the tested contexts. This says nothing about very long-context dominance.
- QKV, o_proj and LM head are smaller compute categories; detailed measured
  values are in [operator-table.md](operator-table.md).
- New minimal traces confirm rank0 launch APIs640/641/643 ->520/521/523;
  rank1 632->512. GPU kernel counts do not fall, and84 NCCL kernels remain per
  verification on each rank. Rank0 has extra output/finite processing.
-40 graph launches cost about1.08-1.17ms of rank0 CPU API time **under profiling**.
  Copies/memsets sum roughly49/53/60us at c1/2/4 versus11/15/20us eager in the
  long-short windows. This is device service, not complete boundary overhead.

Full closed-loop traces also capture eager prefill. Its MLP kernel service grows
with rows: about14.4ms for127 rows and151ms for1916 rows, versus attention about
0.25/3.3ms. These are profiled operator sums, not unprofiled prefill latency.
They support retaining prefill as an unoptimized cost, not changing Scheduler.

## E. GPU Gaps, Rank Skew and NCCL

Profiling is visibly intrusive. Event-only target timings are typically about
39-68ms; minimal profiled spans about58-82ms; rich spans often79-117ms. These
must not replace the unprofiled target means in sectionB.

For replay long-short minimal traces, rank0/rank1 GPU idle-gap estimates are
2.52/45.75ms at c1,2.55/47.96ms at c2 and14.58/27.28ms at c4. Rank0 frequently
appears busy inside NCCL while rank1 has a long GPU gap: **a waiting collective
is not evidence of saturated communication bandwidth**.

Mean matched-collective start skew is about0.52/0.55/0.39ms for those cells;
mixed-output c4 is0.64ms. Joint NCCL residency across both ranks is about
1.18/1.58/3.08ms for long-short c1/2/4, much less than one rank's total NCCL
residency. Even joint residency is not a transport-only measurement.

There is remaining host submission/rank-arrival opportunity, but the new trace
does not isolate its pure unprofiled cause or prove that replacing NCCL will fix
it. Some profiled windows improve strongly; c4 mixed-output minimal host latency
does not. All windows remain in [profile-table.md](profile-table.md).

Current driver peer-access queries are false both ways on SM89; see
[hardware-check.json](hardware-check.json). This blocks treating peer-memory
custom collectives as an immediately qualified transfer. No NCCL configuration,
driver, kernel or synchronization behavior was changed. The historical
eligible-miss penalty is not claimed fully solved by this phase.

## F. Host Preparation and Dispatch

New event-only probes isolate the following nested target timings:

- Verification layout and Sequence descriptors together: roughly0.02-0.06ms.
- `prepare_prefill`: roughly0.12-0.17ms, including construction/device copies.
- `_status`: roughly0.19-0.27ms, including existing synchronization.
- Rank0 shared-memory pack/publication: roughly0.04-0.05ms per verify RPC.

[metadata-table.md](metadata-table.md) contains each measured cell. These are
not the entire target Python/operator dispatch cost. Rank1 `read_shm` wait can
include rank0 draft work; counting it again as extra serving overhead would
double-count the same interval. Cheap explicit metadata does not refute broader
host launch gaps, but it does argue against an MRV2 metadata rewrite as Primary.

## G. Capacity, Acceptance and Validation

Peak used KV blocks are4/8/16 at c1/2/4 out of88. Replay rank0 peak allocated
memory is18.713/18.740/18.798GiB. No cache-capacity/preemption or ownership leak
was observed. This is not evidence for higher untested concurrency.

Paired output IDs, accepted/proposed counts, target forwards and draft forwards
match across all63 serving pairs. Every timing/minimal/rich diagnostic group and
full control/profile pair preserves final outputs. Clipped steps with no draft
call remain valid zero-draft samples. An initial offline parser assumed every
decode step had a `draft_ns` key; it was corrected to retain those steps as zero,
without changing or rerunning inference or deleting any sample.

The CPU suite still contains deliberate failing numerical fixtures in stdout;
its overall result is143 tests passed. Historical strict BF16 cross-shape parity
failures remain unchanged. New profiling did not introduce a tolerance waiver.

## H. Frontier Selection

Nine current source heads,33 pinned files, additional backend source and recent
papers were reviewed. See [frontier-audit.md](frontier-audit.md) for the complete
candidate comparison and immutable links. The scan includes DFlash/DSpark,
P-EAGLE, MRV2, attention backends, async/GPU-native execution, communication, KV
services, and newly reviewed DPara/DeLS-Spec/DFlash2. No paper speedup is treated
as an expected project speedup.

**Primary: cross-request EAGLE draft-step batching.** This borrows the batched
step organization from current SGLang while reusing our checkpoint, rank0 draft
placement, ordered target batch and transaction ownership. First MVP leaves
catch-up serial and batches subsequent autoregressive generation, with temporary
masked scratch for heterogeneous past lengths. No new training/checkpoint, no
target/Scheduler redesign. It targets the growing serial repeated work at c2/c4,
not the false premise that draft is now the largest component.

**Secondary: mode-aware BF16 target MLP GEMM tactic selection.** FlashInfer's
Ada-compatible cuBLASLt interface supports a bounded per-shape candidate study
without quantization or new checkpoints. It addresses the largest target compute
category, but current library kernels may already be close to the useful limit;
there is no measured alternative-kernel win. Keep it as a backup, not a second
parallel implementation project.

Why not keep expanding Graph: the prior bounded replay already removes launch
work but leaves most compute, serial draft and prefill unchanged; higher-c gains
are diluted, and startup/coverage costs remain. Why not immediately change the
drafter architecture: that adds feature taps, checkpoint/memory qualification and
new numerical/state surfaces before testing a simpler execution-level reuse.

The full proposed modules, owner/cursor rules, correctness matrix, benchmarks,
acceptance gates and failure/stop conditions are in
[docs/EAGLE3_PHASE5_2.md](../../docs/EAGLE3_PHASE5_2.md). This is a risk-adjusted
selection, not proof that the selected feature beats every unimplemented option.

## I. Project Value and Next Boundary

1. **Keep MLP replay:** yes, as an experimental opt-in demonstrating a real
   low-concurrency use case. It is neither the default nor the main optimization.
2. **Do not promote it from these measurements:** c1 benefit is reproduced, c2
   varies and c4 is small; startup payback was not newly measured over a long
   horizon, and repeats are not independent process pairs.
3. **Strongest project line:** a correctness-audited concurrent speculative
   runtime with explicit transactional KV/owner/cursor semantics and reproducible
   diagnosis, not an accumulation of fashionable algorithms.
4. **Technical story:** transaction-safe concurrent target execution established
   correctness; numerical audits defined what equivalence means; profiling led
   to bounded MLP replay; reprofiling now motivates batched draft execution while
   preserving that state foundation. Failed controllers/cache policies remain
   honest negative results, rather than being removed from the story.

## J. Reproduction and Limitations

Raw root: `/root/autodl-tmp/eagle3-phase5.2-20260925/`. Manifests contain raw
requests/steps, model revisions, versions, commands, Git status, protected hashes,
diagnostic source, startup/warmup and cleanup. Raw traces/classified intervals
remain on the data disk, not in the repo report. Current HEAD alone does not
identify the dirty working tree.

```bash
# Run each in a fresh process, serially, from the repository root.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase52 --mode eager --output /root/autodl-tmp/eagle3-phase5.2-20260925/eager
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase52 --mode mlp --output /root/autodl-tmp/eagle3-phase5.2-20260925/mlp
```

Those executed output directories are immutable run artifacts: use new paths
for a reproduction; the driver refuses to overwrite them. Analyze with
`eagle3_phase52_analysis --root <raw-root> --output <new-report-directory>`;
`eagle3_phase52_report` renders tables and reruns CPU regression tests.

Main limits: two fresh mode processes, three within-process repeats; short/
medium-context, relatively high-acceptance inputs; profiling overhead and rank
asymmetry; no hardware-counter bandwidth roofline or pure transport isolation;
no new-feature implementation, correctness result or speedup claim. Stronger
causal/process replication remains necessary for future adoption.

New files are Phase5.2 profiling/source/analysis/report scripts and documentation
only. All historical source/tests/artifacts remain unchanged. **Stop here and
wait for confirmation before implementing Primary or Secondary.**
