# Phase 5.2: Reprofiling and Frontier Re-selection

2026-09-25. **Profiling and design complete; no frontier feature implemented.**
Default eager and Phase5.1 experimental replay are unchanged. No Git commit.

## Decision

**Primary: cross-request EAGLE-3 draft-step batching, with serial conditioning
in the first MVP.** Transfer the batch-at-each-autoregressive-step organization
from current SGLang EAGLE serving, not a new drafting architecture or checkpoint.

**Secondary: execution-mode-aware BF16 target MLP GEMM tactic selection.**
Transfer FlashInfer v0.7's cached, mode-aware algorithm selection, initially only
its Ada-compatible cuBLASLt BF16 path. Not an attention replacement, TinyGEMM,
quantization, kernel-writing project or expansion of CUDA Graph coverage.

These are recommendations for a subsequent approval-gated implementation.
Neither has a measured speedup in this repository. Full evidence and rejected
candidates: [report](../benchmarks/eagle3-phase5_2/report.md),
[frontier audit](../benchmarks/eagle3-phase5_2/frontier-audit.md),
[summary](../benchmarks/eagle3-phase5_2/summary.json).

## Why This Order

New unprofiled measurements, not historical inference: with replay enabled,
verification is76.08/67.49/60.28% of wall time at c1/2/4. It remains the largest
component. Draft grows13.39/19.29/24.22%; average total proposal step costs
6.10/11.16/21.44ms. Catch-up is4.20/5.96/7.43% of wall time and is included in
draft, leaving9.18/13.33/16.79% in the remaining generation portion.

Draft is **not** the largest wall-time bottleneck. Primary is chosen for the
best bounded next experiment at c2/c4: independent requests currently reread the
same draft weights through serial forwards, and existing state ownership makes
batching possible without changing target verification or checkpoints. Target
still has more total opportunity, but its remaining dispatch/rank coordination
is not explained by cheap metadata, and peer-memory collective paths are not
supported by the current driver's two-device P2P report. A broad GPU-native
runtime rewrite is not the minimum next transfer.

MLP remains about70% of non-NCCL GPU kernel service, hence the concrete backup.
However its current BF16 kernels are already optimized library kernels; a faster
Ada tactic is unproven. That route has arithmetic-contract and dependency/build
qualification risks absent from merely reusing the existing draft checkpoint.

For scale only: halving generation-only draft cost at unchanged acceptance would
predict about1.049/1.071/1.092x whole-service speedup at c1/2/4. This is an Amdahl
counterfactual, **not a forecast**. The c1 MVP will use unchanged serial execution,
and scratch packing, padding and synchronization may erase c2/c4 savings.

## Primary Provenance

SGLang commit `3e7e6529002db8125967967e0aba394986eec27d`,
[`EagleDraftWorker.draft_forward`](https://github.com/sgl-project/sglang/blob/3e7e6529002db8125967967e0aba394986eec27d/python/sglang/srt/speculative/eagle_worker_v2.py#L782)
iterates speculative steps with a batch-shaped forward input.
[`_draft_extend_for_decode`](https://github.com/sgl-project/sglang/blob/3e7e6529002db8125967967e0aba394986eec27d/python/sglang/srt/speculative/eagle_worker_v2.py#L1131)
is a separate conditioning/extension stage. We borrow that separation, not its
scheduler, tree topology, TP placement, kernels or native KV-retention rules.
Cross-request batching is an established serving technique, not a new algorithm
invented by this project. The source pin identifies the current implementation
used to inform this design, not a claim that its whole stack runs on this setup.

## Primary MVP Boundary

1. Retain Qwen3-14B BF16 TP2, original Scheduler, fixed K3, current dedicated
   EAGLE checkpoint, existing target batching and transactions. No training or
   weight download. Draft remains a single model instance on rank0/GPU0.
2. Separate preparation of each request's confirmed feature-conditioned state
   from subsequent proposal feedback. First MVP keeps conditioning/catch-up
   serial and uses the same target features/token shift as `DraftState.propose`.
3. At each subsequent autoregressive draft position, stack the current hidden
   rows and token IDs for eligible requests and call the existing draft model
   once. This is across-request parallelism; tokens within each request remain
   autoregressive. It is not DFlash/P-EAGLE multi-token parallel prediction.
4. Use transient batch scratch for heterogeneous past lengths: explicit per-row
   positions, valid-key masks and owner-to-row mappings. The current reference
   `cnets.py` forward accepts batch tensors, masks and position IDs. This is a
   source-level feasibility observation, not a tested packed implementation.
   No target padding, target graph bucketing or persistent draft-pool rewrite.
5. Batch only requests selected by the existing schedule. Do not wait for extra
   arrivals or reorder scheduler policy. Honor per-request remaining-token
   clipping and EOS. c1 and unsupported groups use the old serial path.
6. Return proposals in the original ordered request list. The unchanged target
   verification logits and greedy acceptance determine all committed tokens.
   No extra TP collective is required for draft math: only the existing proposal
   payload is delivered to the target ranks.

Proposed files (future, not modified now): a new speculative draft-batch adapter;
small delegation in `DraftState`/`coordinator.py`; opt-in engine/benchmark glue;
new tests and docs. Keep `scheduler_adapter.py`, block transactions, target
runner, attention, TP linears, sampling and frozen historical tests unchanged.
If this boundary cannot be maintained, stop and redesign rather than silently
expanding the implementation scope.

### State and Scratch Ownership

- Each row has immutable `(seq_id, generation)` identity for the call. Scratch
  rows are execution storage, never request state.
- Keep each request's confirmed KV/cursor/conditioned-token sequence separately.
  A staged conditioning result may be published only for the matching owner;
  it must not alias a reused scratch buffer.
- **Preserve Phase2 semantics:** autoregressive proposal-feedback KV is tentative
  and discarded, even for draft tokens later accepted by the target. The next
  iteration catches up using actual target features. Do not import an external
  accepted-draft-KV retention policy.
- A fallback/bonus token remains pending on the target as before. Draft cursor
  follows the existing one-token shifted conditioning invariant, not the length
  of padded batch storage.
- Padding must not advance a cursor, become an output, supply a feature or survive
  as valid KV. Gather/scatter uses explicit valid lengths, not physical width.
- On partial failure, the existing engine exception path closes all affected
  requests and releases scratch. No partially installed batch owner may survive.

### Primary Correctness Gates

CPU owner/reorder/mask/cursor tests first, then TP2 GPU audits. Compare serial and
batched drafting from identical confirmed state, including different contexts,
different accepted lengths, repeated reject and clipped K. For fixed supplied
proposals/target logits, require exact committed IDs, accepted lengths, cursors,
valid target/draft state and cleanup parity.

Exercise accept0/1/K, repeated iterations,255/256/257/1024+ contexts, EOS,
max_tokens, finish/refill/reorder, intentional exception and both target ranks.
No NaN/Inf, stale generation, aliasing, residual transaction or deadlock.
Fixed execution mode must be deterministic. New BF16 batch-shape disagreements
require state/operator/high-precision diagnostics; the Phase1 contract is not a
blanket waiver for new disagreements or changed target acceptance.

### Primary Benchmark and Stop Gates

- Fixed K3, serial-draft vs batched-draft; target eager as primary denominator.
  The existing MLP replay setting is a separately reported optional comparator,
  not a simultaneous new optimization.
- c1/2/4; at least5 repeats with rotated system order and independent fresh
  process checks. Preserve the seven current families and add independently
  fixed low/medium/high-acceptance prompts plus heterogeneous contexts. Separate
  tuning from held-out validation; never select prompts after seeing speedups.
- Measure catch-up, generation, packing/copies/masks, total draft, target,
  committed output wall throughput, tails, acceptance, memory, forward counts,
  eligible group sizes and serial fallbacks. Keep all outliers.
- Before broader integration, require >=10% measured generation-stage reduction
  at both c2/c4 after packing is included. Adoption requires >=5% serving gain
  in at least one c2/c4 aggregate with >=4/5 improving repeats and held-out gain,
  c1 regression <=3%, no >10% repeat-median P95 ITL regression and >=1GiB physical
  headroom/rank. These are prospective gates, not completed experiments.
- Stop immediately on unexplained correctness differences, memory failure,
  inability to preserve state ownership, or packing costs cancelling the saving.
  If the serving gate fails, retain the negative result; do not add adaptive K,
  new checkpoints, graphs or scheduling heuristics to rescue it.

## Secondary Backup Design

FlashInfer commit `dc5b19bb74084a9829ca204d7da5930a36bd01a4` exposes
[`mm_bf16`](https://github.com/flashinfer-ai/flashinfer/blob/dc5b19bb74084a9829ca204d7da5930a36bd01a4/flashinfer/gemm/gemm_base.py#L667)
with a cuBLASLt backend whose capability gate includes SM89. Its cached algorithm
interface separates heuristic enumeration from execution; the inspected header
uses FP32 accumulation with BF16 inputs. The September22
[Autotuner v2 design](https://flashinfer.ai/2026/09/22/autotuner-v2.html)
distinguishes eager and replay measurement objectives. It does not establish
speedup on these4090s. TinyGEMM/Blackwell-specialized paths are not the proposal.

Only if Primary is blocked or fails its bounded gate, consider one target MLP
linear at exact observed M/N/K first. Reuse existing BF16 TP weight shards;
compare current `F.linear` against a bounded set of cached cuBLASLt algorithms.
Do not fuse, quantize, change all_reduce, transpose weights repeatedly or tune
on held-out serving results. No new checkpoint or training. Qualify build/ABI
against CUDA12.8/Torch2.8 in an isolated environment before production adoption.

Tactic identity includes dimensions, strides/layouts, dtypes/accumulation,
workspace, hardware/library version and execution mode, not context length.
If tested later with frozen MLP replay, select a mode-specific tactic before
capture; never perform runtime autotuning in a measured verification.

Proposed future changes: a local linear-execution backend/adapter and benchmark
selector, not a model or TP rewrite. Validate same-input arithmetic against
current tensors and FP32 controls, layer/logit diagnostics and unchanged KV/
acceptance. No arbitrary raw-logit tolerance or silent serial fallback.
Require >=10% useful linear improvement after host overhead, then meaningful
whole-target and fresh-process serving gain; otherwise stop before broadening
the backend. Any required unsupported-architecture kernel, wholesale dependency
upgrade or numerical-contract change is a stop condition, not an implicit scope
extension. This backup has not been installed or benchmarked here.

## Project Positioning

Keep MLP replay as an **experimental opt-in with a demonstrated low-concurrency
use case**, not the main/default optimization. Startup amortization and broad
fresh-process stability remain unproven; do not reclassify Phase5.1 gates.

The strongest main line is correctness-audited concurrent speculative serving:
transactional KV and explicit owner/cursor invariants, a documented BF16
numerical contract, reproducible profiling, and isolated execution strategies.
MLP replay supplies one bounded target strategy; proposed draft batching would
complete the across-request execution story without discarding that foundation.
It is not an invention of EAGLE, batching or CUDA Graph, and not a replacement
for a production serving framework. Negative results remain part of the work.

Stop after Phase5.2. Await confirmation before either proposed implementation.
