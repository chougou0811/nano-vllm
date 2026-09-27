# Phase 5.0: Frontier Audit and Minimal Transfer Design

Date: **2026-09-24 UTC**. Status: **research and design complete; implementation
not started**. No production/test change, model download, dependency upgrade,
GPU run or Git commit. Stop here pending confirmation.

## Decision

**Primary: context-independent target compute-region replay.** Transfer the
segmented execution principle, not an entire framework: keep attention and
all request/KV state eager; replay one post-attention projection/norm/MLP region
per layer at exact M=4/8/16. No context bucketing, padding, dynamic capture or
serving LRU. It addresses the measured launch/rank-arrival opportunity while
avoiding context-sensitive whole-target graph keys.

**Secondary: stable request rows and incremental verification metadata.**
Transfer only MRV2-style input preparation. It is conditional on isolated
metadata preparation being material; it is not permission to rewrite the
scheduler, sampler, CPU acceptance or asynchronous execution pipeline.

These are our engineering selections, **not measured performance claims**.
Primary can fail from copies, graph-boundary launches, pool memory or startup.
Secondary can fail because metadata is already cheap at concurrency <=4.
No third parallel recommendation is made.

## Deliverables and evidence

- [Independent frontier map](../benchmarks/eagle3-phase5_0/frontier-map.md): 22 directions,
  including hardware, training, integration and two-GPU fit.
- [Project gap and bottleneck audit](../benchmarks/eagle3-phase5_0/project-gap-analysis.md):
  module-by-module A-I classification and measured/missing evidence.
- [Candidate pool](../benchmarks/eagle3-phase5_0/candidate-pool.md): 14 candidates with
  compatibility, minimal boundary, risks and interview value.
- [Decision matrix](../benchmarks/eagle3-phase5_0/candidate-matrix.md).
- [Primary design](../benchmarks/eagle3-phase5_0/primary-mvp-design.md).
- [Secondary design](../benchmarks/eagle3-phase5_0/secondary-mvp-design.md).
- [Source ledger](../benchmarks/eagle3-phase5_0/source-ledger.json): 51 typed claims,
  nine external snapshots, exact source paths/symbols/hashes, paper versions,
  public checkpoint metadata and local artifact hashes.

The initial external map was saved at **13:29:45 UTC before reopening project
source/profiling**. Local audit followed. Subsequent external source-deepening
found additional checkpoint and compiler-free graph details, explicitly marked
as a map supplement. This is a bounded public-source scan, not an exhaustive
claim to have read all 2026 work. No external repo code was executed/installed.

### Isolated source snapshots

All clones live under `/root/autodl-tmp/eagle3-phase5.0-sources/`, outside the
project. Each snapshot records `git rev-parse HEAD`, `git branch --show-current`
and `git describe --tags --always` in the ledger. Short IDs below link to full
immutable commits. A shallow describe is not a claim of the latest release.

| Project | Branch | Snapshot | Inspected transfer evidence |
|---|---|---|---|
| vLLM | main | [cccf7e1](https://github.com/vllm-project/vllm/tree/cccf7e1376ba082a2b14190a1a1fa963acb953b7) | MRV2 state/prep/sampler, staged writes, graph splitter |
| SGLang | main | [ea5baf4](https://github.com/sgl-project/sglang/tree/ea5baf4022e42ef13b089430ce2b1927a5c9d6f0) | Breakable graph backend and piecewise design |
| TensorRT-LLM | main | [8be31b0](https://github.com/NVIDIA/TensorRT-LLM/tree/8be31b0c99f3d08e3603b7621265efa7800395ba) | Overlap loop/input-copy ownership guard |
| FlashInfer | main | [28ae778](https://github.com/flashinfer-ai/flashinfer/tree/28ae778e49ab8f6104c59cc9d43efdca4343f91f) | Paged plan/run and autotuner v2 |
| Mooncake | main | [16b7ba3](https://github.com/kvcache-ai/Mooncake/tree/16b7ba3c7364eb8d36d04378bc7d28013bea1a1e) | Transfer API and KV-store integration |
| LMCache | dev | [05fc77a](https://github.com/LMCache/LMCache/tree/05fc77a0a7ababd9a7f2e343a771bc4bbc5b65cb) | Multiprocess cache server/cache-engine lifecycle |
| DFlash | main | [07ebd93](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a) | Parallel draft reference |
| DeepSpec | main | [005e03b](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2) | Qwen3-14B DFlash/DSpark features/checkpoints |
| Mirage | mpk | [bc5d696](https://github.com/mirage-project/mirage/tree/bc5d69612a345efe502a4dfb28faa672dd35dbeb) | Persistent-kernel compile/hardware dependencies |

SGLang's initial checkout encountered TLS/EOF errors; selected cited files were
recovered from pinned Git objects into a sparse tree. File hashes use those
objects. It is not represented as a fully clean/buildable checkout. DeepSpec
is also sparse. No source installation or hardware-support inference follows
from a successful clone.

Evidence labels distinguish source fact, official description/design, author
benchmark, paper claim, checkpoint metadata, local measurements and our
inference. Paper/web entries without a verified Git revision use explicit null
fields, not invented hashes. External speedups are never project forecasts.

## Answers to the 18 Questions

### 1. What are the important 2026 directions?

There are three broad movements: move control/input/output work closer to the
GPU; reduce or amortize target/draft execution with better execution regions
and parallel drafting; separate KV ownership/transport from a single engine.
Attention specialization, low precision, MoE communication, memory hierarchy,
async scheduling and disaggregated routing support these movements. The map
covers each separately instead of treating them as one universal stack.

### 2. What is genuinely new or accelerating?

New 2026 papers include [DFlash](https://arxiv.org/abs/2602.06036v2),
[DSpark](https://arxiv.org/abs/2607.05147v1),
[Blink](https://arxiv.org/abs/2604.07609v1),
[FA4](https://arxiv.org/abs/2603.05451v1) and
[ATSInfer](https://arxiv.org/abs/2607.10183v2). MRV2 and multiprocess KV services
are active architectural evolution; [FlashInfer Autotuner v2](https://flashinfer.ai/2026/09/22/autotuner-v2.html)
is a September2026 engineering development. Paging, graphs, prefix caching and
overlap are not new inventions. Their integration/ownership boundaries are
where recent systems engineering matters. Paper novelty is not deployment maturity.

### 3. Which directions are highly relevant here?

Target execution-region dispatch, metadata preparation, host/rank coordination
and (less immediately) serial draft replacement. Their order comes from local
measurements, not popularity. KV tiers, EP and P/D solve different deployment
pressures. Ledger I1-I3 records this synthesis rather than presenting it as an
external source fact.

### 4. Where is the project already aligned?

Per-request persistent draft state, packed ragged verification, heterogeneous
acceptance, transactional tentative KV, generation ownership and TP agreement
align with contemporary serving design. The numerical audit and immutable
historical negative results make its correctness claims unusually explicit
for a small research engine. This does not mean its runtime matches production
framework throughput or breadth. See N2/N3/N7/N8 and the gap audit.

### 5. Where is it behind?

Eager layer-by-layer Python dispatch, recreated metadata, synchronous CPU
result dependencies, and lack of a modular execution backend are behind
current high-performance runners. Network service, multi-worker routing,
remote KV transfer and failure recovery are absent. These are scoped omissions,
not proof the engine is wrong. At the current scale, not every missing feature
is worth implementing.

### 6. What is repeated framework functionality?

Paged KV, TP linear shards, continuous batching, hashed prefix reuse, EAGLE
accepted-prefix semantics and ordinary CUDA Graph support are established
ideas. The project did not invent these. Scheduler variants and adaptive-K
controllers should not be sold as new algorithms merely because they were
implemented locally. Negative results remain part of the record.

### 7. What still has real learning/engineering value?

The interaction between those mature mechanisms: page ownership across reject,
pending-token/feature cursors, batch reordering, collective agreement, cleanup
on exceptions and same-prefix numerical diagnostics. Reimplementation has
value when these invariants and trade-offs can be explained and tested, not
because its line count is large. Best three interview stories: transactional
concurrent serving; BF16 numerical-contract engineering; profiling-led design
and stopping unsuccessful adaptive/cache policies.

### 8. What is the largest measured bottleneck?

**Target execution wall time, including dispatch/rank-arrival effects**, not
automatically attention, NCCL transport or Python metadata. Phase4.2 assigns
80.82/71.40/61.10% of serving time to verification at c1/2/4. Phase4.3 finds
roughly640 GPU launches and84 collectives per verification, with MLP about69%
of *non-NCCL GPU service*, a different denominator. Phase4.4A's target-only
replay demonstrates a large removable execution overhead, but does not locate
every idle interval. Sources P42/P43/P44A.

The graph-enabled eligible-miss regression is an additional unresolved
path-specific problem, not evidence that all target compute is slow. Its
operator/stack root cause remains unknown (P44C).

### 9. Which external technique addresses it most directly?

Segmented execution around dynamic operations, illustrated by
[SGLang's breakable backend](https://github.com/sgl-project/sglang/blob/ea5baf4022e42ef13b089430ce2b1927a5c9d6f0/python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py#L59)
and [vLLM graph splitting](https://github.com/vllm-project/vllm/blob/cccf7e1376ba082a2b14190a1a1fa963acb953b7/vllm/compilation/backends.py#L549).
Our proposed explicit region is smaller than either framework integration and
does not require importing its compiler/runtime. This is a supported design
hypothesis, not a measured win.

### 10. Was a better previously unconsidered direction found?

Yes: compiler-free **context-independent segments**, rather than another
whole-target exact-key cache policy, provide a cleaner minimum migration point.
Also, DeepSpec has public Qwen3-14B DFlash/DSpark checkpoints (D2/HF1/HF2), so
parallel drafting cannot be dismissed as unavailable. The latter is genuinely
new and attractive, but not currently better on target-dominant profiling,
rank0 memory and state-adapter scope. Availability is not compatibility.

### 11. What does not reasonably fit two4090s?

Blink's SmartNIC architecture, GH200 C2C scheduling assumptions, FA4 Blackwell
fast paths and separate full BF16 Qwen3-14B P/D replicas fail this hardware or
capacity envelope. Whole MPK migration is too broad and its exact Ada TP2 path
is unqualified; a native fallback in source prevents claiming categorical Ada
impossibility. RDMA/MoE EP benchmarks do not establish dense SYS-link TP gains.

### 12. Which options require training?

Neither selected direction does. New drafting architectures need trained
weights; existing public Qwen3-14B DeepSpec weights can avoid training from
scratch. Training to recover domain/target-revision compatibility is a separate
expensive study, not included here. Low-precision conversion may need
calibration/quality validation even without training. Do not incorrectly
classify every new drafter as requiring a fresh training run.

### 13. What can be minimally transferred?

A post-attention compute executor (C1), verification metadata builder (C2),
an attention plan/run adapter (C7), or one BF16 GEMM backend (C8) have identifiable
boundaries. A new drafter adapter (C4) is structurally separable but has larger
feature/state/memory implications. Replacing a whole serving stack is not a
minimal transfer. Only C1/C2 pass the present prioritization.

### 14. What is Primary?

One opt-in execution path for exact-M post-attention `o_proj -> residual/norm ->
MLP`, including the existing two all_reduces per layer. Attention/QKV/RoPE/KV,
feature hooks, final norm/head, status, scheduler, draft and acceptance remain
eager/unchanged. Pre-capture a bounded immutable registry, no hot-path capture.
Forty layers times three M values means **120 executables per rank**. Static
memory and copy cost must be counted, not hidden behind the word piecewise.

### 15. Why over the other candidates?

It combines the strongest local mechanism evidence, unchanged checkpoints and
math, actual CUDA/NCCL hardware, high state-code reuse and a falsifiable boundary.
It avoids a new attention mask contract, new draft architecture, whole async
pipeline and unsupported hardware. This is a risk-adjusted next experiment,
not proof it beats every alternative. The matrix lists evidence weaknesses too.

### 16. What is Secondary and when does it replace Primary?

[MRV2-style stable rows](https://github.com/vllm-project/vllm/blob/cccf7e1376ba082a2b14190a1a1fa963acb953b7/docs/design/model_runner_v2.md)
and incremental metadata updates for verification only. Switch if Primary fails
its bounded gate or diagnostics point specifically to input preparation, and
preparation is >=5% of literal-eager serving in two preregistered cells. Keep
CPU acceptance barriers, original schedule and transaction ownership. If this
opportunity is too small, stop rather than manufacture a backup implementation.

### 17. What is the smallest Primary MVP and its gates?

First diagnose the old eligible-miss overhead with literal eager, disabled
wrapper, shadow metadata, agreement-only and enabled/no-capture controls.
Then one-layer exact-input tests, all-layer fixed shapes and finally five-pair
serving comparisons. Use uniform q4/M4,8,16 only, eager fallback otherwise.
No context padding or dynamic graph policy. All same-shape mismatches need
diagnosis; existing BF16 cross-shape contract is not a universal waiver.

Require unchanged state/TP/numerical correctness, bounded memory with >=1GiB
physical headroom per rank, >=5% serving improvement in two held-out cells
with all five pairs agreeing, and startup paid within a measured512-request
horizon for at least one cell. These are design thresholds, not empirical
results. Report all regressions/outliers. Stop on failed correctness, memory,
fixed-stage or serving gates; do not add padding/fusion/heuristics to rescue it.
Full ownership, files, benchmark and failure rules are in the primary design.

### 18. How should the project be positioned after success?

A correctness-audited concurrent speculative inference research runtime on
constrained TP hardware, with an evidence-based modular target execution
optimization. Describe exactly which serving regimes benefit and how much
startup/memory they cost. Not a replacement for vLLM, not invention of EAGLE,
and not universal production readiness. If the MVP fails, retain the bounded
negative result and do not adopt that success positioning.

## Preservation and validation

Local HEAD remains `0ce2c1e054035766f2ee27027ffc309d58ad8c0c`; existing dirty and
untracked files were left in place. HEAD alone does not identify this worktree.
The ledger hashes121 local source/artifact files;107 source/test/benchmark-code
hashes match Phase4.4C's preservation inventory exactly. No historical result
was edited. The older pre-4.4C snapshot differs at its own final4.4C document,
as explained in the ledger; this is not a Phase5.0 edit.

This phase validates documents, source pins/symbols, JSON and local preservation.
It does not rerun CPU/GPU correctness or performance. Historical139 complete CPU
tests /30 focused tests are **historical** evidence, not new results from this
audit. All future design gates are unexecuted. No performance improvement is
claimed. Await confirmation before any implementation or profiling phase.
