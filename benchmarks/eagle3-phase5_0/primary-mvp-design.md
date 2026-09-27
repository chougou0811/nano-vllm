# Primary: Context-independent Target Compute-region Replay

**Design only; not implemented or authorized for implementation by this report.**
Transfer the segmented execution principle from SGLang's compiler-free
`BreakableCudaGraphBackend` (S1) and vLLM's `split_graph` (V3). Do not import
either serving stack, compiler pipeline, attention backend or padding policy.
This is an engineering transfer of an active frontier runtime technique, not
an invention of CUDA Graph or a claim that piecewise replay is new in 2026.

## Why this boundary

4.3 identifies target dispatch/rank arrival gaps; 4.4A demonstrates a same-shape
replay mechanism can reduce them on the actual two 4090s. 4.4B/C show why the
whole-target exact-context cache does not convert that into serving gains.
Attention is a small sampled compute component but imports context-sensitive
metadata into the capture key. The alternative is to leave it entirely eager
and capture only a context-independent region. This avoids needing a new
attention numerical contract or a promise that padded context is safe.

This does not eliminate all launches or speed up matrix arithmetic. Multiple
region launches, static-buffer copies and NCCL interoperation may make it fail.
That is testable within one or two bounded phases.

## Exact MVP region

One post-attention region per target decoder layer:

```text
inputs: attention output [M, local_q_heads * head_dim], residual [M, H]
  existing o_proj local matmul + existing all_reduce
  existing post_attention_layernorm(residual-add semantics unchanged)
  existing gate_up_proj -> existing SiLU/multiply -> existing down_proj
  existing down_proj all_reduce
outputs: hidden [M,H], updated residual [M,H]
```

Embedding, input norm, QKV, Q/K norm, RoPE, KV writes, FlashAttention,
feature capture, final norm, LM head, finite checks, rank status, greedy
acceptance, KV commit/rollback and scheduler all remain outside. In particular
the region neither reads context/positions/block tables nor writes KV.

Initially support **exact M=4/8/16, uniform q=4 only**, matching the already
qualified operating envelope. Every ragged/output-limit/KV-clipped case falls
back to frozen eager even if it happens to have one of these M values. A later
ragged extension is not necessary for this MVP. No context buckets, input
padding, dummy rows, dynamic capture or selective-cache policy.

Each key contains model/weight identity, layer identity, exact M, tensor widths,
dtype, device, TP group/rank/world size, strides/layout, inference mode, kernel
and software identity. Each graph is bound to that layer's immutable weights
and static storage. **max_k is absent because no captured operation consumes it**,
not because it is approximately bucketed or ignored. An FX/source audit plus
same-region-input tests must verify that claim before serving.

## Ownership and lifetime

- Engine owns an immutable per-rank region registry, allocated and warmed
  before measurement/admission. Forty layers x three M values means **120
  graph executables per rank**, not three. Memory must be measured accordingly.
- Start with separate per-entry static input/output storage and unshared pools.
  Do not assume safe cross-entry pool reuse, patch captured weight pointers,
  or dynamically overwrite executable parameters. Reject the MVP if this
  bound cannot fit the preregistered memory margin.
- Copy both incoming tensors into owned storage on the same execution stream;
  replay; consume output before the next mutation of that entry. Outputs must
  remain live until the next eager consumers and feature hooks finish. No
  request stores a mutable graph-output alias: retained features are owned
  copies under the existing runtime contract.
- In-place fused residual updates require a separate saved input for tests;
  do not accidentally compare against an already-mutated eager input.
- Single in-flight target step per TP engine. No async request pipeline,
  concurrent draft batching or overlapping graph reuse. No request-owned page
  address enters this registry.
- Initialize/capture all ranks in identical layer/M order. Warm compiled
  suboperators before capture; all collectives retain count, dtype, shape and
  communicator. The code does not replace NCCL or introduce overlap.
- Validate the complete ready registry across ranks once at initialization.
  Rank0 includes an immutable execution-mode/registry-epoch identifier in the
  existing verify RPC. Eligibility depends on the same payload on both ranks.
  No per-layer `all_gather_object`, LRU mutation, `gc.collect` or `empty_cache`.
  Existing post-forward/commit status checks are retained.
- Asymmetric initialization failure aborts the experiment. Never catch a rank's
  partially submitted collective sequence and unilaterally retry eager.
  Unsupported inputs are selected as eager on both ranks before model entry.
- After final device completion, close region registry then engine; distinguish
  allocator reservation and cuBLAS workspaces from live request leaks.

## Planned modules (no edits now)

| Boundary | Future change | What must remain unchanged |
|---|---|---|
| New `nanovllm/speculative/target_regions.py` | Region storage, pre-capture, replay, lifecycle | No request state, KV allocation or acceptance ownership |
| `nanovllm/models/qwen3.py` | Optional execution hook exposing post-attention region; original forward remains default | Exact eager operation sequence, residual dtype and weight modules |
| New opt-in runner module | Select region executor only for eligible verification | Existing ordinary runner and old exact-key graph runner remain reproducible |
| `concurrent_engine.py` | Explicit independent execution-mode selection | Default remains eager; scheduler/K/draft unchanged |
| `batched_runtime.py` callback boundary | Reuse `verification_forward` interface where possible | Dispatch phases, cursor, features and status semantics |
| New tests/benchmark modules | Region, lifecycle, frozen-path parity, paired serving | No editing or weakening old tests |

Do not modify `linear.py`, `attention.py`, BlockManager, scheduler, sampler or
draft state. The optional Qwen boundary uses the same `RowParallelLinear`
instances; capturable collectives are not new TP arithmetic. Attribution and
licenses must accompany any copied implementation; the preferred transfer is
the small design principle, not hundreds of lines of framework internals.

## Gates and benchmark plan

**Gate 0: attribution, before optimized code.** Isolate the 4.4C enabled-miss
regression using literal eager, disabled wrapper, shadow metadata, agreement-only
and enabled/no-capture controls. Keep each trace and nested timer definition.
Do not claim a cause from source inspection. Independently estimate how many
launches/time are in the proposed region. If it has negligible removable cost,
stop before building an executor. This gate is a future bounded diagnostic,
not a measurement performed in Phase 5.0.

**Gate 1: one layer, then all layers at fixed shapes.** Compare exact same
inputs/weights/stream order, TP1 small-model control then actual TP2. Record
copies, launches, GPU interval unions, rank skew, captured pool bytes and
startup time. This is not serving speedup. Use the existing Torch 2.8/CUDA12.8
environment; no dependency upgrade or external library installation is required
by the explicit-region design. Failure to capture existing operations safely
is a stop condition, not permission to replace kernels.

**Gate 2: serving.** Compare literal frozen eager, new executor disabled,
new executor enabled. Ordinary continuous batching is a context baseline,
not the primary denominator. Use fixed, preregistered workloads with short/long
prompts and outputs, mixed lengths, c1/2/4, 32/128/512-request horizons. Keep
the six historical families as regression, add a separately fixed held-out set
before measurement; do not choose it after seeing winners. Five paired repeats
per selected configuration, rotated order, key shapes warmed outside the
steady-state window, fresh-process repeats for every c. Preserve all outliers.

Report steady-state and startup-inclusive committed tokens / wall-second,
requests/s, TTFT/E2E/request TPOT/token ITL percentiles, accepted lengths,
target/draft forwards, eager-fallback fraction, copies/launches, per-rank memory,
pool occupancy, and registry entry count. Explicitly charge initialization over
each service horizon. Keep profiled mechanism runs separate from low-overhead
throughput runs; do not sum nested CPU/GPU timings.

## Correctness contract

1. Off-mode original behavior and historical CPU/GPU tests remain unchanged.
2. Same region input and exact M: compare outputs/residuals per layer, features,
   logits/top1 and deterministic repetitions. Start with bitwise expectation
   where the same backend kernels execute. Any mismatch is investigated, not
   automatically allowed by the cross-shape contract.
3. Original Phase 1.1 BF16 cross-shape contract still applies only where shapes
   differ. New disagreement requires state/operator/high-precision controls;
   no arbitrary raw-logit threshold or silent K=0 fallback.
4. Compare identical proposals/verification decisions, accepted-prefix lengths,
   pending fallback/bonus, cursors and actual valid KV. Test accept0/1/K,
   repeated rejection, EOS/max_tokens, q clipping, 255/256/257 and 511/512/513,
   mixed contexts/reordered batches and recycled physical pages.
5. Eager/replay transitions, TP state/order mismatch, intentional initialization
   failure, normal exceptions and repeated requests must fail safely without
   deadlock or stale aliases. Registry failure after partial collective submission
   is fatal for the experiment, not recoverable eager fallback.
6. Final queues/blocks/transactions/target/draft maps zero; repeated cycles show
   bounded static memory and complete release on exit.

## Success and stop rules (design thresholds, not results)

- Must pass correctness without changing its contract. Maintain at least 1 GiB
  measured physical-device headroom on **both** ranks in the worst selected run.
- Material success: >=5% end-to-end improvement over literal eager in at least
  two preregistered held-out family/concurrency cells, same direction in all five
  pairs, paired interval excluding zero; report all other cells and any >5%
  P95/P99 regression. No claim of universal improvement.
- The median whole-service saving must pay startup capture within a measured
  512-request horizon for at least one such cell. Report failure otherwise even
  if warmed target-only latency improves.
- If all-layer fixed-shape benefit is <5%, copies/boundaries erase the gain,
  memory exceeds the bound, or the serving gate fails, stop this MVP. No
  padding, larger cache, heuristics, kernel fusion or scheduler redesign to
  rescue it in the same phase.
- Expected learning even on failure: determine the frontier between removable
  host launches and dense BF16 compute/communication, and quantify the price
  of keeping context/state outside graphs. It does not invalidate 4.4A/B/C.

No speedup number is forecast. The previous full-target 2.8-3.1x is not a
promise for forty smaller region replays.
