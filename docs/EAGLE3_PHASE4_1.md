# EAGLE-3 Phase 4.1: Concurrent Speculative Serving MVP

## Scope

Phase 4.1 is an opt-in correctness MVP for Qwen3-14B BF16, TP=2, eager
execution, greedy acceptance and fixed K=3. It serializes the existing
per-request draft proposal work and performs one ragged target verification for
the selected request batch. It does not change the ordinary engine, Scheduler
policy, target model, Attention, TP, sampler, Phase 2 state semantics or the
single-request `generate_eagle3()` path.

The frozen-source tests require `BlockManager`, `ModelRunner` and the existing
methods of `LLMEngine` to remain unchanged. Therefore Phase 4.1 uses the approved
compatibility layout:

- `TransactionalBlockManager` wraps the frozen `BlockManager`;
- `ConcurrentModelRunner` adds the batched RPC through inheritance;
- `ConcurrentLLMEngine` is an explicitly selected `LLMEngine` subclass;
- the default `LLM` and ordinary `generate()`/`step()` paths are untouched.

Callers instantiate `ConcurrentLLMEngine` directly and pass `draft_path` and
`reference_path`. `SamplingParams.temperature` remains present because the
frozen API rejects zero, but Phase 4.1 never calls the sampler: draft proposals,
target verification and acceptance are greedy argmax operations.

## Execution Chain

1. The inherited `add_request()` creates a `Sequence`. The adapter registers a
   request-owned host state and places the sequence in the original Scheduler.
2. Original Scheduler prefill selection and chunking are reused. Batched runtime
   captures target features while performing the same ragged paged prefill.
3. A completed prompt receives one pending target token and one persistent
   `DraftState`. Prefix-cache lookup is disabled for speculative requests because
   cached target features do not exist.
4. Original Scheduler decode selection is reused. The coordinator serially calls
   each request's frozen `DraftState.propose()` and clips K only for output limit,
   target-token budget or KV capacity.
5. The transaction wrapper reserves each request's `[C, C+1+k)` KV rows without
   publishing a prefix hash.
6. `batched_runtime` packs `[pending_i] + proposals_i` for all requests into one
   variable-length target prefill-shaped forward. RPCs carry ordered request IDs
   and owner generations explicitly.
7. Rank 0 applies the frozen `accept_greedy()` independently to every request.
8. Both TP ranks retain exactly `1 + accepted_i` target rows/features and zero
   each rejected suffix. The host commits the matching block transaction and
   appends only committed output tokens to `Sequence`.
9. Finished requests close target and draft state independently and release all
   block references. Preemption closes both states, increments owner generation
   and reconstructs them through prefill.
10. Any step exception aborts all live Phase 4.1 requests, cleans reservations
    and marks the concurrent engine fail-stop.

## Transaction Invariants

- At most one active transaction exists per `seq_id` and generation.
- Only transaction-owned suffix blocks may be released by commit or rollback.
- Inherited/shared blocks never have their refcount decremented by rollback.
- Tentative rows never enter `hash_to_block_id`.
- Commit retains capacity for `C + 1 + accepted`; rollback restores the original
  block-table length.
- Repeated rollback/deallocation is idempotent.
- A READY target has `feature_rows == C` and one pending token:
  `len(tokens) == C + 1`.
- Draft D remains at the pre-verification cursor until the next proposal catches
  up from committed target features. Proposal-feedback KV never persists.

For cursors 255/256/257 and 511/512/513, CPU tests cover full rollback and every
accepted length 0..3. The GPU heterogeneous batch additionally uses prompt
lengths 255/256/257/511 and first-step accepted lengths 3/0/2/1.

## Ragged Shape Example

For old cursors 255, 512 and 257 with K values 3, 0 and 2:

```text
q lengths       = [4, 1, 3]
cu_seqlens_q    = [0, 4, 5, 8]
cu_seqlens_k    = [0, 259, 772, 1032]
query positions = [255..258], [512], [257..259]
proposal offsets= [0, 4, 5, 8]
```

The existing `prepare_prefill()` and paged Attention path consume the generated
Sequence descriptors. No Attention or KV kernel was added.

## Limits

- TP=2, eager, original Scheduler and fixed K=3 are enforced.
- Active requests are limited to `max_num_seqs`; this is not an overload or
  fairness implementation.
- Draft proposals are serial and the reference draft cache is not ragged-batched.
- Speculative prefix-cache hits are disabled.
- Host commit cannot recover from a distributed CUDA failure; the engine is
  fail-stop after any step exception.
- Adaptive K, SLO coupling, concurrent stochastic sampling, CUDA Graph, kernel
  fusion and performance tuning remain out of scope.

See `benchmarks/eagle3-phase4/report.md` for the validation result.
