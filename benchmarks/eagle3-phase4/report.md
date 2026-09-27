# EAGLE-3 Phase 4.1 Concurrent Serving MVP Report

## Result

The opt-in concurrent speculative serving correctness MVP passes its CPU and
small Qwen3-14B TP=2 GPU gates. It uses fixed K=3, greedy acceptance, eager target
execution, serial request-owned draft proposals and one ragged target verification
per selected batch. This validation is not a speedup or throughput claim.

The compatibility plan approved after the Phase 4.0 conflict was followed. The
frozen `BlockManager`, `ModelRunner`, ordinary `LLMEngine.step()/exit()`,
Scheduler policies, target model, Attention, TP, sampler and Phase 2/3 primitives
were not modified by Phase 4.1. New behavior is selected through
`ConcurrentLLMEngine` and wrapper/subclass modules.

## Files

Production additions:

- `nanovllm/speculative/batch.py`
- `nanovllm/speculative/block_transactions.py`
- `nanovllm/speculative/batched_runtime.py`
- `nanovllm/speculative/coordinator.py`
- `nanovllm/speculative/scheduler_adapter.py`
- `nanovllm/speculative/concurrent_engine.py`

Validation additions:

- `tests/test_eagle3_concurrent_state.py`
- `tests/test_eagle3_block_transactions.py`
- `tests/test_eagle3_batched_runtime.py`
- `tests/test_eagle3_scheduler_adapter.py`
- `benchmarks/serving/eagle3_phase4.py`
- `docs/EAGLE3_PHASE4_1.md`
- `benchmarks/eagle3-phase4/summary.json`
- this report

No Git commit was created.

## CPU Validation

The four Phase 4.1 modules contain 22 focused tests covering ownership and phase
guards, ragged packing, stale/duplicate request IDs, target token budget deferral,
prefix-cache isolation, multi-token postprocess, preemption/abort cleanup and KV
transactions. Boundary transaction cases cover 255/256/257 and 511/512/513,
accepted lengths 0..3, shared-prefix refcounts, capacity failure and idempotent
cleanup.

The complete suite passes: 110 tests, 0 failures. Historical frozen-source
assertions pass unchanged.

## GPU Validation

Environment: Qwen3-14B BF16, dedicated Qwen3-14B EAGLE-3 checkpoint, TP=2,
2 x RTX 4090 24 GiB, eager, original Scheduler, K=3, `max_num_seqs=4`, and
`max_num_batched_tokens=2048`.

- Concurrency 1 exactly matches the frozen Phase 2 persistent path for final
  output IDs and every proposal ID, actual target verification ID, accepted
  length and committed ID.
- Concurrency 2 and 4 are deterministic across repeated fixed-input/fixed-shape
  runs. The concurrency-4 output limits 3/6/9/12 finish independently.
- The 255/256/257 boundary batch is deterministic across two runs.
- A scripted test-only proposal perturbation preserves real draft execution and
  produces first-step accepted lengths 3/0/2/1 for contexts 255/256/257/511 in
  one target verification. Independent rollback and subsequent progress pass.
- Continuous replacement adds a new request after an early-finishing peer while
  another request remains active; output lengths are 3/11/5.
- EOS occurs in speculative decode rather than prefill and closes only that
  request correctly.
- A deliberate proposal exception clears all queues, maps, transactions and block
  references, and the engine enters fail-stop.
- Audit mode validates replicated captured features and identical ordered
  seq_id/generation/cursor/token/feature status from both TP ranks on every RPC.
  No NCCL, CUDA or deadlock error occurred.

Across the validation set, 221 output tokens were produced from 154 proposed
tokens, of which 131 were accepted. The observed 0.851 acceptance rate describes
only these correctness prompts and includes the deliberately rejected test; it
is not a model-quality or performance result.

Rank-0 peak allocated memory was 18.57 GiB. After engine exit, `nvidia-smi`
reported 1 MiB used on each GPU. Every successful run ended with zero waiting and
running requests, used blocks, active transactions, host request states, draft
states and rank-0 target states. TP close acknowledgements confirmed rank-1 target
cleanup as well. Speculative execution published no prefix-cache hashes.

Raw JSON, source snapshot and manifest are retained at:

`/root/autodl-tmp/eagle3-phase4-validation-20260923-4`

Two earlier attempts are retained conceptually rather than treated as performance
samples: one exited before import because `PYTHONPATH` was absent; another found
the new subclass constructor ordering bug before any request ran. The ordering
was fixed, initialization cleanup was hardened, and the final validation was run
from a fresh process with both GPUs returning to 1 MiB afterward.

## Remaining Phase 4.2 Work

- No concurrent draft batching or ragged draft KV.
- No speculative open-loop overload/fairness claim beyond the frozen original
  Scheduler's behavior.
- No speculative prefix-cache feature reconstruction.
- No cancellation API beyond fail-stop abort and preemption cleanup.
- No formal performance benchmark, multiple-repeat throughput study or latency
  conclusion.
- No Adaptive K, SLO-aware K, CUDA Graph, fusion or custom kernel work.

Phase 4.1 is ready to freeze as a correctness MVP. A later Phase 4.2 should first
benchmark serial draft cost versus batched target gains under concurrency before
choosing whether concurrent draft batching is worth implementing.
