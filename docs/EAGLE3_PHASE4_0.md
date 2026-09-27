# EAGLE-3 Phase 4.0: Concurrent Speculative Serving Design

Status: architecture audit and design only. No production code was changed and
no GPU benchmark was run in Phase 4.0. The audited tree is Git checkpoint
`0ce2c1e` plus the frozen, uncommitted Phase 3/3.1 worktree. Phase 3.1 results,
prior/controller behavior and held-out data remain unchanged.

The Phase 4.1 MVP should be opt-in, greedy, eager, fixed K=3, TP=2, and use the
existing EAGLE-3 checkpoint. Its purpose is correctness under continuous batching,
not a performance claim. Adaptive K, SLO coupling, stochastic speculative
acceptance, CUDA Graph, kernel fusion, new attention kernels and P/D
disaggregation remain out of scope.

## A. Current Architecture Audit

### Ordinary continuous batching

The existing non-speculative path is already multi-request:

1. `LLMEngine.add_request()` creates one `Sequence` and appends it to
   `Scheduler.waiting` (`engine/llm_engine.py:43-51`, `engine/scheduler.py:22-23`).
2. `LLMEngine.step()` calls `Scheduler.schedule()` and receives one homogeneous
   prefill or decode batch (`llm_engine.py:53-59`).
3. Prefill admits waiting sequences, allocates per-sequence block tables, and
   observes `max_num_seqs` and `max_num_batched_tokens` (`scheduler.py:25-55`).
   Decode selects up to `max_num_seqs`, ensures one-token KV growth, and can
   preempt a resident request if no block is available (`scheduler.py:57-73`).
4. `ModelRunner.call("run", seqs, is_prefill)` broadcasts the same ordered RPC
   to every TP rank (`model_runner.py:61-93`). Rank-local model execution is
   synchronous.
5. `prepare_prefill()` flattens different query lengths and constructs
   `input_ids`, `positions`, `cu_seqlens_q`, `cu_seqlens_k`, `slot_mapping`, and
   padded block tables (`model_runner.py:133-174`). This is existing ragged,
   paged prefill support.
6. `prepare_decode()` builds a q=1 row per sequence with vector
   `context_lens`, `slot_mapping`, and padded block tables
   (`model_runner.py:176-192`). This is existing batched decode support.
7. `Attention.forward()` writes KV through `slot_mapping`; prefill uses
   `flash_attn_varlen_func`, while q=1 decode uses
   `flash_attn_with_kvcache` (`layers/attention.py:59-75`).
8. Rank 0 samples one output per sequence. `Scheduler.postprocess()` advances
   cached cursors, appends one token, handles EOS/length, releases finished
   block tables, and leaves unfinished sequences in `running`
   (`model_runner.py:218-224`, `scheduler.py:81-92`). The next engine step can
   admit newly waiting requests or service running requests.

`ParallelLMHead` already selects the last row of every packed prefill sequence
using `cu_seqlens_q`; it returns one logit row per request in ordinary prefill
(`layers/embed_head.py:56-65`). TP attention/KV heads remain sharded, while
post-row-parallel hidden states are replicated.

### Current EAGLE-3 path

The EAGLE path is a separate, synchronous API rather than a Scheduler mode:

1. `LLMEngine.generate_eagle3()` calls `speculative.session.generate()` directly
   (`llm_engine.py:64-72`). It never calls `add_request()` or `step()`.
2. `generate()` rejects a non-eager runner and any engine with queued/running
   work or used KV blocks (`speculative/session.py:22-27`).
3. It creates one private `Sequence`, leases all pages needed for prompt plus
   maximum output, and creates one rank-0 `DraftState` owned by that sequence
   (`session.py:53-60`). These pages are not prefix-cache published.
4. Ordered `eagle3` RPCs call `runtime.dispatch()` on every TP rank. `begin`
   replaces `runner._eagle_state` with one dictionary; `prefill` captures three
   target intermediate features and seeds one pending target token
   (`runtime.py:74-88`).
5. `DraftState.propose()` catches its persistent draft cache up from D to C using
   verified target features, produces K greedy proposal IDs, and discards all
   proposal-feedback KV. Only target-feature-conditioned draft KV persists
   (`draft_state.py:22-68`).
6. `verify` constructs `[committed tokens + K proposals]` and calls `_forward`
   from the single target cursor C. `_forward` creates a transient single
   `Sequence`, invokes `prepare_prefill([seq])`, switches LM-head selection to
   all verification rows, and captures target features (`runtime.py:34-71,
   89-95`). Thus target verification is a q=K+1 causal prefill-shaped forward.
7. Rank 0 applies `accept_greedy()` to one proposal list and one target-ID list.
   It commits the longest matching prefix and one fallback/bonus target token,
   with EOS/output-limit clipping (`acceptance.py:13-32`).
8. `commit` keeps `1 + accepted` target KV/feature rows, zeros the rejected
   physical suffix on every rank, advances the single target cursor, and clears
   `tentative` (`runtime.py:96-105`). `DraftState.committed()` checks the D/C/token
   relationship; the next iteration catches up only newly verified rows.
9. `close` zeros the entire private target lease, drops the sole rank-local
   state, closes draft state, and deallocates the private host block table in
   nested `finally` blocks (`runtime.py:106-109`, `session.py:128-139`).

The Phase 2 cursor contract remains authoritative. Let C be valid target KV and
target feature rows, D be target-feature-conditioned draft rows, and T be
committed token IDs. Before a nonterminal proposal, `D <= C` and `len(T)=C+1`;
the last token is pending target input. After accepting `a`, target keeps
`1+a` verification rows and advances to `C'=C+1+a`; draft remains at the old C
until the next catch-up. Proposal-feedback draft rows never become committed.

## B. Single-Request Assumptions Found

| Assumption | Concrete source evidence | Consequence |
|---|---|---|
| Engine must be idle | `session.generate`: `engine.is_finished()` and `bm.used_block_ids` guard at lines 24-26 | EAGLE cannot coexist with queued or running requests. |
| Session owns one request | One local `Sequence`, one `DraftState`, one lease, one while loop (`session.py:53-76`) | No per-request lifecycle inside Scheduler. |
| One target state per rank | `runner._eagle_state = dict(...)` replaces a scalar attribute (`runtime.py:76-80`) | A second request would overwrite the first on every rank. |
| Target forward is batch=1 | `_forward` calls `prepare_prefill([seq])` (`runtime.py:35-49`) | Existing ragged target support is unused. |
| Proposal tensors are batch=1 | Draft inputs use `.unsqueeze(0)` and logits index `[0,-1]` (`draft_state.py:35-49`) | No independent request dimension or boundaries. |
| One accepted length | `accept_greedy()` returns one scalar `accepted`; commit receives one scalar (`acceptance.py:13-32`, `runtime.py:96-103`) | Different A/B/C/D outcomes cannot share a transaction. |
| C/features/tentative are global | Scalar `cached`, one feature tensor, one tentative tensor in `_eagle_state` | Cursors and rollback cannot differ per request. |
| Block table is one list | `_zero` indexes `state["blocks"]`; begin accepts one `blocks` list (`runtime.py:12-19,76-79`) | Physical suffix clearing has no request offsets. |
| Full-capacity private lease | `Sequence([0] * (prompt+max_tokens))`, then `allocate(lease,0)` (`session.py:53-60`) | Correct but incompatible with normal incremental admission/prefix-cache ownership. |
| Output and EOS are scalar | One `seq`, one `result`, one EOS check, one output record (`session.py:75-151`) | Requests cannot finish independently. |
| Cleanup is global | `close` zeros all leased blocks and assigns `_eagle_state=None` | Cannot remove one request while retaining others. |
| Draft model is shared mutable rank-0 state | One `engine._eagle_draft`; reference model exposes global reset/mask state | Per-request caches must remain outside the model and proposal calls must be serialized or shape-grouped safely. |
| Sequence worker serialization is minimal | `Sequence.__getstate__()` omits `seq_id`, status, sampling fields, and arbitrary metadata (`sequence.py:72-82`) | TP workers cannot key state from pickled `Sequence`; request IDs must be sent explicitly. |
| Existing postprocess emits one token | `zip(seqs, token_ids)` and one `append_token()` (`scheduler.py:81-92`) | Multi-token speculative commits need a separate postprocess path. |
| Decode reserves one KV row | `num_scheduled_tokens=1` and `may_append()` once (`scheduler.py:57-70`) | q=K+1 verification can cross an unreserved page boundary. |

Existing `BlockManager` is already per-sequence and ref-counted, and existing
target attention already supports multiple paged sequences. What is missing is
transactional multi-row reservation/rollback and per-request speculative state,
not a new paged-attention design.

Two additional constraints come from the audited code:

- Prefix-cache hits skip complete prompt blocks in target execution. No target
  intermediate features are cached with those blocks. EAGLE draft initialization
  therefore cannot use a prefix-cache hit without separately reconstructing the
  missing features. Phase 4.1 should disable prefix-cache hits for speculative
  requests while leaving ordinary requests unchanged.
- The pinned reference draft accepts a dense batch tensor and one rectangular
  `past_key_values` length (`cnets.py:600-630`). Different D/C values cannot be
  made into one dense call merely by concatenating requests. Exact-shape grouping
  is safe; true ragged draft attention would be a later optimization and is not
  required for a correctness MVP.

## C. Required State Changes

Do not put GPU tensors or draft cache tuples into `Sequence`; doing so would
expand its TP pickle contract and duplicate rank-0-only state. Introduce an
engine-owned `SpeculativeCoordinator` keyed by the stable host `seq_id`.

Host/rank-0 request state:

```text
SpeculativeRequestState
  request_id / seq_id / owner_generation
  phase: PREFILL | READY | PROPOSING | VERIFIED | COMMITTING | FINISHED | FAILED
  fixed_k
  draft_state: existing DraftState (D, confirmed past, conditioned token digest)
  proposed_token_ids
  requested_k / actual_k
  accepted_length / committed_token_ids / fallback_token
  transaction: old_C, verify_end, keep_end, original_block_count, new_block_ids
  finished_reason / error / closed
```

Rank-local target state on every TP rank:

```text
TargetRequestState
  seq_id / owner_generation
  target_cursor C
  committed token IDs or a committed-token checksum
  committed captured target features [C, 3 * hidden_size]
  tentative feature rows [q_i, 3 * hidden_size] or None
  audit flag
```

The host `Sequence` remains the canonical committed token list, status, output
limit and block table. The target map is rank-local and TP-synchronous. The
rank-0 host coordinator owns draft states and transient proposal/results. Every
RPC includes the ordered `seq_ids`; it must reject duplicate IDs, missing owners,
generation mismatches, phase mismatches and rank-order disagreement.

Per-request invariants:

- READY: `D <= C`; nonterminal `len(Sequence.token_ids)=C+1`; no tentative state.
- VERIFYING: `old_C=C`; q_i is `1 + len(proposals_i)`; physical capacity covers
  `[old_C, old_C+q_i)`; committed cursors remain unchanged.
- VERIFIED: target IDs/features are partitioned by the request's packed offsets;
  no host token or committed cursor has changed.
- COMMITTED: `C'=old_C+1+a_i`; only features/KV `[old_C,C')` survive; draft D
  remains at old C; Sequence gains accepted proposals and optional fallback/bonus.
- CLOSED: no target map entry, draft `past is None`, no transaction-owned blocks,
  and a finished/cancelled Sequence owns no block references.

The `owner_generation` protects against stale asynchronous-looking cleanup even
though execution is currently synchronous. A request's draft state, target state,
proposal buffers and block reservation must never be reachable through another
request's key.

## D. Proposed Concurrent Speculative State Machine

```text
add_request
    -> WAITING/PREFILL
Scheduler chooses a prefill batch
    -> batched target prefill + per-request feature append
    -> partial prompts return to WAITING
    -> completed prompts commit first target token and become READY/RUNNING

Scheduler chooses a decode batch
    -> reserve per-request tentative target rows
    -> PROPOSING (rank 0, one DraftState per request)
    -> pack proposals and boundaries
    -> VERIFYING (one ragged target batch on both TP ranks)
    -> compute and validate all per-request greedy Acceptance values
    -> COMMITTING (one vector transaction RPC)
       * keep/zero each target suffix independently
       * append only each request's committed output IDs
       * commit/release each request's block reservation independently
       * advance draft ownership check, but never retain proposal-feedback KV
    -> READY for unfinished requests
    -> FINISHED/CLOSED for EOS or max_tokens
    -> next Scheduler step admits new work and/or forms the next decode batch
```

For the example A/B/C/D with K=3 and accepted lengths 3/0/2/1, all four requests
share one target verification forward with q=4 each. Their target keep counts are
4/1/3/2. Each gets its own bonus/fallback decision, target cursor, suffix clear,
draft catch-up debt and output list. There is no batch-wide accepted scalar.

All acceptances must be calculated and validated before any host Sequence is
mutated. After successful rank commit, host commits are deterministic and
prevalidated. A pre-commit exception rolls every participant back to old C. A
failure during or after a distributed commit is fail-stop for the engine: attempt
best-effort cleanup, mark it faulted, and do not continue with potentially
rank-divergent state.

Preemption invalidates target KV, target feature history and draft cache together.
The request closes both speculative states, returns to ordinary WAITING/PREFILL,
and reconstructs from the prompt/committed tokens. Phase 4.1 must not retain D or
C across a BlockManager preemption.

## E. Batched Proposal and Verification Design

### Logical batch records

Use immutable records for a scheduled step:

```text
ProposalEntry(seq_id, old_C, D, remaining, requested_k, actual_k,
              proposals, q_offset, q_length, block_transaction)
SpeculativeBatch(entries, packed_input_ids, positions,
                 cu_seqlens_q, cu_seqlens_k, block_tables, slot_mapping,
                 proposal_offsets, target_token_budget)
```

`actual_k_i = min(K, remaining_i - 1)` preserves the Phase 2 terminal q=1 drain.
The representation also permits a shorter proposal ending at a draft EOS, but
Phase 4.1 should preserve Phase 2 behavior and not early-truncate on draft EOS.
EOS is authoritative only after target verification. Variable lengths are still
required because output limits differ.

### Draft proposal

The rank-0-only draft model remains unsharded because target features are already
replicated and putting it on every rank would waste memory and require proposal
agreement collectives. The safe MVP API is logically batched:

1. Build all `ProposalEntry` objects in scheduler order.
2. Group requests only when `(D, C, actual_k)` gives rectangular confirmed past,
   catch-up and proposal shapes.
3. For a compatible group, concatenate each layer's confirmed past along batch,
   stack `[D:C]` features and shifted token IDs, run K autoregressive draft steps,
   then split confirmed caches with owned copies per request.
4. Use a singleton group (equivalent to existing `DraftState.propose`) for
   incompatible shapes. This is expected for heterogeneous contexts and is
   correct, though not maximally fast.

An even smaller first patch may implement the same logical API with sequential
per-request proposal calls, then add exact-shape grouping without changing any
state or target semantics. It must be described as serial draft proposal plus
batched target verification, not as a single physical draft batch. Padding
different past lengths is not an MVP shortcut: the pinned reference computes one
past length for the dense batch, so padding would require attention-mask and
position changes that need an independent numerical audit.

### Target verification

This can directly reuse existing variable-length paged prefill machinery. For
request i:

- query tokens are `[pending_i] + proposals_i`, so `q_i=1+k_i`;
- `input_ids` is the concatenation of those query segments;
- `positions` concatenates `range(C_i, C_i+q_i)`;
- `cu_seqlens_q` is the prefix sum of q_i;
- `cu_seqlens_k` is the prefix sum of `C_i+q_i`, matching existing
  `prepare_prefill()` semantics with paged block tables;
- `block_tables` is the existing padded `[batch,max_blocks]` table;
- `slot_mapping` maps every packed query row to its request's physical page/offset;
- `context_lens=[C_i]` is retained as host metadata/audit output but is not passed
  to attention on this prefill-shaped path. Existing q=1 decode is the path that
  consumes `context_lens`;
- `proposal_offsets`/`cu_seqlens_q` partition flattened logits and captured
  features back to requests.

Hooks capture the same three replicated target feature points over the packed
rows. As in Phase 1, LM-head prefill selection must be disabled only for the
explicit verification logits so all q_i rows are returned. No QKV, RoPE, TP or
Attention implementation change is required.

Target prefill also needs a batched feature-capture path. It appends each packed
prompt chunk's feature slice to the matching target state. Only requests whose
prompt completes use the final-row target argmax as their first pending output.
This fills a real gap in the current ordinary prefill API; it is not a new model
operation.

## F. KV Commit/Rollback Invariants

For request i, let old C be committed target rows, `q=1+k`, `verify_end=C+q`,
accepted length a, and `keep_end=C+1+a`.

1. Before verification, reserve enough private writable capacity for physical
   positions `[C,verify_end)`. A reservation records the original block-table
   length and every newly allocated block ID. It does not publish hashes.
2. Existing shared prefix blocks are immutable. Prefix-cache sharing only covers
   complete blocks; a writable partial block must be privately owned. Phase 4.1
   disables speculative prefix hits because feature history is absent.
3. Verification may write all q rows, but C and Sequence tokens remain committed
   at their old values until acceptance for every request is validated.
4. Commit keeps exactly `[C,keep_end)`. Every rank zeros
   `[keep_end,verify_end)` using that request's block table before reporting
   success. Different requests use independent ranges in the same commit RPC.
5. Retained target features are exactly the first `1+a` tentative rows. Rejected
   suffix features are dropped. Target cursor becomes `keep_end`.
6. Append accepted proposal IDs and, unless acceptance ended on EOS/max_tokens,
   the target fallback/bonus ID. Only IDs actually appended to Sequence are
   output tokens and receive timestamps.
7. Publish a prefix hash only for a block whose every row is committed and whose
   token IDs are already in Sequence. Never hash tentative or rejected rows.
8. Trim the block table to capacity required by the surviving committed state
   (and a nonterminal pending token). Pop only transaction-owned extra blocks,
   decrement each ref count once, and return blocks whose ref count reaches zero.
   Never decrement an inherited/shared prefix block during rollback.
9. Draft confirmed KV remains at D=old C. Proposal-feedback rows, including rows
   for accepted token IDs, are discarded. On the next turn, catch up `[D,C')`
   from retained verified target features and real shifted token IDs.
10. Accepted EOS has no fallback and can leave `len(tokens)=C'`; fallback/bonus
    normally leaves `len(tokens)=C'+1`. A terminal pending output need not receive
    KV because the request is immediately closed.
11. A finished request first completes rank commit/clear, then closes draft and
    target state, then deallocates all remaining host block references and leaves
    `running`. Other batch members remain live.

BlockManager therefore needs explicit `reserve_rows`, `commit_rows`, and
`rollback_rows`-equivalent operations or a transaction object implementing them.
The current `may_append()` cannot express multi-row tentative growth. Unit tests
must assert free/used sets, table lengths, ref counts, hash publication and zeroed
suffix slots after every outcome.

Exception policy:

- Proposal failure before reservation/verification closes or fails only affected
  host draft work before the batch is submitted.
- Any exception after reservation but before distributed commit rolls every batch
  reservation back to old C and clears tentative target state on all ranks.
- Host validation happens before rank commit. If rank commit/status disagrees,
  stop the engine rather than serving from uncertain KV.
- `LLMEngine.exit()` calls `coordinator.close_all()` before runner shutdown.
  Cleanup must be idempotent so request finish, cancellation, exception and exit
  can meet without double-decrementing blocks.

## G. Scheduler Integration Design

The existing Scheduler is a usable selection/admission base, but not a complete
speculative scheduler. Reuse its waiting/running queues, prefill selection,
`max_num_seqs`, original policy order, preemption behavior and BlockManager.
Wrap it with an opt-in speculative adapter/coordinator; leave all existing policy
classes and the speculation-off default untouched.

For a prefill batch, execute feature-capturing target prefill and then the
existing scalar postprocess semantics. For a decode batch, use the selected
sequences but replace q=1 model execution/postprocess with proposal, reservation,
ragged verification, vector acceptance and multi-token postprocess.

Budget rules:

- `max_num_seqs` counts requests, unchanged.
- For target verification, `max_num_batched_tokens` counts `sum_i(1+k_i)`, not
  the number of emitted tokens. Select/reduce actual k so this sum fits; never
  silently exceed the configured target query budget.
- Draft catch-up/proposal token work is recorded separately. It consumes compute
  but is not target-attention token budget.
- Physical KV reservation covers the tentative query rows. If K cannot be fully
  reserved, either reduce actual k with an explicit reason/metric or defer that
  request; q=1 remains the correctness fallback. This is capacity clipping, not
  Adaptive K and not a low-margin fallback.

Accepted lengths differ, so per-step output progress differs. MVP fairness remains
one verification opportunity per selected request, not equal emitted tokens. Do
not add an acceptance-aware priority policy in Phase 4.1. Record verification
service gaps and committed progress so later work can evaluate fairness.

Requests that finish leave independently; the next ordinary schedule call may
prefill newly waiting requests and later include them in decode. A preempted
speculative request must trigger the reset described above before re-prefill.

One existing limitation must not be hidden: original `Scheduler` restores the
selected decode prefix to the front of `running` (`scheduler.py:58-73`). When
resident requests exceed `max_num_seqs`, later residents wait until earlier ones
finish; policy schedulers rotate, but they are frozen and out of the Phase 4.1
MVP. Initial correctness/performance workloads should keep active concurrency
at or below `max_num_seqs`. This yields real continuous replacement as requests
finish without claiming general overload fairness. A separate scheduler-policy
integration is required before open-loop speculative overload claims.

## H. Minimal Code-Change Plan

Phase 4.1 should make these focused changes:

| File | Function/class | Minimal purpose |
|---|---|---|
| new `nanovllm/speculative/batch.py` | request/batch/transaction dataclasses | Explicit ownership, packed offsets and phase validation. |
| new `nanovllm/speculative/coordinator.py` | `SpeculativeCoordinator` | Register requests; capture prefill; propose; accept; commit; finish/preempt/error cleanup. |
| new `nanovllm/speculative/batched_runtime.py` | batched begin/prefill/verify/commit/close dispatch | Per-rank target-state map and ragged feature/logit partitioning; keep Phase 1 runtime as reference. |
| new `nanovllm/speculative/scheduler_adapter.py` | opt-in adapter | Reuse original selection, enforce speculative token budget, perform multi-token postprocess and lifecycle cleanup. |
| `nanovllm/engine/llm_engine.py` | opt-in serving setup, `add_request`, `step`, `exit` | Route only enabled greedy requests through coordinator; defaults and `generate_eagle3()` unchanged. |
| `nanovllm/engine/model_runner.py` | one batched EAGLE RPC entry | Dispatch ordered request IDs on all TP ranks; no normal `run()` change. |
| `nanovllm/engine/block_manager.py` | transactional multi-row reserve/commit/rollback | Correct per-request tentative page ownership and ref-count release. |
| new `tests/test_eagle3_concurrent.py` | CPU/state tests | State machine, heterogeneous acceptance, blocks, cleanup and scheduler flow. |
| new `benchmarks/serving/eagle3_phase4.py` | GPU correctness/benchmark harness | Added only after CPU tests; raw concurrent timing and manifests. |

`Sequence` can remain unchanged in the MVP because engine/coordinator maps use
host `seq_id`, while TP RPCs send IDs explicitly. If later mixed speculative and
ordinary requests require a persistent mode flag on Sequence, add only a compact
serialized enum; never serialize draft KV/features.

Files that should remain frozen in Phase 4.1:

- `speculative/draft_state.py`, `draft.py`, `acceptance.py`, current `runtime.py`;
- `speculative/controller.py` and `prior_controller.py`;
- `engine/policy_scheduler.py` and `progress_scheduler.py`;
- `layers/attention.py`, all TP linear/embedding layers, `models/qwen3.py`;
- `layers/sampler.py` and sampling parameters;
- Phase 2/3/3.1 tests, reports, summaries, prior and held-out artifacts.

The existing single-request `generate_eagle3()` stays as the Phase 2/3 reference
oracle. New batched code should not generalize it in place until parity tests pass.
No Attention change is justified by this audit: target verification maps onto
existing varlen paged prefill. No TP change is justified: rank0 proposes and all
ranks verify exactly as before.

## I. Correctness Test Matrix

CPU/state tests before full-model GPU tests:

| Area | Required cases and assertions |
|---|---|
| State machine | concurrency 1/2/4; distinct owners; illegal phase/owner/order rejected; idempotent close. |
| Acceptance vector | A/B/C/D accepted 3/0/2/1; full, partial, accept 0, repeated rejection; each output/cursor independent. |
| Limits/EOS | different max_tokens; q=1 terminal drain; accepted EOS; fallback EOS; one request finishes while peers continue. |
| Blocks | 255/256/257 and 511/512/513; different rollback lengths; tentative extra block released; surviving partial block retained; full committed block hashed only after commit. |
| Ref counts | shared ordinary prefix never modified/decremented by another request rollback; freed block reuse; no duplicate free IDs. Speculative prefix hits remain disabled. |
| Failures | exception before verify, after reservation, before commit, forced rank-status disagreement; every request reaches CLOSED/FAILED with no retained transaction blocks. |
| Preemption | deallocation closes target/draft state; re-prefill reconstructs C/D/features; stale generation rejected. |
| Budget | `sum(q_i)<=max_num_batched_tokens`; batch count <= max_num_seqs; clipping reasons explicit. |

GPU correctness gates, Qwen3-14B BF16 TP=2 eager:

1. Concurrency=1 must replay the Phase 2 persistent fixed-K=3 path exactly for
   proposals, actual parallel target IDs, accepted lengths, commits, cursor/state
   status and final outputs on the same fixed inputs/shapes.
2. Concurrency=2/4 with different prompt lengths and output limits; fixed replay
   must be deterministic for the same batch shapes and execution mode.
3. Scripted proposal/target controls cover full/partial/zero acceptance, repeated
   rejection and heterogeneous accepted lengths in one verification batch.
4. EOS and max-token cases include an early-finishing request while others run
   for multiple further iterations.
5. Boundary prompts 255/256/257 plus longer contexts cross physical target pages;
   hash valid KV and zeroed suffixes per request/rank at transaction boundaries.
6. Verify exact rank agreement for ordered IDs, C, token count/checksum, retained
   feature count and clear status. Hash rank-0 and rank-1 valid target KV in the
   dedicated audit run.
7. Compare concurrent persistent state against per-request Phase 2 reconstruction
   using identical proposals and actual verification logits.
8. Compare final outputs to a test-only non-speculative greedy continuous-batch
   reference. Any new disagreement enters state/operator/high-precision diagnosis
   under the accepted BF16 cross-shape contract; it is not waived by a threshold.
9. Repeat requests and forced exceptions; assert target/draft maps empty, all
   block refs/free sets consistent, and no monotonic allocated-memory or retained
   state growth after each wave.
10. Run speculation-off regression tests and the full existing unit suite. The
    original positive-temperature sampler path must be byte/source unchanged.

Concurrency output parity alone is insufficient: fixed K changes GEMM shapes, so
the strict gates are transaction invariants, actual-verification acceptance,
determinism for fixed shapes, no new unexplained disagreements, and off-path
parity, consistent with `EAGLE3_NUMERICAL_CONTRACT.md`.

## J. Benchmark Plan

Phase 4.0 does not execute this protocol. After correctness closes, use Qwen3-14B
BF16, TP=2, two RTX 4090, eager, original Scheduler, dedicated EAGLE-3, persistent
draft state, fixed K=3. Keep K=2/4 only as optional auxiliary comparators.

Compare:

- non-speculative continuous batching with the test/serving greedy target mode;
- concurrent speculative fixed K=3;
- optionally fixed K=2 and K=4 after K=3 correctness.

Use concurrency 1/2/4, mixed prompt lengths, mixed output limits, fixed closed-loop
replacement first, then a separately labelled deterministic open-loop workload.
Use new prompts, repeated warmup by batch/query shape, at least five measured
repeats, rotated policy order and no outlier filtering. Keep prefix-cache policy
identical and explicitly record that speculative MVP disables prefix hits.

Report two different concepts:

- wall-clock aggregate serving throughput = total committed output tokens divided
  by measurement-window wall time;
- per-request TTFT, E2E, TPOT and token-level ITL distributions.

Never call `sum(request E2E)` throughput under concurrency. Also report P50/P95/P99,
target/draft forwards, proposed/accepted tokens, accepted per verification,
effective outputs per verification, proposal and verification latency, target
verification query tokens, scheduler request batch size, draft physical subgroup
sizes, K/capacity clipping, GPU memory, allocated/used KV blocks, peak tentative
blocks, cleanup state and TP/NCCL/CUDA errors. Burst-committed tokens share a host
timestamp, so zero intra-burst ITLs must remain visible and be explained.

Manifest: exact target/draft/reference revisions and hashes, nano-vLLM checkpoint
plus dirty source snapshot, environment, GPU topology/software, commands, workload,
arrival schedule, warmup coverage, prefix-cache mode and all raw request/step data.
No formal speedup claim follows from a framework-validation run.

## K. Risks and Unknowns

- The reference draft has no ragged paged KV. Exact-shape proposal grouping may
  have low occupancy under heterogeneous contexts; serial proposal can dominate.
- Target feature history is full-length and replicated on both ranks. Four long
  requests plus per-request draft cache increase GPU0 memory pressure; measure
  before raising concurrency/context.
- Disabling speculative prefix hits avoids missing features but sacrifices an
  existing optimization. Feature-aware prefix caching is a separate project.
- Verification uses the q>1 prefill/GEMM path, so the accepted BF16 cross-shape
  behavior still applies. Concurrent shapes introduce new numerical shapes that
  require diagnostics if outputs diverge.
- The original Scheduler is not round-robin once resident count exceeds
  `max_num_seqs`. Phase 4.1 cannot claim overload fairness or starvation freedom.
- Current shared-memory RPC has a fixed 1 MiB buffer and no explicit payload-size
  guard. Send compact IDs/proposals/descriptors, not full feature tensors or full
  token histories every step; add a length check before writing.
- Host-visible commit is not transactionally recoverable from a distributed CUDA
  failure. Rank disagreement must fault the engine rather than attempt continued
  service.
- `max_num_batched_tokens` historically models prefill tokens and ordinary decode
  treats one row/request. Applying it to speculative q rows is new accounting and
  must be tested at small limits.
- Phase 4.1 is greedy-only. The unchanged sampler is stochastic and cannot be
  reused as a speculative acceptance implementation.
- GPU memory headroom used by the single-request experiments does not prove four
  concurrent feature/draft states fit at every context. Begin with short contexts
  and concurrency 2, then 4 after measuring peaks.

## L. Recommended Implementation Order

1. Freeze hashes for Phase 2 state files and current Phase 3.1 artifacts; add pure
   CPU dataclasses/state-machine and vector acceptance tests.
2. Add BlockManager transaction operations and exhaustive boundary/ref-count/
   rollback tests without invoking a model.
3. Add rank-local per-request target maps and batched prefill/verification packing;
   validate synthetic tensors and TP ordering before loading Qwen3-14B.
4. Add coordinator with serial per-request draft proposals and one ragged target
   verification batch. This is the smallest honest concurrent baseline.
5. Integrate the opt-in adapter with original Scheduler, multi-token postprocess,
   independent finish and preemption cleanup. Keep ordinary `step()` behavior
   unchanged when the feature is disabled.
6. Close concurrency=1 parity against Phase 2, then scripted concurrency=2, then
   concurrency=4 and boundary/error cleanup.
7. Add opportunistic exact-shape physical draft batching only after serial-draft
   concurrent correctness passes; re-run all state/numerical gates.
8. Run a small framework validation. Only after it passes, execute the preregistered
   serving benchmark protocol. Do not start performance tuning in the correctness
   stage.

Phase 4.1 therefore minimally changes engine routing, ModelRunner batched dispatch
and BlockManager transactions, while adding isolated coordinator/runtime/adapter
modules and tests. It should keep Sequence, Attention, TP layers, Qwen3 model,
sampler, all Scheduler policy implementations, Phase 2 primitives and Phase 3/3.1
controllers/artifacts frozen.
