# Phase 5.3: Cross-Request Draft-Step Batching

Experimental opt-in implementation. See the
[final report](../benchmarks/eagle3-phase5_3/final-report.md) for measured adoption
status. No default switch, checkpoint change, training, scheduler redesign,
catch-up batching or MLP replay combination is part of this experiment.

## Actual Execution Boundary

The frozen Phase2 drafter uses one feature-conditioned forward, one first-token
head, then K-1 autoregressive feedback forwards/heads. For K3 and B requests:

1. Perform B independent conditioning calls using each request's confirmed KV.
2. Stack the last hidden rows and evaluate one batched first-token head.
3. Right-pad copies of confirmed KV into temporary batch storage. A boolean
   mask excludes every padded column; per-row position IDs use logical length.
4. Feed each row's preceding proposal into the next batched forward, then head.
   Repeat for the third proposal. Tokens remain autoregressively dependent.
5. Remove rows whose K has clipped short, preserving an explicit original-index
   mapping. Return proposal lists in the original scheduler-selected order.
6. Publish only independently allocated confirmed KV to each DraftState. All
   feedback KV is discarded, even for proposals subsequently accepted by target.
7. Existing ragged target verification, greedy acceptance and target transaction
   commit/rollback execute unchanged.

Actual code packs before the first head; the logical steps above describe the
data dependencies, not an additional optimization. There is no persistent
scratch pool and no scatter of feedback KV into request state.

Compared with Phase5.2's sketch, the precise boundary is **three heads and two
feedback forwards**, not three additional draft-model calls. Full model forwards
including serial conditioning decrease B*K -> B+(K-1), while generation-only
forwards decrease B*(K-1) -> K-1 for uniform K. Forward-count reduction alone is
not the performance gate; timings include copies, padding, masks and token reads.

## Ownership and Safety

`DraftBatchInput` carries the DraftState, live request identity, captured
generation, confirmed target features, tokens and clipped K. Check all owners
before conditioning and again before any state publication. Duplicate states,
foreign seq_ids, stale generations, changed prefixes and invalid extents fail.

Request cursors count logical confirmed rows, never padded width. Positions for
feedback step j are `confirmed_length+j`; the feedback row is physically appended
after the padded width, and the mask retains the gap as invalid. Confirmed KV
does not alias scratch. Generation exceptions publish no partial batch state;
the existing engine exception path closes requests and releases transactions.

The adapter has no persistent tensor fields. Its metrics retain numbers only.
Only requests already selected for this iteration can form a group. Single
eligible requests, including c1, execute the unchanged `DraftState.propose`.
K0 rows need no draft call. The original reserve order and KV-capacity clipping
are retained after proposal. No future arrivals are delayed for batching.

## Opt-In Surface

`ConcurrentLLMEngine(..., draft_batching=True)` creates the optional executor.
Default is false. Changed existing files are limited to `concurrent_engine.py`
and `coordinator.py`; implementation lives in new `draft_batch.py`. Frozen
`DraftState`, Scheduler, target Attention, TP, BlockManager, sampling and target
transaction code are unchanged. Benchmarks may toggle the same optional executor
on an idle engine to rotate paired system order without model reload.

## Numerical Contract

Proposal logits may differ under BF16 batch-shape changes. They are not declared
equivalent by an arbitrary maximum-error threshold. Diagnostics compare the
frozen serial oracle, native-length serial, identically padded serial, batched
execution, teacher-forced feedback and FP32 arithmetic on the same checkpoint.
Confirmed state, ownership and positions remain strict invariants. The first new
proposal disagreement stopped serving for diagnosis. Subsequent runs explicitly
retain strict signature failures and mark them pending independent diagnosis;
committed-output disagreements still stop immediately. See the correctness
report for unresolved cross-process reproducibility limits, not a blanket BF16
waiver.

The checkpoint still does not disclose its exact target training revision.
Phase1 historical strict serial-target parity failures are not reclassified.

## Provenance

- SGLang `3e7e6529002db8125967967e0aba394986eec27d`,
  `python/sglang/srt/speculative/eagle_worker_v2.py`,
  `EagleDraftWorker.draft_forward` / `_draft_extend_for_decode`: separate
  conditioning from cross-request autoregressive-step batching. No subsystem
  or KV-retention implementation was copied.
- EAGLE reference `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`,
  `eagle/model/cnets.py`, `Model.forward` / `LlamaAttention.forward`: existing
  loaded checkpoint forward supports batch masks, explicit positions and
  concatenated non-in-place past KV. The pinned source hashes in ReferenceDraft
  remain enforced; no reference source edits.

## Measurements

Raw artifacts stay in `/root/autodl-tmp/eagle3-phase5.3-20260925/`, including the
pre-edit dirty source archive, captured states, manifests, traces and errors.
Captured tensors are test artifacts, not committed model data. No Git commit.
See the phase reports for results, limitations, and the explicit stop decision.
