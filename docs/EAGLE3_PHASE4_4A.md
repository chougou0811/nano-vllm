# Phase 4.4A: Fixed-shape Target CUDA Graph Prototype

## Pre-implementation Audit

Frozen: all production sources, historical tests, Phase 4.1 semantics and Phase
4.2/4.3 artifacts. This is an isolated benchmark prototype, not production-wide
graph integration. No Git commit. Qwen3-14B BF16, TP=2, K=3, original Scheduler.

The frozen verification endpoint resolves request state, constructs paged
prefill descriptors, executes Qwen3, gathers vocabulary logits, validates
finite/replicated features, and returns rank status. Only the tensor-compute
subregion is a capture candidate. Python state transitions cannot be replayed
by a CUDA graph and must remain outside it.

### Static Storage

Each explicitly captured graph owns fixed-address GPU buffers for input_ids
and positions (int64 M), cu_seqlens_q/k (int32 B+1), slot_mapping (int32 M),
block_tables (int32 B x W), hidden output (BF16 M x H), feature output (BF16
M x 3H), local logits (BF16 M x V/TP), and rank0 gathered logits (BF16 M x V).
Model weights, RoPE tables and the engine's paged KV allocation must also retain
their addresses. Capture-private intermediates live in the graph memory pool.
Replay changes contents by copying into existing buffers, not replacing them.

Graph outputs are borrowed storage. Feature rows handed to request state are
cloned outside capture so later replay cannot overwrite retained target state.
No graph owns a Sequence, proposal transaction or request dictionary. Buffers
are scrubbed on release; graphs are destroyed before KV/model/process-group
teardown. Capturing writes tentative KV, so tests restore the original page
contents after warmup/capture and before comparison/replay.

All captured keys share one engine-owned capture stream, but not graph memory
pools. A standalone control reproduced 8,519,680 bytes of retained cuBLAS
workspace per newly used stream; repeatedly creating streams grew allocated
memory even after graph/request cleanup. Reusing one stream bounds this library
cache without clearing backend caches or changing production behavior.

### GraphKey

M alone is unsafe. The bounded prototype key includes B, q lengths (all four),
exact max_seqlen_q/k, block-table width, block size, dtype/device class, model
mode/config identity, TP size and feature-layer selection. Per-rank storage
addresses are validated locally, not compared numerically across devices.
Ordered request IDs, generations and actual layout content are agreed per call,
but are not retained as graph ownership. Missing keys use synchronized eager
fallback; no opportunistic capture or unbounded dynamic graph cache is allowed.

Use exact max_seqlen_k, not a padded ceiling: changing that Python scalar may
change FlashAttention specialization and violates the same-shape comparison.
Thus context changes can require different explicitly prepared keys.

### Captured and External Work

Capture unchanged embedding, transformer projections/norms/RoPE/attention/MLP,
81 model all_reduces, LM-head gather and copies to graph-owned output/feature
buffers. Eager remains the frozen implementation. No custom math or collective.

Keep Scheduler, proposal, acceptance, reserve/commit/rollback, cursor updates,
request ownership, finite/replication/status validation, cleanup and coordinator
bookkeeping outside capture. Retain all original correctness collectives.
Additional benchmark-only Gloo consensus prevents asymmetric replay/fallback
and rejects mismatched keys/layouts before model NCCL is submitted.

Every prototype control RPC ends with a two-rank acknowledgement. The frozen
shared-memory mailbox has one slot, not a message queue; a fast unsynchronized
mode-change call could otherwise be overwritten. This applies only to the new
prototype control endpoint. The original target RPC is left unchanged.

## Feasibility Gate

First run a fresh-process two-rank probe of all_reduce and gather on a capture
stream, repeated replay, graph/eager interleaving and teardown. Then capture the
actual target. Library support is not sufficient evidence of application safety.
Any real target capture/NCCL failure stops the phase; do not remove validation
or change communication settings to force success.

NVIDIA requires uniform capture/replay decisions across participating ranks and
warns about multi-GPU-per-thread launch deadlock. Here there is one GPU per
process. Sources: [NCCL CUDA graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html),
[PyTorch 2.8 CUDA graphs](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs).

## Correctness and Measurement Plan

Before performance: exact same-shape eager/graph comparisons of local/gathered
logits, target IDs, features and all touched KV pages on both ranks; no tolerance
waiver. Cover M=4/8/16, 255/256/257 boundaries, heterogeneous and longer contexts,
repeated replay, synchronized fallback, errors and teardown. Compare complete
eager versus prototype request traces (proposals, verifications, acceptance,
commits, cursors, ordering/status and final outputs). Any disagreement keeps the
gate open and prevents timing claims.

Only after that gate, collect paired, alternating eager/graph target timings
with host monotonic clock and CUDA events; save each sample, including outliers.
Separate core computation from full verification including prototype consensus,
buffer copies and unchanged finite/status work. Minimal dual-rank traces are
mechanism evidence only: count CPU kernel launches versus graph launches, GPU
kernel nodes, device gaps and matching collective start skew. They are not the
speedup denominator. Capture/warmup are outside measured windows.

Primary timing covers the complete verification RPC, excluding rollback. The
additional coordinated-eager comparator uses the same graph-plan consensus but
executes the frozen eager target. This distinguishes coordination from replay;
neither comparator removes correctness checks. Use five rotated repeats with
ten calls per mode and exact context lengths 256/768 for each M. A second fresh
process repeats this matrix. Traces run separately with two targets per mode.

Re-run focused and full CPU suites before/after. Do not begin Phase 4.4B until
the user reviews the report and authorizes integration.

## Completed Experiment

The final target gate passed 17 paired cases / 41 paired requests and 210
rank-level bitwise replay comparisons. A supplemental same-key test changed
request IDs, token contents, physical blocks and per-sequence contexts: 54
additional rank-level comparisons passed, using one capture/M/rank.
Fallback, intentional acceptance exception, graph release and engine exit
passed. Post-release allocated memory is byte-identical across all cases.
Focused tests remain 22/22; the full CPU suite is 122/122 (seven new tests).

Two fresh performance processes completed five rotated repeats of ten calls
per mode/cell: 1,800 primary samples across frozen eager, coordinated eager and
graph. Mean target RPC speedups for M4/M8/M16, equally pooling context256/768,
are 2.930x / 3.126x / 2.820x. All 60 paired repeat means improve. These are
target-only results, not serving speedups.

CPU launch submissions fall from 640/641/643 to one graph launch plus 15/16/18
external kernel launches. Internal matched-collective skew and device gaps
decrease; raw RPC entry skew does not consistently improve. Graph launch API
CPU duration itself remains substantial, including without profiling, and
overlaps device execution. Do not equate fewer submissions with lower summed
API CPU time or add CPU and GPU durations.

Recommend only a user-approved, bounded Phase 4.4B integration experiment.
Exact context keys may have poor hit rate during growing-context serving;
capture cost, key coverage, memory and any future context bucketing need explicit
validation. No production integration or Git commit was made.

Full evidence, retained failures, limits, commands and the proposed integration
boundary: [final report](../benchmarks/eagle3-phase4_4a/report.md).
