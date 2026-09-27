# Phase 4.4B: Bounded Target Graph Serving Integration

## Frozen Boundary

Opt-in concurrent EAGLE target verification only. Ordinary LLMEngine, ModelRunner,
Attention, Qwen3, sampling, Scheduler, draft state, acceptance and transaction
semantics remain frozen. No context bucketing/padding, draft batching, Adaptive K,
kernel/communication optimization or Git commit. Phase 4.4A prototype/results
and all historical tests/artifacts are unchanged.

Two additive concurrent-specific hooks are permitted: select an opt-in runner
in `concurrent_engine.py`; inject an optional verification forward in
`batched_runtime.dispatch`. Both default to the old path. New graph modules
contain the resource ownership, exact key and bounded policy; no benchmark
module is imported by production.

## Preregistered Policy and Matrix

- Exact Phase 4.4A GraphKey, uniform q=4 at M4/M8/M16 only.
- `never`: every miss uses eager, no automatic capture.
- `second`: capture on the second observed occurrence, at most 16 captures per
  measurement epoch, independent of capacity. The capture-causing call returns
  its eager result; only later hits replay. LRU eviction at capacity.
- Capacities 2/4/8. Observed-key metadata is bounded at 4096 entries and event
  records at 8192; dropped-event counts must be zero for benchmark completeness.
- Primary reference capacity is 4, chosen before results, not a tuned winner.
  Full Phase 4.2 matrix: ordinary, EAGLE eager, graph-never, graph-second-4;
  c=1/2/4, all six workloads, five repeats each.
- Capacity sensitivity: second-2 and second-8, c=1/2/4, long-long and
  mixed-output, five repeats. These stress long exact-key sequences and changing
  batch/clipped shapes without selecting workloads based on results.
- Same frozen Phase 4.2 request construction, closed-loop replacement, committed
  token timestamps, warmup isolation and throughput denominator. Rotated order.
- Each measured trial starts with empty graph cache/history/capture budget.
  Capture, eviction and metadata agreement happen inside serving wall time.
  End-of-trial release is reported separately. No outlier removal.

Additional regression control, declared while the primary matrix is running:
after all six processes finish, run the unmodified Phase 4.2 speculative CLI
with the default (no GraphCache) ConcurrentLLMEngine at c=1/2/4, all six workloads,
five repeats. These 90 fresh-process controls check the disabled opt-in runner
against the literal default runner. They do not replace any of the 420 trials,
select a policy, or change the predetermined capacity matrix.

## Safety and Ownership

Graph entries own only executable, static buffers, model/KV storage references
and shared engine capture resources. They never own requests, transactions or
accepted lengths. Request features are copied out. One capture stream per rank
bounds the cuBLAS workspace cache. Graph pools are not shared.

CPU Gloo consensus precedes target/NCCL work. Same layout but key mismatch or
missing entry means symmetric eager. Actual ordered-input/generation mismatch
fails closed: eager cannot make mismatched collective shapes safe. New control
RPCs acknowledge both ranks before the single-slot mailbox can be overwritten.

Opt-in construction (model/draft/reference paths use the existing local assets):

```python
from nanovllm.speculative.concurrent_engine import ConcurrentLLMEngine
from nanovllm.speculative.graph_policy import GraphCacheConfig

engine = ConcurrentLLMEngine(
    model_path, draft_path=draft_path, reference_path=reference_path,
    speculative_length=3, tensor_parallel_size=2, enforce_eager=True,
    scheduler_policy="original", max_model_len=1024,
    max_num_batched_tokens=2048, max_num_seqs=4, gpu_memory_utilization=0.70,
    target_graph_config=GraphCacheConfig(
        max_graph_entries=4, capture_policy="second", max_captures=16),
)
```

Omitting `target_graph_config` selects the original concurrent runner. The
existing `enforce_eager=True` contract remains required: only this explicit
target-verification opt-in uses graphs, not ordinary decode or draft proposal.
The capture allowance applies to the engine's cache epoch, not each request.
The benchmark explicitly clears/resets that epoch between independent trials.

On a capture decision, both ranks first execute the normal eager verification.
At the coordinated safe point, snapshot only the written verification slots,
warm/capture the unchanged core, restore those slots, and return the saved eager
result. Full-page bitwise auditing verifies committed and rollback regions.
Eviction synchronizes both ranks, resets the executable, scrubs/releases static
buffers, and does not modify request features or KV ownership.

## Gates and Reporting

CPU policy/key tests and frozen regressions first. GPU parity starts at c1,
then c2/c4, checking complete traces, both-rank features/KV/state, growing
contexts, clipping/ragged fallback, replacement, eviction and cleanup. No
numerical tolerance waiver. Serving performance starts only after the gate.

Report three separate layers: exact-key replay target saving, graph-enabled
EAGLE versus eager EAGLE serving, and graph EAGLE versus ordinary serving.
Per-key break-even uses capture cost divided by matched same-key eager/replay
forward saving; keys without a positive matched saving are unidentified, not
invented as profitable. Capture plus eviction remains in E2E measurement.

The final report must keep low hit rate/key explosion as a valid result. Do not
introduce context buckets to rescue this phase. Stop after reporting and await
approval for any follow-up.

## Completed Results

All 420 primary/sensitivity trials and 90 literal-default-runner controls
completed with five repeats per configuration and no removed outliers.
Post-run CPU results are 22/22 focused and 130/130 full-suite tests. The GPU
gate covers 52 request pairs and 182 rank-level bitwise comparisons, with
actual M4/M8/M16 hits, eviction, synchronized fallback, boundary and exception
cleanup. All 1280 measured graph/eager request pairs have identical outputs
and step/acceptance structure; default-runner control outputs also agree.

Primary capacity-4 exact-key hit rates at c1/c2/c4 are **2.27% / 0% / 0%**.
Conditional c1 target endpoint speedup is 2.15x, but graph/eager serving
throughput ratios are **0.553x / 0.764x / 0.600x**. No primary workload cell
has a net aggregate gain over eager EAGLE. The stricter fresh default-runner
comparison gives 0.537x / 0.725x / 0.599x.

All 516 primary capture lifetimes are unprofitable; 476 are never reused.
Capture plus eviction occupies 40.52% / 20.60% / 33.50% of graph serving wall
time. At c2/c4 no exact key occurs more than twice in a measured trial, so
second-occurrence capture cannot replay before the short trial ends. Larger
capacity does not fix that horizon/reuse limitation. No context padding or
bucket was added to improve these numbers.

The largest measured live cache allocation-delta proxy across capacities is
64.60 MiB. After release, both ranks return to fixed allocated-memory plateaus
with a single 8.125 MiB capture-stream workspace increment, not per-key growth.
All state and graph-entry counts clear to zero; GPU processes exit normally.

Decision: retain opt-in status and eager defaults. Do not proceed to draft
batching: remaining eager target execution and capture lifecycle, not draft,
dominate these measurements. A separate longer-horizon exact-key reuse study
and, if warranted, context-bucketing/padded-graph safety design may be proposed,
but neither is implemented or authorized by this phase's completion.

Full results, numerical caveats, memory accounting, cold-cache limitations,
per-cell tables and next-phase decision:
[Phase 4.4B report](../benchmarks/eagle3-phase4_4b/report.md).
The focused diff and source-preservation record are in the same directory.
No Git commit was made; phase work stops awaiting confirmation.
