# Secondary: Stable Request Rows and Incremental Verification Metadata

**Conditional design only. Not a second simultaneous implementation.**
Transfer one MRV2 principle: persistent request storage is independent of the
packed order of the current model batch. Use changed-row updates plus a gather
to build target verification inputs. Sources: V1/V2; ownership caution: T1.
This is not the whole MRV2 runner, async scheduling, GPU acceptance or a sampler.

## When to switch

Switch only after Primary fails its bounded correctness/memory/serving gate,
or its initial diagnostic establishes input preparation as the material
removable cost rather than repeated layer dispatch. First separately time
verification layout validation, descriptor creation, host tensor filling,
H2D, GPU input preparation, model execution and result synchronization.

Require input preparation on the non-overlapped critical path to account for
at least 5% of literal-eager serving time in two preregistered cells. This is a
prospective selection threshold, not an observed result. If it does not, stop
the backup too. A graph-enabled miss regression alone does not establish this
condition. Current c<=4 block tables are small; the likely saving could be too
small to justify even this contained design.

## Minimal boundary

Replace only verification input preparation for the concurrent runner:

```text
rank-agreed entries + transaction-owned pages
    -> validate owner and apply changed metadata rows
    -> pack inputs in this batch's exact order
    -> existing Context + unchanged target model
    -> existing CPU acceptance/status/commit
```

The original Python preparation remains default, reference oracle and fallback.
Prefill stays on the old path. No new in-flight step, no read-ahead across
acceptance, no scheduler change, no persistent-KV semantic change, no padding,
no attention replacement and no CUDA Graph dependency.

## Data and ownership

- Stable slot maps `(seq_id, generation)` to one row until close/preemption.
  The transaction layer remains the authority for physical pages. A metadata
  row mirrors that decision; it never allocates/frees a KV page itself.
- Device-resident rows store page IDs and valid page count, committed cursor,
  pending token and owner epoch. On reserve, mirror the legal tentative page
  extension; on commit/rollback truncate metadata visibility exactly as the
  frozen transaction does. Never infer commit merely because verification ran.
- A bounded pinned staging buffer carries changed page cells and small per-step
  records: ordered slot IDs, cursors, q lengths, pending/proposal IDs and
  generations. Use normal asynchronous copies on the existing stream. Do not
  import UVA assumptions or add a second transfer stream in the first version.
- Construct positions, q/k prefix sums, slot mapping and packed page tables
  using device indexing from these inputs. Preserve exact q/k lengths, padding
  sentinels, integer widths and original batch order. `max_q/max_k` needed as
  host API scalars are known from the host entries; never add a GPU `.item()`
  just to recompute them.
- Implementation ladder within this one stage: first preallocated host/device
  buffers and delta copies; measure. Only if device packing is itself justified,
  add one small bounded gather kernel. Do not introduce a kernel framework or
  fuse target math. A new gather needs its own bounds and race tests.
- One buffer set is sufficient for one in-flight step. Do not reuse pinned
  storage before copy completion or packed output before target completion.
  Sequence completion returns the slot only after its consumers finish and
  increments generation. No double-buffered async scheduler in this MVP.
- Rank0 sends the same ordered owner/update data through the existing RPC.
  Both ranks independently apply identical updates; retain current rank checks.
  Fail on stale generation, missing pages, inconsistent tentative end or reorder.

## Planned files and reuse

New `nanovllm/speculative/input_metadata.py` owns mirrors and packed buffers.
An opt-in concurrent runner selects it; `batched_runtime.py` exposes a narrow
input-preparation callback if needed. Keep `ModelRunner.prepare_prefill` as the
unchanged fallback oracle. Existing coordinator, descriptors, transaction
records, proposal offsets, original scheduling and audit flags are reused.
No changes to BlockManager, TP math, attention, sampler or draft state. Add new
focused tests and a standalone preparation benchmark, not edits to old tests.

No training, model weights, extra GPU or NVLink is required. Torch2.8 integer
tensor/copy primitives should suffice for the first ladder step. SM89 support
must be checked before selecting any optional kernel implementation. Small
metadata tensors do not guarantee a speedup: GPU launches may cost more than
the removed Python work.

## Correctness and benchmarks

Compare every generated integer tensor **exactly** with the original oracle.
Test reordered/inserted/removed requests, repeated ID with new generation,
255/256/257 and 511/512/513 boundaries, heterogeneous q lengths, accept0/1/K,
rollback crossing pages, EOS/output clipping, preemption, stale update,
intentional exception and buffer reuse. Invalid slots must not address live
pages belonging to another request. No numerical tolerance is needed for
metadata identity; same inputs/shapes must retain the existing target output
contract, with diagnostics for any new floating-point disagreement.

Run CPU ownership tests, TP1 small-model control, TP2 target tests and all frozen
regressions. Final state/transactions/used pages zero; metadata slots all free.
No new NCCL/CUDA/deadlock errors, no progressive allocation growth.

Then use the Primary's fixed paired serving suite, c1/2/4 and five repeats,
with literal-eager and new-path-disabled controls. Report preparation CPU time,
H2D bytes/calls, new GPU packing cost, synchronization, whole serving throughput
and latency tails, startup, memory and all outliers. Keep profiling separate.

Success requires exact correctness, at least 50% reduction of the independently
identified preparation stage and >=3% serving improvement in two preregistered
cells, all five pairs agreeing with paired uncertainty reported. These are
design gates, not forecasts. If launches/copies erase the gain, or only a
microbenchmark improves, stop. Do not expand to async scheduling/GPU acceptance
to rescue the result. Failure would bound CPU metadata's relevance at c<=4;
it would not disprove MRV2 at data-center throughput or larger concurrency.
