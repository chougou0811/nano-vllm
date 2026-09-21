# EAGLE-3 Phase 2: Persistent Draft KV / State

## Status

User acceptance: 2026-09-21. Phase 2 is frozen; the checkpoint made after this
acceptance contains production code, test/benchmark sources and lightweight
documentation/summary only. The text below records the pre-checkpoint results.

Phase 1 / 1.1 checkpoint: `087cc27`. Historical strict serial-token parity
failures remain failures. The accepted BF16 cross-shape contract is unchanged.
Phase 2 passes the scoped state/transaction tests below. Phase 2 changes are
not committed. Scheduler, target attention, TP, BlockManager, ModelRunner,
sampling and target KV transaction implementation were not changed.

## Implementation

New `nanovllm/speculative/draft_state.py` owns a per-request rank-0 draft cache.
The engine API adds `draft_state_mode="persistent"`; `full_rebuild` stays default.
The unchanged Phase 1 `ReferenceDraft.propose` remains an independent reference.
`session.py` creates, checks and closes the request state, including exceptions.

Target committed KV has C valid rows and, while generating, C+1 token IDs.
The final token is pending. Draft row i conditions on target feature i and
token i+1. The persistent draft cursor D is no greater than C. Catch-up uses
only features [D,C) and their shifted real token IDs, then saves that cache.

K proposals use K-1 additional autoregressive feedback rows. All these rows
are tentative and discarded: even accepted token IDs do not make draft-hidden
features equivalent to verified target features. After accepting a tokens,
target retains 1+a rows, zeros its rejected suffix and advances C by 1+a.
Draft D remains at the previous C; the next iteration rebuilds only these
1+a new rows from verified target features. Fallback/bonus remains pending.
Terminal EOS can leave token length C rather than C+1; no further proposal runs.

Reference attention concatenates into new cache storage. Keeping the confirmed
tuple separately prevents tentative suffix retention and never slices a view
that owns the whole tentative allocation. Close drops all draft cache references;
target closes its exclusive page lease on both ranks. No cross-request cache
reuse, speculative batching, dynamic K or scheduler change is introduced.

## Correctness

- Full suite: **72 unit tests pass**, including frozen-source/original behavior
  checks, acceptance, target cleanup, owner guards and new persistent tests.
- Exact feature-dependent toy cache tests cover accept 0/1/K/partial, repeated
  rejection, 255/256/257/1024 lengths and eight consecutive transactions.
  They explicitly compare every retained draft KV row against full rebuild.
- Pinned reference model, small FP32 configuration: six transactions including
  0/1/3/2 accept lengths have matching proposals and numerically consistent KV.
  This is an architectural control, not a full-size FP32 model claim.
- Qwen3-14B BF16 TP=2: 255/256/257/1024-token prompts, 16 output tokens,
  full rebuild vs identical-proposal persistent replay: output IDs, accepted
  lengths, verification top-1 IDs, rank status, target cursor, retained features
  and rank-0 valid target KV SHA256 agree exactly at recorded iteration boundaries.
  Both ranks validate replicated features and transaction cursors/token checksum;
  each rank verifies suffix zeroing. No separate rank-1 KV hash was recorded.
- Natural proposals on all four boundary/context cases also match full rebuild.
  The traces exercise accept K, partial acceptance and accept 0; a separate
  forced first-iteration case confirms accept 1.
- Repeated forced rejection, max_tokens 1/2/5, initial/accepted/fallback EOS,
  intentional host exception after proposal, target/draft cleanup all pass.
- Sky and arithmetic chat prompts, 32 outputs: persistent vs full rebuild
  outputs match; repeated persistent runs have identical outputs and proposals.
  These are Phase 2 comparisons, not reclassification of Phase 1 serial failures.
- Seeded speculation-off generation before/after the tests is unchanged.
- Both GPU runs exited normally, with no observed NCCL/CUDA errors or deadlock.

## Draft Numerical Controls

**Do not claim bitwise full-rebuild/incremental draft KV identity.** Across 41
same-target-prefix shadow checks, all proposals agree with unchanged Phase 1
reference, all caches are finite, but maximum BF16 draft KV difference is 0.125.
Identical-input full/chunk FC calls differ by as much as 1.0 in BF16. A local
FP32 control using the actual features and first 64 output rows of the real FC
weight gives maximum full/chunk difference 7.62939453125e-6.

The first shape-changing draft projection is FC. These operator observations,
the FP32 reference architectural control, immutable-prefix tests, and exact
target replay support the documented numerical contract in this coverage.
No raw-logit/KV threshold is used to accept outputs, and no disagreement is
hidden by fallback. This is not an exhaustive full-size FP32 draft audit or a
proof that future draft top-1 decisions never change. Any new token disagreement
still requires state/operator/high-precision diagnosis before acceptance.

## Initial Timing

Qwen3-14B BF16, TP=2, two RTX 4090, eager, original default Scheduler,
K=3, 16 outputs, max length/batched tokens 2048, max sequences 4,
GPU memory utilization 0.70. Each mode/length has a separately recorded warmup
and three measured repetitions, alternating execution order. No outliers removed.
Full per-repeat metrics are in `summary.json`; all warmup/raw records remain on disk.

Medians, milliseconds per request unless noted:

| Prompt | Mode | Proposal | Conditioning | Verification | TTFT | TPOT | End-to-end | Output tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 255 | full rebuild | 26.29 | 11.08 | 215.50 | 55.20 | 16.21 | 298.31 | 53.64 |
| 255 | persistent | 22.95 | 7.82 | 211.32 | 55.11 | 15.72 | 290.97 | 54.99 |
| 1024 | full rebuild | 55.05 | 37.43 | 201.71 | 166.13 | 17.24 | 424.62 | 37.68 |
| 1024 | persistent | 31.60 | 14.36 | 188.35 | 166.13 | 14.94 | 390.13 | 41.01 |

Processed draft input rows drop 1051 -> 274 (255 prompt) and 4126 -> 1043
(1024 prompt); reused prefix row totals are 777 and 3083 respectively.
Both modes use identical forward counts: target/draft 5/11 and 6/12 respectively.
Acceptance is 11/11 and 10/12 proposals, or 2.75 and 2.0 accepted tokens per
verification. Effective output/verification is 3.75 and 3.0, excluding prefill's
first output. Discarded feedback rows total 7 and 8, including accepted-ID rows;
this is distinct from rejected proposal token counts.

Measured median proposal time decreases about 12.7% and 42.6%; end-to-end
decreases about 2.5% and 8.1% in these tiny runs. **These are initial observations,
not a general speedup claim.** Verification also varies despite unchanged code,
so the entire end-to-end change cannot be attributed to draft reuse.

All times use monotonic host clocks, not kernel events. Conditioning timing has
an explicit CUDA synchronization in both modes. Target auditing remains enabled.
The initial state-hash audit includes CPU copies and is excluded from this timing
table. Natural-task diagnostic runs also include shadow forwards and are not
performance samples. This is single-request eager generation, not online serving.

## Memory and Limitations

Rank-0 allocated memory after every timing request returns to **19,566,306,816
bytes**, including the resident target/draft models and preallocated target KV.
Median peak allocation: 19,805,304,832 -> 19,806,398,464 bytes for length 255;
20,174,977,536 -> 20,179,216,896 bytes for length 1024. This small extra persistent
cache is expected; there is no monotonic post-request growth in these runs.
Allocator-reserved memory is not a live KV leak. Rank-1 peak allocation was not
separately sampled. Target occupancy is an exclusive capacity lease with valid
rows tracked by C, not a shared serving cache; cleanup returns all leased pages.
Peak retained draft occupancy is 267 rows / 1,093,632 bytes and 1035 rows /
4,239,360 bytes respectively. Full-rebuild retained occupancy is zero between
proposals, not zero temporary KV allocation during a forward.

Reference `torch.cat` still copies prefix KV when extending cache. Initial draft
prefill still processes all prompt features. Full target feature history is
retained; feature-history compaction and preallocated draft KV are not implemented.
Coverage reaches 1024 context plus generation, not maximum supported context.
Exact target training revision of the dedicated draft checkpoint remains unknown.

## Reproduction and Files

Raw runs: `/root/autodl-tmp/benchmarks/eagle3-phase2/{initial,followup}`.
Logs: `/root/autodl-tmp/eagle-phase2-{initial,followup}.log`.
Target revision `40c069824f4251a91eefaf281ebe4c544efd3e18`;
draft revision `3d13517724e81cb409ddf1d4650772ec52f1e18e`.
Manifests contain GPU/config/software data; source snapshots accompany the runs.

```bash
env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN TMPDIR=/root/autodl-tmp/tmp \
 TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/tmp/torchinductor \
 TRITON_CACHE_DIR=/root/autodl-tmp/tmp/triton \
 /root/autodl-tmp/venvs/nano-baseline/bin/python \
 -m benchmarks.serving.eagle3_phase2 --output /path/to/new-initial-run
# Use the same environment for benchmarks.serving.eagle3_phase2_followup.
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest discover -s tests -q
```

Files added: `draft_state.py`, `test_eagle3_persistent.py`, three Phase 2 benchmark
modules, `docs/EAGLE3_PHASE2.md`, this report and summary. Files modified:
`llm_engine.py` (opt-in API argument), `session.py` (request state and metrics).
User-owned `AGENTS.md` changes remain untouched and uncommitted.

## Next Stage

Persistent single-request state is ready for review/freeze within this coverage.
Adaptive K can be a separate next phase after reviewing this diff; it is not
implemented here. Concurrent speculative serving is **not ready**: it needs
per-request page ownership, batching/rank protocols and cancellation isolation
tests beyond the current exclusive idle-engine lease. Do not infer concurrent
safety from the request-owner guard. No Scheduler changes are needed or made here.
