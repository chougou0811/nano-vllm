# Phase 4.4C: Long-Horizon Exact-Key Reuse

**Status: complete. All 36 measured trials finished; no outliers removed.**
Long-window exact-key reuse exists, but the unchanged bounded policy loses
end-to-end in all nine paired comparisons. No production or historical-test
changes, no optimization, no Git commit.

## A. Setup and preregistration

Protocol: [EAGLE3_PHASE4_4C.md](../../docs/EAGLE3_PHASE4_4C.md).
Raw directory: `/root/autodl-tmp/eagle3-phase4.4c-20260924`.

- Qwen3-14B BF16, TP=2, 2 x RTX 4090 24 GB, original scheduler, eager,
  dedicated EAGLE-3, persistent draft, fixed K=3, greedy, prefix cache disabled.
- Target revision `40c069824f4251a91eefaf281ebe4c544efd3e18`;
  draft revision `3d13517724e81cb409ddf1d4650772ec52f1e18e`.
- `max_model_len=1024`, batched token limit 2048, sequence limit 4,
  GPU memory utilization 0.70, unchanged from 4.4B.
- PyTorch 2.8.0+cu128, FlashAttention 2.8.3, Triton 3.4.0,
  Transformers 4.57.1. Full commands, topology, GPU and software metadata,
  source hashes, raw inputs, outputs and per-step records are in manifests.
- Each concurrency uses a fresh process. No graph runner/control group in
  shadow runs. Warmup and observer-off/on checks are outside measurement.

## B. Workloads and windows

The six frozen shape distributions are unchanged:

| Family | Prompt tokens | Output limits |
|---|---|---|
| short-short | 64 | 32 |
| short-long | 64 | 128 |
| long-short | 768 | 32 |
| long-long | 768 | 128 |
| mixed-prompt | 64/192/384/768 cyclic | 64 |
| mixed-output | 256 | 16/32/64/128 cyclic |

Extend the original deterministic phrase/offset/unique-first-token generator
to indices 0..511. Do not copy the old four-request inputs. The original
prefix and distributions are unit-tested. No post-observation workload change.
Closed loop immediately replenishes requests at engine-step boundaries.
The frozen runner retains `ignore_eos=True` to enforce exact output counts;
these are controlled fixed-length workloads, not natural EOS distributions.

There are 18 actual 512-request streams, 9,216 measured shadow requests and
54 nested windows. Windows end at the step completing request 32/128/512;
actual completed count and any simultaneous-completion overshoot are recorded.
Intermediate windows are not separately drained trials. Final windows retain
the normal end-of-stream drain, which is annotated, never filtered.

The observer uses frozen `verification_layout` and `make_key`, without graph
capture, tensor copies, extra collectives or CUDA synchronization. Its own
recorded CPU span is 0.261% / 0.218% / 0.145% of shadow wall time for c1/c2/c4.
This does not remove the inherited benchmark collector overhead.

## C. Correctness protection

- Historical complete CPU suite before changes: 130 passed.
- Phase 4.1 plus 4.4B focused CPU suite: 30 passed.
- New observer/statistics tests cover frozen workload parity, no duplicated
  payloads, occurrence/capture accounting, distances, capture cap, unknown and
  negative savings, empty windows, and monotonic cumulative counts.
- Observer-off/on GPU outputs and step signatures match at all three
  concurrency levels. This is an instrumentation check, not a new universal
  BF16 serial/parallel numerical-parity claim.
- All 18 shadow streams finish normally. Every request has matching successful
  zeroed-KV close acknowledgments from both ranks. Final host queues, blocks,
  transactions, target ownership and draft ownership are empty; prefix hits 0.
- The frozen-file hash comparison reports no changed historical files.
- Final complete CPU suite: 139 passed, including nine new tests. Final
  Phase 4.1/4.4B focused suite: 30 passed.
- All nine bounded eager/graph pairs have identical committed outputs and
  step signatures. Both ranks agree on graph keys/actions; measured replay
  counts equal the frozen-policy simulation. All 36 measured trials exit
  normally, with no NCCL/CUDA/deadlock failure or state/transaction/KV leak.

## D. Observed reuse

Counts below sum **cell-local key instances** across six independent families;
they are not a globally deduplicated key universe or one combined traffic run.
The 18-replay column requires at least 20 occurrences: the first is eager,
the second captures and still returns eager, and occurrences 3 onward can replay.

| C | Requests per cell | Unique key instances | Keys with >=18 replay opportunities | Verification coverage of those keys | Unbounded second-capture hit opportunity |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 460 | 13 | 8.96% | 73.05% |
| 1 | 128 | 539 | 375 | 88.29% | 89.25% |
| 1 | 512 | 566 | 481 | 94.61% | 94.23% |
| 2 | 32 | 378 | 0 | 0% | 56.10% |
| 2 | 128 | 481 | 157 | 50.80% | 80.61% |
| 2 | 512 | 553 | 372 | 88.00% | 89.01% |
| 4 | 32 | 297 | 0 | 0% | 38.74% |
| 4 | 128 | 371 | 48 | 31.68% | 70.08% |
| 4 | 512 | 435 | 231 | 77.69% | 81.21% |

At 512, keys occurring at most twice are only 26/566, 70/553, 85/435
for c1/c2/c4. Thus the short-window observation that most keys rarely recur
does **not** persist in these long, bounded-distribution streams.

Detailed per-family/per-horizon numbers: [horizon-table.md](horizon-table.md),
[horizon-table.csv](horizon-table.csv). Every key's first/second/later indices,
occurrences, full key fields, context tuples, request IDs, nominal and actual
admission waves, lifetimes and distances are under `keys/`.
Lifetimes are observed first-to-last verification/request-index ranges, not
actual graph residency. They are right-censored at the observation horizon.
Reuse distance is reported both as verification-index delta and intervening
distinct eligible keys, not just one ambiguous distance definition.

## E. Fragmentation and replacement waves

Exact `max_k` remains the main source of key variety. Removing only `max_k`
offline changes summed 512-window cardinalities:

- c1: 566 -> 8; c2: 553 -> 13; c4: 435 -> 14.
- This is diagnostic field ablation only, **not permission to reuse graphs
  across different max_k**, and not a padded-graph safety result.
- Other varying classes are M/batch/q tuple and table width. Model identity,
  dtype, device class, TP size, mode and feature layers remain fixed.
- M, batch and q lengths are correlated fields. Removing one redundant field
  alone need not reduce cardinality; the joint shape fields still identify it.

Exact key does not include the complete context tuple: those are runtime tensor
values. Different replacement-wave context tuples can share a max_k and key.
This is the existing 4.4B contract, not a newly relaxed key.

Fixed-length families saturate rapidly. For example c1 short-short has 28
keys at all three horizons, and short-long has 124 at all three. Increasing
output horizon creates a broader context range and longer reuse distance:
c1 short-short median distinct-key distance 22, short-long 119.

After 128 requests, new key instances are 27/72/64 for c1/c2/c4. Of those,
0/62/43 first appear during final drain. The non-drain additions are therefore
27/10/21, not continuing linear key explosion. Mixed-prompt c1 still grows
123 -> 146, so saturation is not universal at exactly request 128.

Higher concurrency does not monotonically increase key cardinality. In
mixed-prompt, the largest-context request dominates max_k, reducing cardinality
146 -> 78 -> 31 at c1/c2/c4. But there are fewer verification events per fixed
request horizon and more ragged/ineligible transitions. The aggregate replay
opportunity and conservative reuse coverage decrease with concurrency.

## F. Hypothetical cost projection

Use 4.4B capacity-4 measured lifetimes. Match identical keys where old replay
measurements exist; otherwise transfer the same-M median. These are transferred
estimates, not new per-key cost measurements. Some historical key-level savings
are attached to multiple lifetimes and are not independent statistical samples.

- M4 median capture 266.372 ms; pooled per-hit saving 21.087 ms.
- M8 median capture 268.039 ms, **replay saving unknown** in 4.4B.
- M16 median capture 322.410 ms, **replay saving unknown** in 4.4B.
- Per-key break-even is `ceil(capture_ms / saving_ms)` for positive savings.
  Nonpositive measurements are retained and have no finite break-even.
- The 6/14/18-replay scenarios are explicit sensitivity cases, not fabricated
  M8/M16 performance measurements.

For c1, 515 of 566 cell-local keys reach their transferred break-even at 512;
they cover 60,530 / 63,382 verifications (95.50%). Each family has all 512
requests participating in at least one such key, which does not mean every
verification or the entire request is accelerated.

The unbounded second-capture model projects +1,111.59 seconds net across all
six c1 streams; the hindsight-profitable-only model projects +1,115.17 seconds.
These intentionally optimistic values omit finite residency, agreement,
eviction, prediction error and most lifecycle effects. They are **not serving
time saved**. Most c2/c4 keys have unknown empirical saving; zero contribution
from their few M4 drain keys is not a zero-saving conclusion about M8/M16.

## G. Cache capacity and admission

Simulate the unchanged policy: capture on second occurrence, capacities 2/4/8,
maximum 16 captures, same seen-key bound. Capture calls do not replay.
Also report a zero-cost, always-admit LRU reference at 2/4/8/32/128/512/2048.
That reference is not an upper bound across different admission policies.

The key working set is much larger than four entries. At 512, frozen capacity-4
simulation has only 653 hits for c1 short-long, 278 for c2 long-long, and 47
for c4 long-long despite high unbounded reuse coverage. Cache residency and
early capture selection therefore remain separate from natural recurrence.

Larger capacity usually increases projected hits, but not universally under
the 16-capture budget: c1 mixed-output has 1,027 hits at capacity 4 versus 952
at capacity 8. Capacity changes which captures consume that budget. Do not
infer monotonic serving benefit or memory safety from the hit simulation.

## H. Conditional bounded validation

The preregistered >=5 keys / >=10% coverage rule at 18 replay opportunities
selects c1 short-long, c2 long-long, c4 long-long. Each uses three paired
512-request trials, rotated eager/graph order, cold graph history, capacity 4,
second-occurrence capture, maximum 16 captures. No selective policy is added.
The graph event-record bound is increased to 200,000 solely to retain long-run
logs; the capture/key/shape semantics and resource policy remain frozen.

Eager comparison uses the existing graph runner with graph execution disabled;
the shadow runs use the literal original concurrent runner. We do not
substitute cross-process shadow times as paired controls. Runtime key/action
parity, actual hits versus simulation, per-request
outputs, cleanup and normal exit are checked.

All repeats are complete. Each arm generates 65,536 committed output tokens.
Throughput uses total committed tokens / total wall time, never summed request
latency. Speedup is sum(eager wall) / sum(graph wall) within the three pairs.

| C / family | Eager tok/s | Graph tok/s | Speedup | Repeat speedup range | Replays / verifies | Serving wins |
|---|---:|---:|---:|---:|---:|---:|
| 1 / short-long | 73.18 | 65.96 | 0.9012x | 0.8877-0.9129x | 1959 / 57849 (3.39%) | 0/3 |
| 2 / long-long | 111.23 | 89.51 | 0.8048x | 0.7999-0.8112x | 834 / 27732 (3.01%) | 0/3 |
| 4 / long-long | 145.18 | 129.09 | 0.8891x | 0.8842-0.8947x | 141 / 13875 (1.02%) | 0/3 |

Full repeats, latency summaries, actions and lifetime accounting:
[bounded-table.md](bounded-table.md), [bounded-results.json](bounded-results.json).
Together with shadow runs, there are 18,432 measured requests in 36 trials;
the 54 nested shadow windows are not additional independent trials.

### Actual capture amortization

There are 144 logical capture lifetimes, not 288 independent TP observations.
The legacy within-graph-run eager comparison classifies 33 as profitable.
A stronger matched control compares each replay with the same verification
in its paired graph-disabled run. This yields **30/144 capture-only profitable
lifetimes**, or **27/144 including recorded release costs**. Grouped across
recaptures of a key within a trial, 27/120 captured key-trial instances are
capture-only profitable. These are different denominators from shadow's
cell-local key universe; they must not be substituted for oracle coverage.

Paired profitable lifetime counts by c1/c2/c4 are 12/12/6; including release
cost they are 12/12/3. Paired per-hit break-even estimates for replayed
lifetimes have medians 21 / 16.5 / 11 replays (ranges 19-30 / 13-20 / 10-11).
They are measurements of this subset, not universal costs for every key.
The old 4.4B transfer estimate is not silently overwritten. Unlike 4.4B,
some long-lived captures now repay their direct cost, but that does not repay
the whole graph-enabled system.

### Where the remaining wall time goes

Partition identical verification indices by the actual graph action. Values
below sum graph endpoint time minus paired eager endpoint time, in seconds
across all three repeats. Negative means time saved. Every event is retained.

| C | Eligible miss | Ineligible fallback | Capture | Replay | Total verify delta | Serving wall delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | +294.096 | +3.362 | +29.794 | -35.726 | +291.526 | +294.378 |
| 2 | +394.189 | +21.401 | +23.537 | -17.687 | +421.440 | +428.777 |
| 4 | +136.945 | +11.746 | +22.626 | -4.522 | +166.794 | +168.839 |

Most lost endpoint time occurs on eligible misses, not capture events alone.
This is an action-conditioned wall-clock measurement, **not an operator-level
root-cause diagnosis**. Do not attribute all of it to NCCL, Python GC, graph
capture, or a particular kernel. Logged agreement times total 49.762 / 28.721 /
15.208 seconds; capture times 19.600 / 16.027 / 16.398 seconds; release times
9.040 / 6.394 / 5.620 seconds. These internal spans are nested in endpoint
timing and must not be added again to its totals. Selective admission alone
is not proven sufficient while graph-enabled misses have this penalty.

### Memory and cleanup

Rank-0 peak allocated memory is 18.353 / 18.737 / 18.808 GiB for c1/c2/c4.
Rank-1 process-scope peaks are 15.743 / 16.113 / 16.170 GiB, not isolated
trial peaks. After cache clear, allocated bytes are stable across all repeats:
19,574,825,984 on rank0 and 16,778,348,544 on rank1. Model/runtime allocations
remain resident; request ownership, transactions, used blocks and graph entries
are zero. Both ranks report successful state cleanup; graph event drops are
zero. After all engines exit, both GPUs show 1 MiB and no compute processes.

## I. Plots and data

- [Unique keys vs requests](plots/unique-keys.png)
- [Cumulative reusable keys](plots/reusable-keys.png)
- [Occurrence histogram](plots/occurrence-count.png)
- [Reuse-distance distribution](plots/reuse-distance.png)
- [Hypothetical profitable coverage](plots/profitable-coverage.png)
- [Projected hit opportunity](plots/projected-hit-rate.png)
- [Transferred-cost net-saving projection](plots/projected-net-saving.png)

Each curve has underlying CSV/JSON data. M8/M16 unknown savings are explicitly
labeled; they are not extrapolated from M4. No outliers or failed samples removed.

## J. Limits

One deterministic 512-request stream per family/concurrency characterizes
this fixed workload support, not arbitrary production traffic. Nested windows
are not independent repeats. Families have independent observation histories;
we did not measure a shared cross-family graph cache. Output limits are capped
at 128, prompts at 768; wider/longer distributions can have larger key spaces.
Prefix-cache isolation and synthetic unique first tokens follow the frozen
generator. The exact target-training revision of the draft remains undisclosed.

Bounded validation has three paired repeats per selected cell, not a broad
statistical sweep. Clocks are not locked. Instrumented timings include retained
event logs (the logging bound is 200,000), observer and host collection costs.
Only one family per concurrency receives bounded validation. The enabled-miss
penalty is localized by action, not explained at the operator level. No
selective-policy benefit, larger-cache memory safety, or padding safety has
been established. Empirical projections with unknown M8/M16 costs remain
unknown; plots never present them as zero saving.

## K. Final answers and decision

1. **Natural reuse:** yes, for these unchanged bounded workload distributions.
   Most keys recur more than twice by 512 requests. This is observed reuse,
   not a shadow-study serving-speedup claim.
2. **Horizon:** coverage by keys with at least 18 replay opportunities rises
   from 8.96/0/0% at 32 requests to 94.61/88.00/77.69% at 512 (c1/c2/c4).
3. **Fragmentation:** exact max_k dominates; remaining differences include
   query/batch shape and block-table width. Full context tuples are not keyed.
4. **max_k:** still the dominant field, but finite supported distributions
   eventually revisit its values; high variety is not perpetual novelty.
5. **Break-even keys:** the 4.4B transferred cost model gives 515/566 c1 keys;
   most c2/c4 costs were unknown. The 18-replay sensitivity yields 481/372/231
   keys. Actual bounded tests repay capture for 30/144 lifetimes, not all keys.
6. **Coverage:** c1 cost-estimated profitable keys cover 95.50% of shadow
   verifications under unlimited residency; conservative 18-replay coverage
   is 94.61/88.00/77.69%. Actual cap4 replay rates remain only 3.39/3.01/1.02%.
7. **Larger cache:** meaningful as a residency question, not yet a deployment
   recommendation. The working set exceeds four; admission/capture-budget
   interactions mean even simulated hit counts are not always monotonic.
8. **Selective capture:** sufficient evidence for a bounded design/offline
   evaluation, not evidence of serving benefit or authorization to implement.
   Estimate future resident hits, matched savings, capture/release cost and
   graph-enabled miss/control cost; do not capture unknown-cost keys on an
   assumed positive return. Avoid startup-biased admission, retain TP-consistent
   actions and existing safety checks, and validate on independent traces.
9. **Terminate exact-key?** Stop treating current second-occurrence/cap4 dynamic
   capture as a serving optimization; keep it off-default. Do not conclude all
   exact-key research is futile: observed long-window reuse falsifies that
   stronger claim. Any further policy work requires separate approval.
10. **Bucketing/padding:** fragmentation supports considering a separate safety
    audit, but these results neither prove safety nor make it the only viable
    route. First distinguish residency/admission and graph-enabled miss costs
    from shape fragmentation. No bucketing or padded graph was implemented.

**Phase 4.4C is complete. Stop here and await confirmation.** Frozen production,
historical tests and prior artifacts are unchanged; no Git commit was made.
