# Phase 4.4C: Long-horizon exact-key reuse

## Preregistered protocol

Written before Phase 4.4C GPU measurements. Phase 4.4B production code,
historical tests, scheduler, sampling, and state semantics remain frozen.

- Literal eager concurrent EAGLE-3, Qwen3-14B BF16 TP=2, original scheduler,
  persistent draft, fixed K=3, no speculative prefix caching.
- Six unchanged Phase 4.2/4.4B shape distributions and phrase families.
  Extend the existing deterministic index/offset/unique-first-token generator
  to 512 requests; do not duplicate the original four-request payload.
- One fresh process per concurrency (1, 2, 4). Each family has its own empty
  observation history, a discarded short warmup, and 512 measured requests.
  Family order rotates with concurrency. No tuning or outlier removal.
- Observe nested windows ending at the step completing request 32, 128, 512.
  Intermediate windows retain in-flight requests; they are not drained
  stand-alone 32/128-request trials. Record overshoot when several finish in
  one step. This preserves continuously replenished serving and its actual
  replacement waves. The final window includes normal drain.
- Observer constructs the frozen CPU layout and GraphKey before the original
  verify RPC; no tensor copies, CUDA synchronization, graph capture, or new
  collectives. Record observer host cost separately. Keep raw inputs, outputs,
  requests, steps, and every eligible/ineligible verification event.
- CPU regressions precede GPU work. An observer-off/on deterministic small
  comparison checks outputs and step structure. Existing close acknowledgments
  check zeroed KV on both ranks; verify final host ownership is empty.

## Projection contract

Observed occurrence counts and distances are measurements, not speedup.
Use Phase 4.4B capacity-4 lifetime costs. Exact matching key estimates are
identified separately from same-M transferred estimates. M8/M16 have no
observed serving replay savings in 4.4B, so their empirical net-saving
projection is **unknown**, not zero and not an extrapolated M4 speedup.
Report sensitivity at 6/14/18 required replays as explicitly hypothetical
reuse thresholds, not measured M8/M16 break-even.

The optimistic unbounded oracle captures on the second occurrence, returns
eager on that call, and can replay occurrences 3 onward. Charge one capture
per key; report both all-captured net cost and hindsight-profitable-only net
potential. It excludes agreement, eviction, memory pressure and prediction
error. Also simulate the unchanged bounded second-occurrence LRU policy
(capacities 2/4/8, maximum 16 captures) on the recorded stream. No new policy
is implemented. Larger ideal caches are diagnostic counterfactuals only.

Conditional bounded validation: for a concurrency, select at most one family
with >=5 keys reaching 18 replay opportunities and >=10% of all verification
events belonging to those keys at 512 requests. Choose the highest coverage,
ties in the fixed family order. This conservative reuse-only trigger also
permits direct measurement where old replay cost is unknown. Run three paired
512-request eager/unchanged capacity-4 graph trials, rotate order, cold graph
cache per trial, preserving the 16-capture limit. These are validations of the
old policy, not a selective-capture implementation. If this trigger is not
met, do not launch real capture runs.

## Output and decision

Report every key, occurrence positions, inter-arrival and distinct-key reuse
distance, context tuples, waves, field-ablation cardinalities, horizon
coverage, bounded-policy projections, and seven requested plots/data tables.
No serving speedup claim from shadow data. A positive oracle alone does not
validate an implementable capture policy. Completion and conclusions will be
appended after all runs; no Git commit.

## Completed results (2026-09-24)

All 18 shadow streams and nine paired bounded comparisons completed normally:
36 measured trials, 18,432 requests. The complete CPU suite passes 139 tests;
Phase 4.1/4.4B focused regression passes 30. Outputs and step signatures match
within all eager/graph pairs. Rank actions, cleanup and frozen-file preservation
checks pass. No production changes, historical-test edits, or Git commit.

Natural exact-key reuse increases strongly with horizon. At 512 requests,
keys with >=18 replay opportunities cover 94.61% / 88.00% / 77.69% of
verifications at concurrency 1/2/4. Removing max_k diagnostically reduces
cell-local key cardinality 566->8, 553->13, 435->14; this is not a safety proof
for removing the field. The old M4 cost model estimates 515 profitable c1
keys covering 95.50% of verifications under unlimited residency. Old M8/M16
replay costs were unknown and are not extrapolated from M4.

Unchanged cap4 bounded serving still loses all nine pairs: aggregate speedups
0.9012x / 0.8048x / 0.8891x, replay rates 3.39% / 3.01% / 1.02%. Matched
true-eager controls show 30/144 capture lifetimes repay capture alone, 27/144
including recorded release costs. Most added verification endpoint time occurs
on eligible misses; its operator-level cause is not established. Local capture
amortization does not imply whole-system amortization.

Decision: current dynamic capture remains off-default. Natural recurrence is
sufficient to justify a separate, bounded selective-admission/residency design
and offline evaluation, including the measured graph-enabled miss penalty,
but not to claim or implement an optimization now. A context-bucketing/padding
safety audit remains separate; exact-key scarcity alone no longer compels it.
No selective capture, bucketing, padding or draft batching was implemented.

See [final report](../benchmarks/eagle3-phase4_4c/report.md),
[summary](../benchmarks/eagle3-phase4_4c/summary.json), and the report's seven
plots, per-key histories and paired raw-data references. Observed reuse,
transferred-cost projections and measured serving results are explicitly
separated. Phase 4.4C stops here pending user confirmation.
