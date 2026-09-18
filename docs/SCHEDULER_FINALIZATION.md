# Scheduler V2.1 Finalization Protocol

Frozen before new GPU measurements. No production source changes, parameter
search, new scheduler variant, default switch or commit. The V2 manifest's
production hashes are checked before and after execution; the dirty worktree is
archived under `/root/autodl-tmp/benchmarks/scheduler-v2.1/pre-finalization/`.

## Independent Workloads

Eight requests per workload, two copies of each listed (prompt, output) pair:

| Mix | Pairs | Arrival intervals (ms) |
| --- | --- | --- |
| Interactive | (96,24), (192,40), (320,72), (640,112) | 240, 650 |
| Prefill-heavy | (768,4), (1152,12), (1408,20), (1856,28) | 180, 550 |
| Decode-heavy | (48,80), (160,120), (448,144), (896,160) | 300, 900 |

All distributions, rates, mixtures and random seeds differ from V2 tuning and
held-out. Old request first tokens are excluded. Workload contents are paired
across policies; each of five repeats has new tokens and seeded shuffled order.
Five cyclic policy rotations balance execution position. This is a finite
synthetic workload matrix, not a general traffic model or indefinite stability
test. No filtering of slow trials/tokens/steps.

Five configurations: original, static-256, static-512, frozen V2, V2 no-overload.
Each runs all six workloads five times (150 trials). No third overload variant
is introduced: the existing no-overload flag is the simplest isolated comparison.
V2 parameters stay min256/max1024/overload512 with bucketed cost model. SLO
scoring remains TTFT2000ms and request-average TPOT100ms. It is not an every-token
ITL SLO. Qwen3-14B TP2 BF16 eager, max model length/token budget2048, batch cap4,
memory utilization0.85. Temperature0.6, ignore EOS, seed42. Open-loop arrivals.

Fresh scheduler/KV metadata per trial and distinct warmup first tokens prevent
cross-trial cache hits. Shared shape warmup covers batch1..4; per-policy cost
warmup is outside measurement. Actual initial prefix hits must be zero and all
KV blocks must be released. Raw partial data are saved before aborting a failed
measured trial. Model hashes, revision, environment, source snapshot and all
planned tokens are saved before execution.

The complete controller state after policy warmup, including overload latch and
healthy-check count, carries into measurement just as in V2; it is not an
EWMA-only warm start. Last warmup overload states are reported per repeat. No
post-hoc reset or extra warmup is introduced after seeing finalization results.

## Correctness and Diagnostics

Reuse frozen V2 HF numerical rules and the independent correctness corpus, not
the old performance workload. Check all five policies, original exact repeat,
teacher-forced prefixes, finite values, full output coverage and KV release.
No candidate-driven recalibration. The existing small-model FP32 control remains
applicable because production code is unchanged; this is not a new FP32 run.

A benchmark-only observer records the existing overload predicates and state
transitions without changing the scheduling decision. Unit tests compare traces
and cost state with and without it. Extra Python observer overhead is included
in timing and can slightly perturb feedback; this is reported, not subtracted.

Report step-weighted and elapsed-step-time-weighted overload fractions, and
separately fractions among actually evaluated waiting-present steps. With no
waiting, V2 retains its prior overload state; high recorded occupancy alone
does not prove persistent overload. The backlog predicate pessimistically
charges a decode cost even without running requests, and compares draining the
whole backlog to the head deadline. These semantics are diagnostic limitations,
not production fixes in this frozen experiment.

Report all requested latency percentiles, throughput/goodput, queue depths and
waiting tokens, first-decode waits, resident decode service gaps, requested
prefill budget versus actual token counts, and per-repeat outcomes. Distinguish
arrival-to-admission lag from engine first-schedule waiting.

## Decision Rule

Do not require V2 to win every cell. Assess stable completion/correctness,
bounded feasible-class progress, lack of V1-style minimum lock-in, and a clearly
stated static-versus-adaptive trade-off. A behaviorally responsive budget alone
does not prove a performance benefit. Compare paired repeats and uncertainty;
five repeats still do not justify universal claims. Make no automatic default
switch. If evidence supports closing Scheduler, recommend (but do not implement)
EAGLE-3. Otherwise state the specific unmet condition without more tuning here.
