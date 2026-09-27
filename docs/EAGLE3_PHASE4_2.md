# EAGLE-3 Phase 4.2: Performance Characterization Protocol

Status: preregistered before measured GPU trials. Phase 4.1 correctness behavior
is frozen. This phase adds benchmark-only observation and analysis; it does not
implement concurrent draft batching or any serving optimization.

## Questions

1. Does fixed-K concurrent EAGLE reduce serving wall time relative to ordinary
   continuous batching in any tested regime?
2. How does the result change at closed-loop concurrency 1, 2 and 4?
3. Does serial per-request draft work become the dominant wall-clock component?
4. Does ragged batched target verification amortize enough target work to offset
   draft, transaction and host overhead?
5. Does rank-0 memory or paged-KV occupancy constrain concurrency up to four?

## Frozen Systems

Both systems use Qwen3-14B BF16, TP=2, two RTX 4090 GPUs, eager execution,
`max_model_len=1024`, `max_num_batched_tokens=2048`, `max_num_seqs=4`, original
Scheduler, identical target revision/tokenizer and no network/tokenization time.

- **ordinary**: existing continuous batching. Because its frozen SamplingParams
  rejects zero temperature, the benchmark uses temperature `1e-9` and
  `ignore_eos=True`; this is operationally near-greedy but still executes the
  unchanged sampler. This policy difference is reported as a limitation.
- **speculative**: Phase 4.1 `ConcurrentLLMEngine`, dedicated EAGLE-3 draft,
  fixed K=3, greedy target acceptance and speculative prefix-cache disabled.

No Phase 3.1 prompt is reused. Each process has isolated prompt contents across
warmup, repeats and workloads. A unique first token prevents full-block prefix
reuse while preserving identical inputs between the two systems.

## Workloads

Six fixed workload families are used:

| Workload | Prompt tokens | Output tokens |
|---|---:|---:|
| short-short | 64 | 32 |
| short-long | 64 | 128 |
| long-short | 768 | 32 |
| long-long | 768 | 128 |
| mixed-prompt | 64, 192, 384, 768 | 64 |
| mixed-output | 256 | 16, 32, 64, 128 |

At concurrency 1 and 2, each trial completes four requests; at concurrency 4 it
completes eight. A finished request is replaced immediately until the fixed set
is exhausted. Serving throughput is total committed output tokens divided by the
single closed-loop measurement window, never by summed request latencies.

Every system/concurrency pair runs in a fresh process. Each workload has an
unmeasured shape warmup and five measured repeats. Workload order rotates by
repeat and reverses on alternating repeats. No observation is removed.

## Measurements

Raw request records contain arrival, first schedule, every committed-token time,
finish, output IDs and step counts. Raw step records contain Scheduler duration,
batch size, target query rows, draft/proposal and catch-up time, KV reservation,
target verification RPC/forward, acceptance, rank commit RPC, host block commit,
postprocess, residual host/synchronization time, used blocks and rank-0 allocated
and reserved memory.

Reported distributions are linear P50/P95/P99 for TTFT, request TPOT, token ITL
and E2E. Aggregate results also include output tokens/s, requests/s, proposal and
acceptance counts, accepted/verification, effective outputs/verification, target
and draft forwards, target query rows, Scheduler batch size, KV occupancy and
memory.

No extra CUDA synchronization is inserted. `runner.call("eagle3_batch",
"verify", ...)` is reported as combined target execution plus TP RPC/NCCL
synchronization because safely separating them would perturb the measured path.
The draft's existing synchronization remains unchanged.

For speculative decode steps:

```text
T_step = T_schedule + T_draft + T_reserve + T_target_verify
       + T_accept_commit + T_sync_other
```

`T_accept_commit` includes vector acceptance, target commit RPC, host transaction
commit and Scheduler multi-token postprocess. `T_sync_other` is the nonnegative
residual, including Python packing, coordinator bookkeeping and synchronization
not attributable without new barriers. Admission/refill time outside engine
steps is reported separately as driver overhead.

Fractions use the complete serving measurement window, including prefill and
closed-loop refill:

```text
draft_fraction  = sum(T_draft) / serving_wall_time
verify_fraction = sum(T_target_verify) / serving_wall_time
speedup         = ordinary_wall_time / speculative_wall_time
```

## Decision Rule

Recommend grouped/concurrent draft batching only if all are observed:

1. draft fraction rises materially with concurrency and is the largest measured
   speculative wall-clock component in the relevant regimes;
2. target verification shows effective batching amortization, such as lower
   verification time per target query/output or a decreasing verify fraction;
3. speculative serving has a repeatable benefit, or a clear near-break-even
   regime whose remaining loss is quantitatively explained by serial draft;
4. rank-0 memory headroom is sufficient for the smallest grouped implementation.

If draft is not dominant, Phase 4.2 must not recommend batching it. The next work
must target the largest measured component instead. A component is called
dominant only when it is the largest named share of the serving wall time, not
merely when its absolute duration increases.

Results must keep every repeat and report paired speedup separately for
concurrency 1, 2 and 4 and each workload. No performance conclusion is allowed
until all planned repeats finish and post-run correctness/cleanup checks pass.

## Final Status

The complete preregistered matrix finished: 180/180 formal trials, with five
repeats in every system/concurrency/workload cell and no filtered observation.
Post-run correctness and cleanup gates passed. Aggregate ordinary/speculative
wall-time speedup is 2.332x at concurrency 1, 2.313x at concurrency 2 and 1.755x
at concurrency 4; all 90 paired comparisons favor speculative execution.

Serial draft fraction rises from 10.96% to 17.25% to 24.32%, while target
verification remains the largest component at 80.82%, 71.40% and 61.10%.
Accordingly, the preregistered dominance condition for concurrent draft batching
is not met. Phase 4.2 recommends profiling batched target verification next,
without implementing an optimization in this phase. Full results are in
`benchmarks/eagle3-phase4_2/report.md` and `summary.json`.
