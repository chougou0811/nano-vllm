# EAGLE-3 Phase 4.2 Performance Characterization

## A. Experimental Setup

Qwen3-14B BF16 ran with TP=2 on two RTX 4090 24 GiB GPUs, eager execution,
original Scheduler, `max_model_len=1024`, `max_num_batched_tokens=2048`,
`max_num_seqs=4`, and `gpu_memory_utilization=0.70`. Target, tokenizer, dedicated
EAGLE-3 checkpoint and pinned reference are unchanged from Phase 4.1.

The compared systems are the existing ordinary continuous-batching engine and
the frozen Phase 4.1 concurrent EAGLE engine at fixed K=3. Ordinary uses the
unchanged sampler with temperature `1e-9`; speculative acceptance is greedy
argmax. No CUDA Graph, kernel change, Adaptive K, scheduler change or concurrent
draft batching was introduced.

Each system/concurrency pair ran in a fresh process. The formal matrix contains
180 measured trials: 2 systems x 3 concurrency levels x 6 workloads x 5 repeats.
Every repeat is retained. Workload order rotates and reverses across repeats.

One pre-formal ordinary/c1 instrumentation pilot is retained in the raw directory
but excluded from the formal matrix: its raw spans were valid, but the derived
residual formula did not subtract `target_run_ns`. The formula was corrected and
ordinary/c1 was rerun from a fresh process as `ordinary-c1-v2.json` before any
formal comparison.

## B. Workload Design

The new fixed workloads are short-short 64/32, short-long 64/128, long-short
768/32, long-long 768/128, mixed prompt lengths 64/192/384/768 with output 64,
and prompt 256 with mixed output limits 16/32/64/128.

Concurrency 1 and 2 complete four requests per trial; concurrency 4 completes
eight. A completed request is immediately replaced until the fixed set drains.
Serving throughput is total committed output tokens divided by this single
closed-loop wall-time window. It is never computed from summed request E2E.

Prompt contents are identical between systems and isolated across warmup,
workload and repeat. A unique first token prevents full-block prefix reuse.
Measured prefix-cache hit requests: 0.

## C. Correctness Regression

- Before benchmark: Phase 4.1 focused tests 22/22; complete suite 110/110.
- After benchmark: Phase 4.1 focused tests 22/22; complete suite 110/110.
- All six formal processes exited normally; all 180 trials completed.
- All 480 requests had their requested fixed output length.
- Cross-system final token IDs matched exactly for 480/480 requests. This is an
  observation for these fixed inputs/shapes, not a universal BF16 parity claim.
- Every process ended with zero waiting/running requests, used blocks,
  transactions, host request/draft state and rank-0 target state.
- No NCCL, CUDA or deadlock error occurred. After all work, both GPUs reported
  1 MiB used.

## D. Throughput Results

Values pool all six workloads and five repeats using summed tokens divided by
summed wall time.

| Concurrency | Ordinary tok/s | EAGLE tok/s | Ordinary req/s | EAGLE req/s | Speedup | Paired wins |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.79 | 60.13 | 0.348 | 0.813 | **2.332x** | 30/30 |
| 2 | 42.95 | 99.33 | 0.580 | 1.342 | **2.313x** | 30/30 |
| 4 | 79.43 | 139.37 | 1.073 | 1.883 | **1.755x** | 30/30 |

Paired speedup P50/P95/P99 is 2.289/2.786/2.897 at c1,
2.219/3.206/3.290 at c2, and 1.726/2.350/2.468 at c4. No paired trial loses.

Per-workload aggregate speedup:

| Workload | c1 | c2 | c4 |
|---|---:|---:|---:|
| short-short | 1.940x | 2.113x | 1.647x |
| short-long | 2.230x | 2.252x | 1.559x |
| long-short | 2.191x | 1.800x | **1.430x** |
| long-long | 2.445x | 2.172x | 1.705x |
| mixed-prompt | 2.467x | 2.569x | 1.894x |
| mixed-output | 2.539x | **2.969x** | 2.346x |

The largest tested gain is c2 mixed-output; the smallest is c4 long-short.

## E. Latency Results

P50/P95/P99 in milliseconds, pooled across all workload families and repeats:

| c | System | TTFT | TPOT | ITL | E2E |
|---:|---|---|---|---|---|
| 1 | ordinary | 55.2 / 129.3 / 129.7 | 39.9 / 44.3 / 45.3 | 41.7 / 46.7 / 51.3 | 2341 / 5690 / 5759 |
| 1 | EAGLE | 58.3 / 132.1 / 132.8 | 15.9 / 20.5 / 22.5 | 0.0 / 62.3 / 65.8 | 1026 / 2395 / 2589 |
| 2 | ordinary | 79.2 / 249.2 / 249.7 | 45.5 / 49.6 / 53.7 | 45.7 / 51.6 / 60.5 | 2949 / 6217 / 6501 |
| 2 | EAGLE | 72.6 / 251.3 / 252.5 | 17.8 / 22.0 / 24.2 | 0.0 / 69.6 / 79.1 | 1048 / 2778 / 2904 |
| 4 | ordinary | 200.1 / 500.6 / 508.2 | 46.6 / 54.2 / 56.9 | 45.6 / 51.8 / 65.5 | 3098 / 6493 / 6876 |
| 4 | EAGLE | 130.1 / 502.7 / 504.1 | 24.6 / 33.3 / 41.2 | 0.0 / 91.0 / 131.8 | 1561 / 3686 / 4082 |

EAGLE improves request-level TPOT and E2E. Token-level ITL has a different
shape: multiple committed tokens share one timestamp, producing a genuine 0 ms
median, followed by a longer inter-verification gap. Consequently ITL P95/P99 is
worse, especially at c4. The throughput benefit must not be described as a
universal per-token tail-latency improvement.

TTFT is mostly the same target prefill path. Long-prompt batches dominate its
tail, so c2/c4 P95/P99 remains close between systems.

## F. Concurrency Scaling

Ordinary throughput scales 25.79 -> 42.95 -> 79.43 tok/s. Its target decode cost
per scheduled row falls from 37.83 ms at c1 to 22.47 ms at c2 and 11.78 ms at
c4. This is strong evidence that ordinary q=1 decoding already gains substantial
GPU batching utilization at higher concurrency, even though continuous GPU
telemetry was intentionally not sampled during measured trials.

EAGLE throughput scales 60.13 -> 99.33 -> 139.37 tok/s. Verification time per
target query token falls from 12.33 to 6.59 to 4.01 ms. Median decode request
batch is 1/2/4 and median verification q tokens are 4/8/16. Thus ragged target
verification also amortizes strongly.

Because ordinary batching catches up faster from c2 to c4, relative EAGLE speedup
falls from about 2.31x to 1.75x even though EAGLE absolute throughput continues
to rise.

## G. Acceptance Statistics

| c | Proposed | Accepted | Acceptance | Accepted / verification | Effective outputs / verification | Target forwards | Draft forwards |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7,216 | 6,294 | 87.22% | 2.55 | 3.55 | 2,466 | 7,216 |
| 2 | 7,219 | 6,294 | 87.19% | 4.83 | 6.72 | 1,303 | 7,219 |
| 4 | 14,487 | 12,572 | 86.78% | 9.22 | 12.84 | 1,364 | 14,487 |

Acceptance remains stable around 87%, so scaling is not caused by a changing
acceptance population. Larger request batches turn a similar accepted length per
request into substantially more committed progress per target verification.

## H. Wall-Clock Breakdown

Fractions use the complete speculative serving window, including prefill and
closed-loop replacement. Draft catch-up is a subset of draft time.

| Component | c1 | c2 | c4 |
|---|---:|---:|---:|
| serial draft total | 10.96% | 17.25% | 24.32% |
| draft catch-up subset | 3.60% | 5.40% | 7.51% |
| target verification + TP RPC | 80.82% | 71.40% | 61.10% |
| target prefill | 6.67% | 9.71% | 12.87% |
| accept + commit + postprocess | 0.81% | 0.92% | 1.03% |
| reservation | 0.02% | 0.03% | 0.04% |
| Scheduler | 0.02% | 0.01% | 0.01% |
| residual sync/host | 0.19% | 0.21% | 0.23% |
| driver/refill | 0.47% | 0.40% | 0.31% |

The decomposition accounts for more than 99.9% of each aggregate window.
`target verification` deliberately combines target kernels and shared-memory
RPC/NCCL synchronization; separating them safely would require a perturbing GPU
profiling run. No claim about the internal NCCL fraction is made.

## I. GPU Memory / KV Utilization

| c | Ordinary rank-0 peak allocated | EAGLE rank-0 peak allocated | EAGLE peak reserved | Peak used / capacity blocks |
|---:|---:|---:|---:|---:|
| 1 | 15.69 GiB | 18.68 GiB | 18.78 GiB | 4 / 88 |
| 2 | 15.77 GiB | 18.71 GiB | 19.06 GiB | 8 / 88 |
| 4 | 15.77 GiB | 18.77 GiB | 19.13 GiB | 16 / 88 |

At c4, `nvidia-smi` after measurement reported 20,233 MiB on rank 0 and
17,649 MiB on rank 1, leaving roughly 4.3 GiB and 6.9 GiB of device-memory
headroom respectively. KV occupancy peaks at 18.2% of the configured 88 blocks,
so neither KV capacity nor memory limits the tested c<=4 workloads.

Peak transaction-owned extra blocks are zero in this performance matrix. The
Scheduler's q=1 append reservation already opens the writable block, and these
contexts do not cross another page within K=3. Transaction correctness at page
boundaries remains covered by Phase 4.1 tests; this result must not be read as
proof that tentative blocks are never needed.

## J. Serial Draft Bottleneck Analysis

Serial draft cost clearly grows with concurrency: 10.96% -> 17.25% -> 24.32%.
The proposal-generation portion excluding catch-up is about 7.36%, 11.85% and
16.81%. It is therefore a meaningful secondary cost at c4.

It is not the primary bottleneck. Target verification remains 80.82%, 71.40%
and 61.10% of serving wall time, and is the largest component at every tested
concurrency. The preregistered rule required draft to become the largest named
component before recommending concurrent/grouped draft batching. That condition
is not met.

## K. Non-Speculative vs Speculative Comparison

Phase 4.1 concurrent EAGLE provides a repeatable performance benefit in every
tested cell: 90/90 paired trials win, with aggregate speedup 2.33x, 2.31x and
1.75x at c1/c2/c4. The mechanism is supported by the measurements: about 87%
acceptance reduces target forward count, while batched verification amortizes
larger q shapes. Serial draft and prefill costs reduce the attainable benefit but
do not erase it.

Higher concurrency helps both systems. Ordinary decoding becomes much more
efficient, shrinking EAGLE's relative advantage. EAGLE verification still has
batching value, demonstrated by its decreasing time/query-token and increasing
effective outputs/verification. Serial draft increasingly offsets that value,
but does not yet dominate it.

## L. Limitations

- Ordinary executes its frozen sampler at temperature `1e-9`; EAGLE is greedy.
  Outputs nevertheless matched for all 480 requests in this matrix.
- The workloads are synthetic, pretokenized and closed-loop; there is no network,
  tokenizer, open-loop overload or production arrival distribution.
- Only c<=4, K=3, eager, one target/draft pair and one GPU topology were tested.
- GPU utilization was not continuously sampled because `nvidia-smi` polling can
  perturb short trials. Batching utilization is inferred from measured target
  cost/row and throughput scaling.
- Target verification time is not split into GEMM/attention/NCCL/RPC subspans.
- No fixed K=2/K=4 auxiliary runs were added; the required K=3 matrix was kept
  complete rather than widening scope.
- ITL percentiles reflect burst timestamps by design and are not directly
  interchangeable with ordinary one-token cadence.

## M. Decision For Next Phase

1. **Is there repeatable benefit?** Yes. Every tested regime wins; 90/90 paired
   trials have speedup greater than one.
2. **Largest and smallest benefit?** Largest: c2 mixed-output, 2.969x. Smallest:
   c4 long-short, 1.430x.
3. **Is serial draft the primary bottleneck?** No. It reaches 24.3% at c4, while
   target verification remains 61.1%.
4. **Is there enough evidence to implement concurrent draft batching next?** No,
   not under the preregistered rule. Draft batching may have future value, but it
   is not the largest measured cost.
5. **If it later becomes justified, what is the minimum design?** Group only
   requests with compatible `(D, C, actual_k)` and concatenate their existing
   confirmed draft KV along batch; keep request-owned caches and Phase 2 commit
   semantics unchanged. Do not introduce ragged draft attention first.
6. **What is the next real bottleneck?** Batched target verification, including
   its inseparable TP RPC/NCCL wait. The next step should be a diagnostic-only
   target verification profile that separates attention/GEMM, LM head and
   communication before choosing any optimization.

Phase 4.2 ends as a characterization result. It does not authorize an immediate
optimization implementation.

Raw request/step records and manifests are retained at
`/root/autodl-tmp/eagle3-phase4.2-raw-20260923`. Machine-readable aggregates are
in `benchmarks/eagle3-phase4_2/summary.json`.
