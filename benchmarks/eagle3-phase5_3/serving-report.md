# Serving Validation

## Setup and Estimands

Pinned Qwen3-14B BF16, TP2, two RTX4090 GPUs, original Scheduler, K3,
persistent per-request draft state and frozen target transactions. Eager target;
MLP replay OFF in both systems. Prefix cache disabled. Catch-up stays serial.
`draft_batching=False` remains the default. Full configuration, software, model
revisions, source hashes and commands are linked from [summary.json](summary.json)
and [commands.md](commands.md). Raw data is outside the repository at
`/root/autodl-tmp/eagle3-phase5.3-20260925/`.

Closed-loop serving replaces completed requests at step boundaries. Each cell
has one warmup run/system and five measured repeats, rotating system order. Inputs
and seeds match within each pair. There are ten prespecified families at c1/2/4:
seven retained families plus copy-pattern, observatory-prose and routing-table.
The main process completes 300 measured trials (150 pairs). An independent
fresh process repeats heldout-mixed and routing-table with reversed order:
60 trials (30 pairs). No timing outlier is removed. Warmups use short outputs
(2-6 tokens in retained families; at most 6 in new families), not exhaustive
coverage of every later context/shape. Isolated generation has two warmups/call;
those must not be confused with serving warmup counts.

Throughput is committed output tokens / measurement wall time, not sum of
request latencies. The headline speedup is the geometric mean of paired
serial-wall / batch-wall ratios. Pooled throughput weights long trials more
heavily and is therefore a different estimand. Neither metric implies statistical
independence of all requests sharing one process.

## Main Results

| Concurrency | Paired geomean speedup | Improving pairs | Serial pooled tok/s | Batch pooled tok/s |
|---|---:|---:|---:|---:|
| 1 | 0.9973x | 28/50 | 50.302 | 49.519 |
| 2 | 1.0944x | 47/50 | 72.060 | 78.879 |
| 4 | 1.2216x | 50/50 | 96.328 | 122.603 |

Fresh-process geomean speedups for heldout-mixed are 1.0016/1.0641/1.1571x
at c1/2/4; routing-table gives 1.0176/1.0762/1.2545x. Both c4 cells improve
in all five repeats; c2 improves in four/five and five/five respectively.
See [throughput-table.md](throughput-table.md) for every family, including
negative cells. c4 copy-pattern has the largest main speedup (1.3714x), while
c1 long-long has the smallest (0.9416x). This ranking is descriptive, not tuning.

## Latency and Numerical Qualifications

[latency-table.md](latency-table.md) includes TTFT/E2E/request-TPOT/token-ITL
P50/P95/P99. Committed tokens in one verification share a timestamp, so
intra-burst ITL can be zero; these are not separate GPU kernel completion times.
The per-cell table retains P95/P99 ratios rather than hiding them in pooling.
c2/c4 generally improve, but main c1 long-long and copy-pattern have median
P95 ITL ratios of 1.312 and 1.230. c1 executes the unchanged serial path with
no batched feedback calls. That fact alone does not explain away observed tails.
A targeted independent c1 check is explicitly post-hoc diagnostic, not a new
unbiased held-out selection; its results remain separate in the tables.

The final no-overlap c1 diagnostic has ten pairs. Long-long gives0.9857x
geomean (4/5 repeats improve), with median repeat P95/P99 ITL ratios0.9821/0.9704.
Copy-pattern gives1.0027x (2/5 improve), with ratios1.0018/1.0391. All ten
trajectories match exactly and clean up. One long-long repeat dominates its
geomean despite four improving repeats. This supports run-to-run variance and
the absence of a batching mechanism at c1; it does not delete the main c1
long-long0.9416x result or prove a universal <=3% regression bound.

All main and fresh paired final outputs match. Strict proposal/target/acceptance
path signatures do NOT universally match: 37 main pairs vary. Actual acceptance
is recomputed using frozen greedy semantics at every step. Same-state numerical
audits are reported separately; no signature failure is relabeled as exact parity.
Across processes, all 60 matching final outputs agree, but 25 full signatures
differ, all in routing-table and including frozen serial execution. Differences
in prior execution history are a possibility, not an established explanation.
See [correctness-report.md](correctness-report.md).

## Mechanism and Attribution

| c | Generation serial/batch seconds | Catch-up serial/batch seconds | Target verify serial/batch seconds | Draft model calls serial/batch |
|---|---|---|---|---|
| 1 | 26.623 / 26.611 | 12.251 / 12.388 | 253.780 / 258.729 | 18270 / 18270 |
| 2 | 26.279 / 17.146 | 11.510 / 11.361 | 162.595 / 152.775 | 18154 / 13477 |
| 4 | 52.338 / 18.907 | 22.321 / 22.160 | 217.336 / 180.764 | 36404 / 19956 |

These are sums across measured main trials, not per-step latencies. Generation
includes packing, masks, copies, token reads, mapping and publication, but not
serial catch-up. It decreases about 34.75% at c2 and 63.88% at c4. Effective
feedback batch sizes are 1.636 and 3.138, below active concurrency because
requests finish, clip K or enter different serving phases. Physical model calls
include serial catch-up; request-level attribution must not double-count shared
forwards. Raw JSON retains both counts and batch-size sequences.

Target forwards are 6233/6233, 3729/3729, 3890/3892 (serial/batch at c1/2/4).
Mean target verification wall is 40.716/41.510, 43.603/40.969, 55.871/46.445 ms.
Target code is unchanged, yet its measured wall decreases substantially at c4.
Thus the complete serving gain cannot be attributed solely to faster draft GEMM
or fewer draft calls. Changed execution cadence, rank waits and GPU/host state
remain possible contributors; this experiment does not isolate which one.
The isolated generation experiment establishes the draft mechanism independently.

## Acceptance, Memory and Cleanup

Main accepted/proposed is approximately 0.5215/0.5215 at c1,
0.5270/0.5269 at c2 and 0.5248/0.5248 at c4. Aggregate similarity does not imply
per-iteration proposal or acceptance-length identity. JSON also reports proposed,
accepted, accepted/verification, effective outputs/verification and all forwards.

The new workload acceptance hypotheses were wrong: serial copy-pattern is about
0.215, observatory-prose 0.105 and routing-table 0.498. They remain fixed and
are not renamed or replaced to improve performance. The retained families supply
high acceptance (approximately 0.759-0.990), so the full corpus still spans low,
medium and high observed acceptance. No new prompt was tuned after measurement.

Rank0 main peak allocation stays near 18.80-18.91 GiB in both systems; peak
reserved memory is about 19.49 GiB. Post-trial allocated bytes are constant at
19,566,306,816 on rank0 and 16,769,829,376 on rank1. Minimum measured post-trial
free memory is about 3.40/6.02 GiB. This is post-trial headroom, not guaranteed
instantaneous minimum free memory. Scratch and peak memory by rank are in JSON.
All completed trials release used blocks, transactions and target/draft state;
prefix hits remain zero. No progressive allocated-memory growth is observed.
No concurrency above four is tested.

## Preserved Errors and Limits

Earlier partial/error manifests are retained separately rather than combined
with complete five-repeat cells. They include a benchmark closure error, a
tuple/list assertion, the first strict BF16 proposal failure, a disconnected
fresh process and its subsequent orphan-shared-memory setup failure. Recovery
terminated only that failed process group and unlinked its verified orphan
segment. The successful fresh process is a new directory, not an overwrite.
No completed measured inference run reports NCCL/CUDA failure or state leakage.
This is not a claim that the entire experiment had no errors.

The independent audit adds full KV hashing and local serial oracles, so its
timings are excluded by design, not treated as serving performance samples.
No MLP replay combination, catch-up batching, adaptive K or secondary technology
was added after seeing results. The adoption decision is in
[final-report.md](final-report.md).
