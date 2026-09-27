# Isolated Draft Generation

Target-eager TP2 captured real confirmed features/KV at255/256/257 and
511/512/513/1025 prompt boundaries, across three successive speculative
iterations. The isolated experiment then uses the identical pinned drafter on
rank0's GPU, without concurrently running the target. Five paired repeats/cell,
two warmups, alternating serial/batch order; all90 measured calls retained.

Generation timing starts after serial catch-up and includes packing/copies,
mask/position construction, heads, feedback forwards, token reads and gathering.
It excludes model loading, warmup, diagnostic hooks and FP32 controls. Isolated
calls do not publish request state; only validation of publication preconditions
is timed. Full serving includes the cursor/KV publication cost. These numbers
are not whole-serving speedups.

| Batch size | Serial median ms | Batch median ms | Pooled wall reduction | Feedback forwards | Total forwards incl. catch-up |
|---|---:|---:|---:|---|---|
| 2 | 7.884 | 4.315 | 45.16% | 4 -> 2 | 6 -> 4 |
| 3 | 11.838 | 4.407 | 62.75% | 6 -> 2 | 9 -> 5 |
| 4 | 15.692 | 4.576 | 70.17% | 8 -> 2 | 12 -> 6 |

These are three prespecified context/history cells per batch size, five repeats
each, not15 independently sampled workloads. Strong c2/c4 stage gains justify
the serving experiment but do not satisfy its correctness/adoption gates alone.

## Packing and Submission

Mean host spans, ms (GPU work can execute later and overlap):

| B | Pack/pad/copy submission | Mask/positions | Gather | Model submission | Head/read including waits |
|---|---:|---:|---:|---:|---:|
| 2 | 0.116 | 0.135 | 0.0008 | 1.558 | 2.395 |
| 3 | 0.151 | 0.142 | 0.0008 | 1.670 | 2.407 |
| 4 | 0.198 | 0.150 | 0.0008 | 1.689 | 2.597 |

Pack+mask+gather submission accounts for approximately5.96/6.68/7.44% of batched
generation wall at B2/3/4. These are not separately synchronized GPU-copy costs.
No feedback-KV scatter occurs: returned IDs are mapped into per-request lists,
and independently allocated confirmed tensors remain the persistent state.

Additional profiler runs, excluded from timings, show kernel counts
334/501/668 ->181/185/189. GPU kernel service sums are about5.680/8.500/11.441ms
serial vs3.070/3.073/3.500ms batch. CUDA API interval time also falls. GPU service,
API intervals and wall time overlap and must not be added together. CUDA-event
stream intervals in raw trials are not described as GPU active time.

Initial packed KV scratch maxima are about2.06/3.11/16.14MiB for B2/3/4.
Measured draft-only peak allocation increases by approximately9.13/16.56/83.79MiB
relative to serial, including transient downstream allocations. Neither metric
is the two-rank serving memory footprint. Catch-up remains serial and its cost
is reported separately in serving data.

Raw: `isolated/manifest.json` and six separate profiler traces under the phase
raw root. See [summary.json](summary.json) for exact counts and bytes.
