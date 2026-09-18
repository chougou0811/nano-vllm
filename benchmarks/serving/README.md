# Serving benchmark

Independent, rank-zero, in-process measurement adapter. Production code is not
modified. `LLMEngine.add_request -> Scheduler.add -> LLMEngine.step ->
Scheduler.schedule -> ModelRunner.run -> Scheduler.postprocess -> Sequence.append_token`
remains the original execution path. A delegating scheduler proxy records schedule
returns and inspects actual Sequence growth after original postprocess returns.

Run from the repository root with `python -m benchmarks.serving --help`.
Required arguments: `--model`, `--model-revision` (40-character commit SHA), and
`--output-dir` (new directory). Supply `--model-manifest` to verify the existing
download manifest's file checksums against the claimed revision. Without that
manifest the revision is explicitly marked unverified; local SHA256s are saved.

Example (substitute pinned local model and revision):

```bash
python -m benchmarks.serving --model /data/models/Qwen3-14B \
  --model-revision MODEL_COMMIT_SHA --output-dir /data/results/new-run \
  --tensor-parallel-size 2 --dtype bfloat16 --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 --num-requests 12 --prompt-length 128 \
  --output-length 32 --concurrency 4 --arrival-interval-ms 100 \
  --length-mode exact --warmup-requests 4 --verify-observer
```

All timestamps are `perf_counter_ns` offsets from the measured workload origin.
Logical arrivals are fixed at `i * interval`; FIFO admission happens between
engine steps subject to the concurrency cap. Delayed admission never shifts
arrival times. This is not an HTTP server or a measurement of actual network
receipt. Inputs are reproducible synthetic vocabulary IDs, not natural-language
quality tests; input/output text processing is outside timing.

- Queue delay: first scheduled - arrival; split into admission - arrival and
  first scheduled - admission. Initial queueing only, not total preemption wait.
- TTFT: first committed output - arrival.
- Request TPOT: (last - first token time) / (output tokens - 1); null for <=1 token.
- Token ITL: adjacent committed-token intervals, pooled across completed requests.
- E2E: finish - arrival. Linear percentiles use completed requests only.
- Throughput window: first logical arrival to final completion or failure,
  including idle arrival gaps and drain, excluding startup and warmup. Request
  throughput counts completions; output throughput counts all committed tokens,
  including EOS and partial failed output. No prompt tokens are counted.
- Optional TTFT/TPOT SLOs are conjunctive. Failures/timeouts violate configured
  SLOs; denominator is arrived requests. Goodput is compliant requests / window.
  No configured SLO yields null metrics. One-token TPOT is not applicable.

`--length-mode exact` uses the original `ignore_eos=True`; `eos` allows early
termination. Sampling temperature and torch seed are recorded. No sampler changes
or extra CUDA synchronization are introduced. Only actual Sequence append deltas
are counted, never prefill-discarded samples or speculative proposals. A future
multi-token commit shares one timestamp, yielding zero within-burst ITL.

Original constructor warmup, explicit warmup, and optional observer parity checks
are excluded. Prefix cache remains intact; warmup tokens and exact workload are
saved. Parity compares two short fixed-burst runs with identical seeds and no full
prefix-cache blocks: original vs observed schedule decisions, output token IDs,
and released KV blocks. Baseline tracing is only enabled during parity, never
measurement. Bookkeeping elapsed host time is reported, but instrumentation can
still perturb timing and boundary admission. No zero-overhead claim is made.

Outputs: requests.json/csv, tokens.csv, steps.json, summary.json, report.md,
manifest.json, workload.json, warmup.json, source_snapshot/, source.diff, and
pip_freeze.txt. Snapshot includes untracked framework code. Model hashes,
effective config, software versions and GPU topology are saved. Output directories
cannot be overwritten. Initialization failures save error.json; step failures save
partial request records. The timeout is checked between engine steps and cannot
interrupt a hung CUDA/NCCL call; use an external process timeout as a hard guard.

Tests: `python -m unittest discover -s tests -p 'test_serving*.py' -v`.
Initial Qwen3-14B runs validate the framework, not comparative performance.

## Version 2: open-loop and repeated validation

`--arrival-mode concurrency-gated` preserves the original benchmark gate;
`--arrival-mode open-loop` admits every due request at the next step boundary,
regardless of active count or max_num_seqs. Concurrency is ignored in open-loop.
The synchronous engine is not thread-safe; no concurrent scheduler mutation is
introduced. Logical arrivals stay fixed, and step-boundary admission lag remains
visible. The original scheduler owns its waiting and running queues.

`arrival -> admitted` is benchmark admission lag, including closed-mode gating.
`admitted -> enqueued` is submission overhead, ending after original Scheduler.add.
`enqueued -> first_scheduled` is engine initial waiting. The old compatibility
metric `scheduler_queue_delay_ms` includes submission overhead. Never attribute
the entire `queue_delay_ms` to the Scheduler. Queue snapshots and waiting-token
counts are saved before schedule, after schedule and after token commit. Running
requests may also await decode; initial waiting is not cumulative scheduling delay.

`--repeats 3` retains one engine but resets torch sampling seeds before each trial.
Workload lengths, distributions and arrival offsets are identical; token content
varies and every request has a distinct first token across all trials and warmup.
This prevents shared full prefixes without clearing or modifying prefix cache.
The original BlockManager.allocate call is delegated unchanged and actual cached
block counts are observed, separating initial reuse from later reallocations.
Nonzero initial reuse fails isolation validation. `--prefix-probe` adds two rounds
of block-size-plus-one prompts; positive cache-hit controls are CPU unit tests.

`--warmup-mode legacy` reproduces the earlier warmup pattern. `decode-shapes`
runs batches 1..max_num_seqs with the requested prompt length and full output span
(at least two tokens). Actual decode shapes and prefix reuse are validated and
saved in warmup/; all warmup is outside the measured windows. This covers these
shapes, not arbitrary future shapes, compiler guards, or memory layouts.

Each repeat gets requests.json/csv, tokens.csv, steps.json/csv, summary.json and
report.md. Steps include batch size, context lengths, block-table widths, scheduling
start, scheduling return, token commit and step return. Root manifest is authoritative
for final exit/cleanup; each trial manifest is a pre-cleanup snapshot. Root summary
and report aggregate repeats. `--burst-probe` adds a zero-interval open-loop check.

`--diagnostics` adds rank-zero CPU spans around original runner methods and sampler,
GC callbacks, process CPU time/context switches/fault counts, and external GPU
telemetry at roughly 200 ms. It does not add CUDA synchronization. Its host spans
cannot alone distinguish GPU kernels from NCCL waiting; compile logs and, if
necessary, a separate GPU trace are needed. Diagnostic timings are perturbed and
must not be treated as ordinary trials. Enable `TORCH_LOGS=recompiles,dynamo,inductor`
before process launch to capture compiler evidence from both ranks.

`python -m benchmarks.serving.suite --help` runs serial A/B/C groups (legacy closed,
shape-warmed closed, shape-warmed open), each with three repeats, and a separate
diagnostic D group. It clones the same existing compiler cache snapshot per group,
records its inventory, and leaves the original shared caches untouched. A fresh
process is not a cold-cache experiment. Failed/slow samples are never removed.

## Scheduler Policy Experiments

The default remains `--scheduler-policy original`. Opt-in `static` and `slo-aware`
select new scheduler policies; they do not change the model, TP or sampling code.
See `docs/SCHEDULER.md` for policy design and the first pilot's limitations.
Use `--arrival-mode open-loop` to expose engine waiting pressure.

Control targets `--scheduler-ttft-ms` / `--scheduler-tpot-ms` are distinct from
scoring thresholds `--ttft-slo-ms` / `--tpot-slo-ms`. Specify both pairs explicitly
for comparisons. `--scheduler-prefill-chunk` controls static chunk size;
`--scheduler-min-prefill-chunk` is the SLO policy's minimum budget.

`python -m benchmarks.serving.scheduler_experiment --help` lists the reproducible
mixed-workload pilot arguments: three policies, two arrival intervals, three
counterbalanced repeats, exact workload records, prefix metadata resets and
untimed shape warmup. The script first checks teacher-prefix logits. Its default
strict raw-logit guard currently fails on a repetitive 14B input, including the
unchanged original-chunk256 control. `--exploratory-numerics` retains that failure
explicitly and permits exploratory timing only; it is not a correctness pass.
All ordinary performance runs use the original sampler.

Scheduler V2 is opt-in with `--scheduler-policy slo-v2`; `slo-v1` aliases the
unchanged `slo-aware` V1. `original` is still default. V2 progress/cost/overload
controls are exposed as `--scheduler-v2-*`. See `docs/SCHEDULER_V2.md` for the
algorithm, progress limits, revised numerical protocol and frozen held-out suite.
`benchmarks.serving.scheduler_v2_correctness` captures logits and independent
references; `benchmarks.serving.scheduler_v2_experiment` requires accepted 14B
and small-model reports before running tuning, sweep and ablations. V1 artifacts
are not overwritten. Metric-revision failures are kept as separate evidence.
