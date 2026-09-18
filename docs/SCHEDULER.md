# Scheduler Policies

## Architecture and Scope

The original runtime runs one homogeneous prefill or decode batch per step.
Rank zero chooses sequences and sends existing Sequence state through shared
memory to TP workers; all ranks execute the same model operations. No mixed
prefill/decode kernel batch is introduced. KV allocation still reserves complete
prompt blocks even for chunked prefill, so token chunking alone is not a memory
admission policy. Original prefill-first scheduling can delay decode continuously,
and its extendleft queue update can repeatedly select the same running requests.
`max_num_seqs` limits a scheduled batch, not the total running queue. Open-loop
trials can therefore have more resident running requests than that setting.

`scheduler_policy="original"` constructs the unchanged Scheduler class. The
default and original schedule/postprocess/add code remain unchanged. No extra
clock is used by the original policy. New static and SLO policies derive from
Scheduler and reuse its commit, EOS and KV release behavior. Sequence TP
serialization, model execution, attention, TP collectives and sampling are intact.

## Policies

`static` uses a fixed prefill token budget (default 256), alternating one prefill
step with one decode batch while both types are available. Decode rotates serviced
requests to the tail. It shares the new policy's memory admission and fairness
machinery, providing a comparison for the adaptive arbitration rather than only
comparing a completely different scheduler with the original.

`slo-aware` maintains rank-zero request timing state (arrival, first/last committed
token, last service step) and EWMAs of measured decode step time and prefill time
per token. Samples include host scheduling/model/commit time, not CUDA kernel
time. Defaults: EWMA alpha=0.2, initial decode estimate=50 ms, initial prefill
rate=50 ms / 256 tokens. These are configurable seeds, not model-size constants.

- TTFT slack: arrival + TTFT target - now - estimated remaining prefill cost.
- Decode slack: earliest last-token + TPOT target - now - estimated decode cost.
- The TPOT target is used as a next-token interval deadline. This is a stricter
  control signal than a request's mean TPOT, while reporting preserves both
  request TPOT and token ITL. No future output lengths or future arrivals are
  inspected for urgency; output limits are used only for memory reservations.
- Choose the largest prefill budget in a doubling grid from minimum chunk
  (default 128) through max_num_batched_tokens that fits the estimated decode
  slack, including a reserved decode step. Prefill prediction is conservatively
  floored by the decode estimate. The entire budget is charged, not just the
  first request's remaining tokens, because a prefill batch can include others.
- If no chunk fits, decode normally wins. An already-expiring TTFT can win one
  minimum chunk if its normalized negative slack exceeds decode's pressure.
- While both classes are runnable, at most 2 consecutive prefill steps and 4
  consecutive decode steps are allowed. A forced prefill still needs feasible KV
  admission; otherwise decode drains residents. These are progress safeguards,
  not promises that all deadlines are feasible under overload.
- Waiting requests stay FIFO. Decode selects earliest next-token deadlines;
  with the uniform target this is oldest-token-first, preventing an overdue
  request from monopolizing service after it has just received a token. Static
  uses rotation instead of deadlines. Neither drops, aborts, nor skips requests
  simply because they have already missed an SLO.

## KV Safety and Fairness Limits

New-policy admission reserves enough free capacity logically for all resident
requests to reach their configured output limits, plus the candidate's full
prompt/output capacity. Blocks are still allocated by the existing BlockManager
on demand. Prefix-sharing savings are conservatively ignored for this capacity
check, but actual prefix reuse still works. An individually impossible request
is rejected before insertion. A reservation-invariant failure raises an explicit
error rather than mutating KV state to recover silently.

Consequently new-policy normal execution does not require preemption. The
original policy retains its original preemption behavior. Under memory pressure
the new admission rule can reduce concurrency and increase TTFT, especially when
max_tokens grossly overestimates actual EOS length. FIFO waiting prevents younger
small requests continually bypassing a memory-blocked older request. Progress
requires finite feasible requests; there is no finite latency guarantee for
unbounded overload. No KV implementation change is made.

## API and Observation

Config adds scheduler_policy, scheduler_prefill_chunk, scheduler_min_prefill_chunk,
scheduler_ttft_ms, scheduler_tpot_ms, scheduler_initial_step_ms,
scheduler_ewma_alpha, scheduler_max_prefill_steps and scheduler_max_decode_steps.
`LLMEngine.add_request(..., arrival_time_ns=...)` optionally accepts a timestamp
in the same perf_counter_ns clock domain. This host-only attribute is omitted by
existing Sequence serialization. Without it, new policies use scheduler admission
time. Original policy ignores it. The benchmark passes logical arrival offsets
converted to this monotonic domain; engine waiting and external admission lag
remain separately reported.

The serving CLI exposes policy, chunk and SLO target options. Its observer records
policy reasons, chosen budget, predicted costs/slacks, logical KV reservations
and service-gap counters. Token timestamps still only count actual commits.

## Validation and Experiment Design

CPU tests compare the original factory with direct original construction, test
block boundaries/EOS/discarded intermediate samples, logical arrivals, round-robin
and EDF selection, deadline arbitration, class progress guards, prefix reuse,
limited-memory draining, and randomized finite workloads.

GPU correctness uses test-only teacher forcing of four identical output tokens
per request, comparing full-vocabulary committed-position logits for lengths
31/255/256/257/511/512 across original/static/SLO. The predeclared BF16 maximum
absolute-error guard is 0.5; top-1 agreement is reported separately. This controls
RNG reassignment when batch order changes. No production sampling code is changed;
ordinary performance runs use the original sampler at temperature 0.6.

The `benchmarks.serving.scheduler_experiment` module runs a counterbalanced
pilot with one engine (`--help` lists its arguments). Every trial constructs a fresh Scheduler and BlockManager
metadata, explicitly resetting prefix-cache lookup state while reusing allocated
GPU KV storage. The model overwrites its assigned slots before using them.
Corresponding trials use identical prompts across policies; warmup prompts are
separate and all actual initial prefix hits must be zero. Measured patterns are
six repetitions of (128 input,64 output)/(1536 input,16 output), at deterministic
350 ms and 100 ms arrival intervals, 3 repeats, rotating policy order. Targets
are TTFT 2000 ms and TPOT 100 ms, set before inspecting results. All policies use
the same maximum batch/token limits and open-loop driver. Shape warmup and policy
EWMA calibration occur outside measurement; no sample is filtered.

Limitations: simple cost model (not context-aware or per-batch-size regression),
global per-request SLO targets, no optimal admission or overload controller,
conservative max-output memory reservation, CPU sorting/snapshot overhead, and
step-boundary arrivals rather than a threaded network frontend. Compare adaptive
against static to separate arbitration effects from new shared safety/fairness
mechanisms. Small repeated trials cannot establish broad performance improvements.

## Initial Findings and Safety Status

The first 14B TP=2 experiment did NOT establish a generally better scheduler.
Static was more balanced on the tested workloads. SLO reduced extreme ITL tails
but increased TTFT and lowered throughput; at 100 ms arrivals its SLO goodput
regressed substantially. Original remains the default. Deadline arbitration,
cost prediction and minimum-chunk overhead need held-out ablations, not tuning
and evaluating on the same pilot data.

The strict 0.5 raw-logit guard failed on a 511-token repetitive shared-prefix
input. The unchanged original with chunk256 also fails against the original
full-prefill reference; independent HF BF16 eager shows original-vs-HF error
above 3 at that position as well. All 24 top-1 comparisons agree, but this is NOT
proof of full numerical equivalence or a definitive root-cause diagnosis.
`--exploratory-numerics` explicitly allows a timing experiment while preserving
`strict_logits_pass=false`; the default still stops. Non-finite logits, missing
commits, top-1 disagreement or leaked KV always fail. No tolerance is enlarged
and no sample is removed. Test-only sampler replacement is always restored.

`--correctness-only --correctness-corpus varied` adds non-shared-prefix inputs
through 2044 tokens without replacing the original failing corpus. Independent
references can be reproduced with `benchmarks.serving.scheduler_hf_reference`.
The full local report, raw data, source snapshots and exact commands are under
`/root/autodl-tmp/benchmarks/scheduler-v1/`; these are separate from prior baselines.

The expanded varied-input test also fails strict identity: static/SLO and the
unchanged original-chunk256 control agree with original top-1 at 31/32 positions.
The differing position has nearly tied BF16 logits. Independent HF agrees with
static/SLO/control at 32/32 and original at 31/32. Raw-logit discrepancies remain
above the 0.5 guard even for original-vs-HF. These finite comparisons support
numerical sensitivity but do not establish a complete root cause. No model,
attention or sampling change was made to hide or repair the discrepancy.
