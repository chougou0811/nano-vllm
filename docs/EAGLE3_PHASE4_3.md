# Phase 4.3: Target Verification Diagnostic Profiling

## Scope and Frozen Boundary

Diagnostic-only attribution, not a new performance benchmark or optimization.
Phase 4.1 semantics, Phase 4.2 artifacts, model, scheduler, attention, TP and
sampling remain unchanged. No Git commit and no Phase 4.4 implementation.

Use Qwen3-14B BF16, TP=2, eager, original scheduling and fixed K=3. Profile
mixed-output and long-short at concurrency 1, 2 and 4, and matching ordinary
decodes. Preserve the Phase 4.2 prompt construction and use fresh processes per
system. Capture two early decode/verification steps with all requests active;
these are attribution samples, not full-serving latency distributions.

## Measurement Contract

The benchmark substitutes a diagnostic-only ModelRunner subclass via the engine
module binding. The subclass delegates computation to the frozen implementation.
It adds CPU ranges, CUDA events, module hooks and dual-rank PyTorch profiler.
No mathematical operation, kernel, collective or transaction is replaced.
Profiler start/stop barriers and event synchronization are outside target spans.

- `T_verify_host`: rank-local verification endpoint entry to exit, monotonic ns.
- `T_verify_gpu`: CUDA-event elapsed span on the model stream. Includes device
  idle/wait and is NOT the sum of kernel execution durations.
- `T_cpu_pack_dispatch`: original shared-memory serialization/write/event set.
- `T_rpc_wait`: rank-1 read_shm duration, including idle time awaiting the next
  command. It is NOT an additive verification cost or a pure transport metric.
- `T_device_execution`: interval union of correlated target GPU kernels/copies.
- `T_rank_sync`: observed collective CPU/GPU spans and host synchronization APIs;
  no claim that NCCL kernel duration equals wire-transfer time.

Use CPU operator correlation IDs to associate GPU work with module ranges.
Report summed kernel service time separately from timeline unions. NCCL takes
precedence over enclosing projection modules. Report compute/communication
intersection and device gaps, never add overlapping intervals as wall time.
Hardware occupancy, DRAM bytes, achieved FLOPs and link bandwidth require
hardware counters and must not be inferred as measured from shape data.

## Controls and Reproducibility

Each cell runs warmup, untouched-computation control, event-timed control, and
profiled replay using identical inputs. Drain requests after capture and compare
complete output IDs, verify IDs, cleanup and rank state. Ordinary cache metadata
is recreated only while idle between independent trials; speculative prefix
cache remains disabled. Preserve all diagnostic outliers and profiler errors.

Save raw dual-rank Chrome traces, per-rank host spans/events/collective metadata,
commands, versions, source hashes, environment, and frozen-artifact hashes to
the data disk. Run focused Phase 4.1 and complete CPU tests before and after.
The report must explicitly quantify profiler perturbation and limit conclusions
to the sampled shapes. PyTorch profiler API reference:
https://docs.pytorch.org/docs/stable/profiler.html

## Decision Rule

Rank candidate optimizations by observed critical-path evidence and a clearly
labeled zero-cost ceiling, not by summed overlapping fractions. Prefer one
isolated, testable Phase 4.4 experiment. Do not implement it until user approval.
If serial draft remains secondary to verification, do not start draft batching
merely to increase apparent parallelism.

Results: `benchmarks/eagle3-phase4_3/report.md` and `summary.json`.

## Completed Runs

Completed on 2026-09-23: all 12 cells, six replay modes per cell (warmup,
control, target-event timing, rich profiler, minimal profiler, collective-event
timing). Rich profiling uses module hooks and input shapes. Minimal profiling
omits both. Collective-event mode brackets existing collective stream
dependencies without synchronizing inside the target operation.

The two original CPU suites passed before and after; five new offline-analysis
tests increased the complete suite from 110 to 115. All 168 replay requests
completed and cleaned up; the 14 ordinary/speculative control output pairs
matched. Source and frozen artifact hashes were unchanged.

Two diagnostic qualifications are important:

- An early harness assumed one prefill step could admit 4 x 768 tokens under
  the 2048-token budget. The harness now waits for original prefill scheduling
  to empty the waiting queue. The failed run is retained.
- A supplemental NCCL TUNING-log run emitted per-collective messages. It is
  retained separately; the primary speculative run excludes TUNING logging.

Current hardware topology is SYS across NUMA 0/1, with NCCL SHM/direct/direct
channels. Do not substitute the older project description's NODE topology.

Finding: MLP dominates *compute kernel service*, but verification wall time also
contains substantial eager-host dispatch gaps and peer-arrival waits. NCCL
residency must not be interpreted as pure transfer cost. No same-rank
compute/NCCL overlap was observed in the captured windows.

The recommendation is an approval-gated, fixed-shape target-only launch/replay
experiment, not draft batching or a new attention/NCCL kernel. A future CUDA
Graph prototype would need explicit authorization and correctness checks; none
was implemented in this phase. Phase 4.2 performance results remain unchanged.
