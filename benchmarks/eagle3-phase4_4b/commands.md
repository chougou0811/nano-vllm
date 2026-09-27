# Phase 4.4B Commands and Reproduction

Working directory: `/root/autodl-tmp/projects/nano-vllm`.
Python: `/root/autodl-tmp/venvs/nano-baseline/bin/python`.
Raw root: `/root/autodl-tmp/eagle3-phase4.4b-20260924`.
All two-GPU commands run sequentially. Do not launch competing GPU experiments.

## CPU Regression

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest -v tests.test_eagle3_concurrent_state tests.test_eagle3_block_transactions tests.test_eagle3_batched_runtime tests.test_eagle3_scheduler_adapter
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest discover -s tests -q
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest -v tests.test_eagle3_graph_cache
```

The initial full suite had 122 tests; the new graph policy suite adds eight.
No historical test was edited. Test fixtures may print expected negative-result
JSON; the unittest final status, not an embedded fixture's `passed` field, is
the suite result.

## GPU Correctness Gate

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1 /root/autodl-tmp/venvs/nano-baseline/bin/python -u -m benchmarks.serving.eagle3_phase44b --stage correctness --output /root/autodl-tmp/eagle3-phase4.4b-20260924/correctness-final
```

The output directory must not already exist. The retained pilot and earlier
gate live in `pilot/` and `correctness/`; they are not overwritten or counted
as measured serving trials. The final gate includes uniform replacement waves
that exercise actual M4/M8/M16 graph hits, not merely supported-shape misses.

## Measured Serving Matrix

For each `C` in 1, 2, 4, execute the following commands separately. The actual
process order is ordinary-c1, speculative-c1, speculative-c2, ordinary-c2,
ordinary-c4, speculative-c4; `pipeline.json` records commands and exit codes.

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1 /root/autodl-tmp/venvs/nano-baseline/bin/python -u -m benchmarks.serving.eagle3_phase44b --stage serving --gate /root/autodl-tmp/eagle3-phase4.4b-20260924/correctness-final/manifest.json --output /root/autodl-tmp/eagle3-phase4.4b-20260924/run-ordinary-c${C} --concurrency ${C} --systems ordinary
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1 /root/autodl-tmp/venvs/nano-baseline/bin/python -u -m benchmarks.serving.eagle3_phase44b --stage serving --gate /root/autodl-tmp/eagle3-phase4.4b-20260924/correctness-final/manifest.json --output /root/autodl-tmp/eagle3-phase4.4b-20260924/run-speculative-c${C} --concurrency ${C}
```

Each process warms all six workloads, then runs five measured repeats in
rotated/reversed workload order. Speculative mode order also rotates. The
default modes are eager, never, second4, second2 and second8. Capacities 2/8
are restricted to the preregistered long-long/mixed-output sensitivity cells.
There are 420 measured trials in the primary and capacity matrices combined.

Every trial records inputs, output IDs, committed-token times, request/step
metrics, graph events, exact keys, capture lifetimes and both-rank memory.
Manifests include model revisions, environment, topology, source hashes and
the precise command. The source hash gate rejects changes after correctness.
The offline analyzer alone is excluded from the measured source hash set.

## Literal Default-runner Control

After the primary matrix, run the frozen Phase 4.2 CLI in a fresh process for
each `C` in 1, 2, 4. This creates 90 supplemental trials with no graph runner or
Gloo graph-control group at all. The primary same-process eager comparator has
GraphCache disabled but uses the opt-in runner; this cross-check makes that
distinction explicit. No historical benchmark script is modified.

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1 /root/autodl-tmp/venvs/nano-baseline/bin/python -u -m benchmarks.serving.eagle3_phase42 --system speculative --concurrency ${C} --output /root/autodl-tmp/eagle3-phase4.4b-20260924/frozen-eager-c${C}.json
```

## Offline Analysis

After the GPU processes exit, rerun regression and save the logs used by the
analyzer's completion gate:

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest -v tests.test_eagle3_concurrent_state tests.test_eagle3_block_transactions tests.test_eagle3_batched_runtime tests.test_eagle3_scheduler_adapter > /root/autodl-tmp/eagle3-phase4.4b-20260924/post-focused.log 2>&1
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest discover -s tests -q > /root/autodl-tmp/eagle3-phase4.4b-20260924/post-cpu.log 2>&1
```

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44b_summary --raw /root/autodl-tmp/eagle3-phase4.4b-20260924 --output benchmarks/eagle3-phase4_4b
```

The analyzer refuses to publish an incomplete primary matrix, requires five
repeats per cell, checks cleanup/prefix-cache isolation and exact graph/eager
output equality, and keeps every trial. It emits summary JSON, per-trial CSV,
per-capture lifetime CSV, throughput, latency and cache tables.

Measurement time is monotonic host wall time. Shell log timestamps use the
server process timezone; they are not used in latency or throughput formulas.
No model download, checkpoint update or Git commit is part of this phase.
