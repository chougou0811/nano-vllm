# Phase 4.3 Commands

Working directory: `/root/autodl-tmp/projects/nano-vllm`.
Python: `/root/autodl-tmp/venvs/nano-baseline/bin/python`.
All runs are diagnostic-only. Run GPU processes sequentially (both use TP=2
and the original shared-memory name / rendezvous port).

## CPU Regression, Before and After

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest -v tests.test_eagle3_concurrent_state tests.test_eagle3_block_transactions tests.test_eagle3_batched_runtime tests.test_eagle3_scheduler_adapter
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest discover -s tests -q
```

Logs: `/root/autodl-tmp/eagle43-{pre,post}-{focused,cpu}.log`.

## Primary Ordinary Process

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase43 --system ordinary --raw /root/autodl-tmp/eagle3-phase4.3-final-20260923 --modes warmup control timing profile minimal collective-events > /root/autodl-tmp/eagle43-ordinary-final.log 2>&1
```

## Primary Speculative Process

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,ENV /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase43 --system speculative --raw /root/autodl-tmp/eagle3-phase4.3-spec-final-20260923 --modes warmup control timing profile minimal collective-events > /root/autodl-tmp/eagle43-speculative-clean.log 2>&1
```

Primary modes capture two target steps at each c=1/2/4 and workload
mixed-output/long-short, using default CLI values. GPU computation is delegated
to unchanged production methods. The runner refuses an existing manifest:
use new raw/log paths when reproducing; do not overwrite the recorded runs.

## Offline Analysis

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase43_summary --raw /root/autodl-tmp/eagle3-phase4.3-final-20260923 --spec-raw /root/autodl-tmp/eagle3-phase4.3-spec-final-20260923 --output benchmarks/eagle3-phase4_3
```

## Retained Supplemental Commands

These were run before the final harness/observer checks. They remain separate
from primary attribution, with no sample removal within a run.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase43 --system speculative --raw /root/autodl-tmp/eagle3-phase4.3-pilot --concurrency 1 --workloads mixed-output > /root/autodl-tmp/eagle43-pilot.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase43 --system speculative --raw /root/autodl-tmp/eagle3-phase4.3-raw-20260923 > /root/autodl-tmp/eagle43-speculative.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING,ENV /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase43 --system speculative --raw /root/autodl-tmp/eagle3-phase4.3-final-20260923 --modes warmup control timing profile minimal collective-events > /root/autodl-tmp/eagle43-speculative-final.log 2>&1
```

The second command used the old single-prefill harness assertion and stopped
at c4 long-short. The present harness fixes that diagnostic-only assumption.
The third command enables per-collective TUNING logs and is retained as a
verbose observer-sensitivity run. Its separate full analysis is at
`/root/autodl-tmp/eagle3-phase4.3-verbose-analysis/summary.json`.

## Saved Sources and Tools

Primary `*-manifest.json` contains versions, topology, GPU information,
config, revisions, command arguments, source hashes, trial inputs/outputs and
cleanup. `*-diagnostic-source.py` is the actual runner snapshot at process
start. `source-snapshot.tar.gz` additionally preserves the working source tree
without weights, raw tensors or caches. Chrome traces include CUPTI/CUDA
runtime/driver metadata and may be opened with a compatible trace viewer.
`nsys` and `ncu` were not installed; no Nsight Compute run is claimed.
