# Phase 4.4A Commands

Working directory: `/root/autodl-tmp/projects/nano-vllm`.
Python: `/root/autodl-tmp/venvs/nano-baseline/bin/python`.
GPU commands run sequentially: TP=2 uses both devices and the frozen rendezvous
port/shared-memory name. Raw paths must be new; the driver refuses overwrite.

## Regression

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest -v tests.test_eagle3_concurrent_state tests.test_eagle3_block_transactions tests.test_eagle3_batched_runtime tests.test_eagle3_scheduler_adapter
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest discover -s tests -q
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest -v tests.test_eagle3_graph_prototype
```

Before/after logs: `/root/autodl-tmp/eagle44a-{pre,post}-{focused,cpu}.log`.

## NCCL Probe

```bash
timeout 240s /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_graph_probe --output /root/autodl-tmp/eagle3-phase4.4a-probe-20260924 > /root/autodl-tmp/eagle44a-probe.log 2>&1
```

## Final Correctness Gate

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN timeout 1800s /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44a --stage correctness --output /root/autodl-tmp/eagle3-phase4.4a-correctness-final-20260924 > /root/autodl-tmp/eagle44a-correctness-final.log 2>&1
```

## Two Fresh Timing Processes

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN timeout 1800s /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44a --stage performance --gate /root/autodl-tmp/eagle3-phase4.4a-correctness-final-20260924/manifest.json --output /root/autodl-tmp/eagle3-phase4.4a-performance-b-20260924 > /root/autodl-tmp/eagle44a-performance-b.log 2>&1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN timeout 1800s /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44a --stage performance --gate /root/autodl-tmp/eagle3-phase4.4a-correctness-final-20260924/manifest.json --output /root/autodl-tmp/eagle3-phase4.4a-performance-c-20260924 > /root/autodl-tmp/eagle44a-performance-c.log 2>&1
```

## Analysis

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44a_summary --correctness /root/autodl-tmp/eagle3-phase4.4a-correctness-final-20260924 --raw /root/autodl-tmp/eagle3-phase4.4a-performance-b-20260924 /root/autodl-tmp/eagle3-phase4.4a-performance-c-20260924 --reuse /root/autodl-tmp/eagle3-phase4.4a-metadata-reuse-20260924 --output benchmarks/eagle3-phase4_4a
```

## Supplemental Same-key Metadata Reuse

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN timeout 1200s /root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44a_reuse --stage correctness --pilot --output /root/autodl-tmp/eagle3-phase4.4a-metadata-reuse-20260924 > /root/autodl-tmp/eagle44a-metadata-reuse.log 2>&1
```

`--pilot` marks this as supplemental, not a substitute for the full gate. It
tests all three M values, reuses one graph per M for three unrelated batches,
and records unprofiled rank0 `CUDAGraph.replay()` API wall duration separately.

## Workspace Control

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -c "import torch,gc,json; torch.cuda.set_device(0); a=torch.ones((4,5120),device='cuda',dtype=torch.bfloat16); b=torch.ones((5120,5120),device='cuda',dtype=torch.bfloat16); torch.cuda.synchronize(); base=torch.cuda.memory_allocated(); rows=[]
for i in range(6):
 s=torch.cuda.Stream()
 with torch.cuda.stream(s): y=a@b
 torch.cuda.synchronize(); del y,s; gc.collect(); torch.cuda.empty_cache(); rows.append({'new_stream_iteration':i,'allocated_above_base':torch.cuda.memory_allocated()-base})
s=torch.cuda.Stream()
for i in range(6):
 with torch.cuda.stream(s): y=a@b
 torch.cuda.synchronize(); del y; gc.collect(); torch.cuda.empty_cache(); rows.append({'reused_stream_iteration':i,'allocated_above_base':torch.cuda.memory_allocated()-base})
print(json.dumps(rows,indent=2))" > /root/autodl-tmp/eagle44a-stream-workspace-control.json
```

## Retained Development Runs

`eagle3-phase4.4a-pilot-20260924`: M4, context255 generation passed; later control
mailbox race timed out in Gloo. Exit also exposed missing inference-mode scope
in the prototype buffer scrub. No numerical/NCCL-capture failure was observed.

`eagle3-phase4.4a-correctness-20260924`: 17 cases passed, but post-release
allocated memory revealed one retained cuBLAS workspace per capture stream.
This is superseded, not deleted or relabeled as the final gate.

`eagle3-phase4.4a-performance-a-20260924`: explicitly terminated during startup
after detecting that workspace growth in the preceding correctness manifest.
No measured target samples from this process are used.

`/root/autodl-tmp/eagle44a-stream-workspace-control.json`: standalone BF16 GEMM,
six distinct streams versus six uses of one stream. Distinct streams retain
8,519,680 bytes each; reused-stream allocated memory is constant. No backend
cache clearing or production edits are used to address this.

Each completed run saves the exact command, prototype source snapshot/hashes,
frozen-source hashes, revisions, software/GPU topology, key/address records,
raw timings, correctness traces and cleanup. Raw artifacts remain on the data
disk, outside Git. No Git commit is made.
