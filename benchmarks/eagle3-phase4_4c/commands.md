# Executed GPU commands

Working directory: `/root/autodl-tmp/projects/nano-vllm`.

Environment: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1`.

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44c --stage bounded --concurrency 1 --workload short-long --count 512 --output /root/autodl-tmp/eagle3-phase4.4c-20260924/bounded-c1
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44c --stage bounded --concurrency 2 --workload long-long --count 512 --output /root/autodl-tmp/eagle3-phase4.4c-20260924/bounded-c2
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44c --stage bounded --concurrency 4 --workload long-long --count 512 --output /root/autodl-tmp/eagle3-phase4.4c-20260924/bounded-c4
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44c --stage shadow --concurrency 1 --count 512 --output /root/autodl-tmp/eagle3-phase4.4c-20260924/shadow-c1
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44c --stage shadow --concurrency 2 --count 512 --output /root/autodl-tmp/eagle3-phase4.4c-20260924/shadow-c2
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase44c --stage shadow --concurrency 4 --count 512 --output /root/autodl-tmp/eagle3-phase4.4c-20260924/shadow-c4
```

Warmup, input construction, stage order and hashes are retained in manifests and the source snapshot.
