# Executed Commands

Working directory: project root. Common environment: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1. Mixing default except explicitly marked no-replay diagnostic.

## diagnostic

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage diagnostic --output /root/autodl-tmp/eagle3-phase5.1-20260925/diagnostic
```

## prototype

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage prototype --output /root/autodl-tmp/eagle3-phase5.1-20260925/prototype
```

## correctness

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage correctness --output /root/autodl-tmp/eagle3-phase5.1-20260925/correctness
```

## lifecycle-default

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage lifecycle --samples 5 --output /root/autodl-tmp/eagle3-phase5.1-20260925/lifecycle-default
```

## lifecycle-mixing-off-no-replay

```bash
NCCL_GRAPH_MIXING_SUPPORT=0 /root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage lifecycle --samples 5 --output /root/autodl-tmp/eagle3-phase5.1-20260925/lifecycle-mixing-off-no-replay
```

## serving-c1-literal

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage serving --serving-mode literal --concurrency 1 --output /root/autodl-tmp/eagle3-phase5.1-20260925/serving-c1-literal
```

## serving-c1-optimized

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage serving --serving-mode optimized --concurrency 1 --output /root/autodl-tmp/eagle3-phase5.1-20260925/serving-c1-optimized
```

## serving-c2-optimized

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage serving --serving-mode optimized --concurrency 2 --output /root/autodl-tmp/eagle3-phase5.1-20260925/serving-c2-optimized
```

## serving-c2-literal

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage serving --serving-mode literal --concurrency 2 --output /root/autodl-tmp/eagle3-phase5.1-20260925/serving-c2-literal
```

## serving-c4-literal

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage serving --serving-mode literal --concurrency 4 --output /root/autodl-tmp/eagle3-phase5.1-20260925/serving-c4-literal
```

## serving-c4-optimized

```bash
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase51.py --stage serving --serving-mode optimized --concurrency 4 --output /root/autodl-tmp/eagle3-phase5.1-20260925/serving-c4-optimized
```
