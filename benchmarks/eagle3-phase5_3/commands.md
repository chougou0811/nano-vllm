# Executed Commands

Working directory: `/root/autodl-tmp/projects/nano-vllm`.
Raw root: `/root/autodl-tmp/eagle3-phase5.3-20260925`. Never overwrite a run directory.

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53.py capture
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage correctness --name correctness
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage correctness --name correctness-r2
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53.py isolated
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage serving --name serving-c1-tail --record-draft-variation --reverse --families long-long copy-pattern --concurrencies 1
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage serving --name serving-c1-tail-r2 --record-draft-variation --reverse --families long-long copy-pattern --concurrencies 1
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage serving --name serving-fresh --record-draft-variation --reverse --families heldout-mixed routing-table --concurrencies 1 2 4
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage serving --name serving-fresh-r2 --record-draft-variation --reverse --families heldout-mixed routing-table --concurrencies 1 2 4
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage serving --name serving-fresh-r3 --record-draft-variation --reverse --families heldout-mixed routing-table --concurrencies 1 2 4
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage serving --name serving-main
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage serving --name serving-main-r2 --record-draft-variation
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python /root/autodl-tmp/projects/nano-vllm/benchmarks/serving/eagle3_phase53_serving.py --stage serving --name serving-main-r3 --record-draft-variation
```

## Additional Diagnostics and Reporting

GPU commands used HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1. Full source/environment hashes are in manifests.

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_numerics
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_edges
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_disagreement
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_numerics --disagreement
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_disagreement --name serving-audit --manifest /root/autodl-tmp/eagle3-phase5.3-20260925/serving-main-r3/manifest.json
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_numerics --disagreement --input-name serving-audit --output-name serving-audit-numerics
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_analysis
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_tables
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m benchmarks.serving.eagle3_phase53_finalize
```

```text
/root/autodl-tmp/venvs/nano-baseline/bin/python -m unittest discover -s tests -q
```
