# Project Context

This project extends upstream nano-vllm for online LLM inference. Preserve the
upstream copyright and license. The target is Qwen3-14B on two RTX 4090 24GB GPUs
with tensor parallelism (TP) = 2.

## Current Stage: Serving Benchmark Measurement

Qwen3-0.6B TP correctness and Qwen3-14B TP=2 functionality baselines are complete
within their recorded coverage. The user has authorized an independent serving
benchmark framework for deterministic arrivals, instrumentation, latency and SLO
measurement, raw artifacts, and reproducible reports. Preserve production
scheduler, batching, sampling, and model execution semantics. Validate with unit
tests and a small Qwen3-14B TP=2 BF16 eager run; do not draw performance conclusions
from that framework validation. Do not implement EAGLE-3, SLO-aware scheduling,
adaptive speculation, Poisson traffic, or performance optimizations yet.

Trace the existing call graph, tensor shapes, and data flow before changing code.
Cover QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
VocabParallelEmbedding, ParallelLMHead, distributed initialization, NCCL,
rank/world_size, weight loading, and KV cache partitioning.

## Environments and Responsibilities

The following environment details were supplied by the user. Verify the actual
runtime before experiments; these are not measurements from this checkout.

- Local: RTX 5070 Ti 16GB, for learning, source reading, design, small changes,
  unit tests, and small-model validation. Do not require Qwen3-14B or TP=2 here.
  Prioritize explanations of call paths, Python/PyTorch syntax, tensor shapes,
  and design choices unless implementation is explicitly requested.
- Server: two RTX 4090 24GB GPUs, PyTorch 2.8.0+cu128, CUDA 12.8. The user reports
  successful NCCL communication, NODE topology, no NVLink, and PCIe communication
  on the same NUMA node. Use this environment for Qwen3-14B TP=2, EAGLE-3,
  multi-GPU correctness, and performance experiments.
- On the server, explain the design and risks before editing TP, NCCL, KV cache,
  CUDA Graph, or model execution paths.
- Changes affecting multi-GPU behavior, memory, performance, EAGLE-3, or TP/NCCL
  must ultimately be validated on the server.

## Collaboration and Change Rules

- GitHub is the sole synchronization intermediary between the two checkouts:
  local changes -> commit/push -> server pull -> validation/fixes -> commit/push
  -> local pull.
- The user reviews git diff before deciding whether to commit. Do not commit or
  push automatically; do so only when requested.
- Keep changes small and scoped. Do not perform automatic large refactors or
  implement the whole roadmap at once.
- Derive model dimensions, attention head counts, and KV head counts from config.
  Do not hardcode model sizes, cuda:0, or world_size=1. New code should support
  both TP=1 and TP=2 where applicable.
- Report changed files, reasons, correctness validation, and measured performance
  impact. If performance was not measured, say so. Never claim speedups without
  a benchmark.
- Preserve reproducibility: save the code revision and local diff, environment
  and dependency versions, GPU/topology details, commands, model/tokenizer
  revisions, parameters, workload and seeds, warmup/timing methodology, and raw
  results for each benchmark. Do not put model weights or secrets in Git.

## Future Roadmap (Not Authorized for Implementation Yet)

1. SLO-aware scheduling: instrument TTFT, TPOT/ITL, decode/prefill backlog,
   request waiting time, and step latency. Adjust prefill token budget and decode
   priority using SLO pressure; compare against fixed chunked prefill.
2. EAGLE-3: use a dedicated Qwen3-14B EAGLE-3 speculator, not a generic small
   Qwen3 draft model. Adapt multi-token proposals, target parallel verification,
   accepted-prefix computation, multi-token Sequence advancement, and TP=2 target
   execution. Verify speculator availability and compatibility before adoption.
3. Adaptive speculation: adjust speculative length K using acceptance rate,
   accepted length, proposal/verification latency, batch size, and SLO slack.
4. Speculative KV state: design reserve/commit/rollback or equivalent handling
   for paged KV, partial blocks, block boundaries, slot mapping, preemption,
   and prefix caching.
5. CUDA Graph and Triton/CUDA optimization: select a measured hotspot such as
   verification, accepted-prefix reduction, or KV commit metadata; implement
   a focused optimization with an independent benchmark.

Eventual evaluation should cover TTFT, TPOT/ITL, P50/P95/P99 latencies, token and
request throughput, SLO goodput and violation rate, acceptance rate, accepted
tokens per step, proposal/verification latency, GPU memory/utilization, and TP
AllReduce/communication overhead. These metrics are roadmap requirements, not a
claim that current nano-vllm already exposes them.
