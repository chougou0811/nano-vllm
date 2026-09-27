# 2026 Frontier Map (external scan first)

Access cutoff: 2026-09-24 UTC. The initial map was written before reopening the local
inference code or historical profiling in Phase 5.0; later source-deepening is
explicitly identified below. It is a bounded public-source
scan, not a claim to enumerate every 2026 paper. No direction is selected here.
Pinned source files and evidence types are resolved in `source-ledger.json`.

## Directions

`Training/checkpoint` concerns migration to an existing dense target, not whether
the original research trained a model. `2 GPU` means a plausible small experiment,
not demonstrated compatibility with this project's exact software stack.

| Direction | Core problem / representative evidence | 2025-26 activity and bottleneck | Hardware | Training/checkpoint | Integration / 2 GPU fit |
|---|---|---|---|---|---|
| Speculative decoding | EAGLE-3 plus vLLM/SGLang speculative runtimes: amortize target execution over accepted outputs | Active, increasingly integrated with async runtime; acceptance and verification cost both matter | Target plus draft must fit | Dedicated trained speculator, matching target features | High state complexity; yes with existing checkpoint |
| Parallel/block drafting | [DFlash](https://arxiv.org/abs/2602.06036), February 2026: diffusion draft produces a block in one forward | New 2026 approach to sequential draft dependency; author speedups are not portable predictions | Model-specific draft plus target | New trained draft required; reuse published checkpoint only when target matches | New draft state/mask/features; possible, checkpoint-gated |
| GPU-native serving runtime | [vLLM MRV2](https://github.com/vllm-project/vllm/blob/main/docs/design/model_runner_v2.md): stable request rows, GPU input preparation and sampling | 2026 active implementation; removes CPU bookkeeping and repeated transfers | Ordinary supported GPU, not inherently NVLink-dependent | Neither | Medium/high for a small stage, very high for whole runner; plausible |
| Fully GPU-controlled serving | [Blink](https://arxiv.org/abs/2604.07609) and [MPK](https://arxiv.org/abs/2512.22219) | Research: eliminate host control and per-operator launch paths, not merely overlap them | Blink SmartNIC/RDMA; MPK specialized GPU runtime and multi-GPU transport | Usually neither, supported model mapping needed | Very high; do not equate reference hardware with consumer-GPU support |
| Async scheduling/execution | [TensorRT-LLM overlap](https://nvidia.github.io/TensorRT-LLM/latest/torch/features/overlap_scheduler.html), SGLang overlap, MRV2 | Older principle, expanded 2025-26 feature combinations; hides host work behind GPU work | Streams/events, host buffers | Neither | Medium/high, speculative result dependencies and cancellation matter; possible |
| Dynamic/piecewise CUDA Graph | [vLLM graph modes](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md), [SGLang PCG](https://github.com/sgl-project/sglang/blob/ea5baf4022e42ef13b089430ce2b1927a5c9d6f0/docs/docs/advanced_features/piecewise_cuda_graph.mdx) | Active; separate capture-compatible regions from dynamic operations, support multiple token shapes | CUDA GPU, address-stable buffers | Neither | Medium for one region, high for whole runner; plausible. Dynamic does not mean arbitrary shapes are safe |
| Attention runtime | [FlashInfer](https://arxiv.org/abs/2501.01005): plan/run, workload-aware kernel specialization | Active 2025 paper and 2026 APIs; orchestration and backend selection as well as math | Backend-specific, CUDA/SM requirements vary | Neither for exact attention | Medium adapter, independent numerical qualification; plausible with a supported backend |
| Paged attention | FlashInfer paged wrappers; production vLLM/SGLang backends | Mature foundation, still evolving for ragged, sparse, hybrid and speculative shapes | GPU pages and layout-compatible kernels | Neither | Medium; exact page layout and causal semantics are critical |
| KV allocation/state | vLLM KV manager, SGLang memory pools, TensorRT-LLM KV system | Mature paging expands into hybrid-model and speculative lifecycle handling | GPU/host memory | Neither for exact storage | Medium/high; transactional lifecycle must survive replacement |
| Distributed KV | [Mooncake](https://github.com/kvcache-ai/Mooncake), [LMCache](https://github.com/LMCache/LMCache) | Active: share/offload cache independently of worker lifetime | Host/SSD usable locally; RDMA/datacenter fabric for intended scale | Neither | High for distributed service; local tier possible, benefit workload-dependent |
| Prefix reuse | SGLang Radix/HiCache, LMCache layer | Active for long-lived agent/RAG prefixes; saves prefill recomputation | GPU/CPU/SSD tiers | Neither for exact cache identity | Medium; needs repeated prefixes and invalidation/security ownership |
| Scheduling | [SuperInfer](https://arxiv.org/abs/2601.20309) | 2026 memory/scheduling co-design; SLO under KV pressure rather than one universal priority heuristic | GH200/NVLink-C2C for evaluated rotation engine | Neither | High; its transfer economics do not carry over to PCIe-only GPUs |
| Prefill/decode disaggregation | Mooncake and [Dynamo/NIXL](https://github.com/ai-dynamo/nixl/blob/main/docs/doxygen/nixl_doxygen.md) | Active serving decomposition; isolate resource demands and scale stages separately | Separate engine capacity plus KV transport, often RDMA | Neither | High; two cards already occupied by one TP target leaves little independent capacity |
| TP/EP/communication overlap | [DeepEP](https://github.com/deepseek-ai/DeepEP), SGLang overlap and TRT-LLM parallel runtime | Active large-MoE scaling; overlap dispatch/combine with compute | Usually high-bandwidth GPU fabric / RDMA; backend-specific | EP needs MoE architecture, not a dense-model drop-in | High; separate dense TP launch skew from MoE all-to-all bottlenecks |
| Kernel fusion / megakernels | MPK; [FlashAttention-4](https://arxiv.org/abs/2603.05451) | 2026 asynchronous MMA/softmax pipelines and persistent task execution | FA4 paper targets Blackwell; MPK specializations must be checked | No new checkpoint for exact BF16 operators | Medium for one operator, very high for end-to-end megakernel |
| Metadata preparation | MRV2 stable request state + gather; FlashInfer plan/run | More prominent as small-batch control cost becomes visible | GPU + pinned host buffers; no special interconnect intrinsically | Neither | Small/medium stage transfer; distinguish launch gaps from input-prep cost |
| GPU-side sampling/acceptance | MRV2 Triton sampler and speculative result indirection | Active: avoid unnecessary softmax/materialization/CPU round trips | GPU reductions/RNG | Neither, but RNG/numerical semantics change risk | Medium; greedy reduction easier than stochastic distribution equivalence |
| Distributed serving control | Dynamo routing/NIXL and Mooncake store | Active: prefix-aware routing, worker lifetime, tier transport and service reliability | Multiple workers/network, production orchestration | Neither | Very high for real service; a single TP group is not distributed serving |
| Memory hierarchy | LMCache multiprocess architecture, Mooncake store, SGLang HiCache | 2026 separates cache ownership from engine process, tiers HBM/DRAM/SSD | CPU RAM/SSD; faster fabric improves trade-off | Neither for lossless transfer | Medium local, high distributed; performance requires reuse/capacity pressure |
| Heterogeneous serving | [ATSInfer](https://arxiv.org/abs/2607.10183), SGLang multiple backends | 2026 tensor-granularity CPU/GPU placement; capacity rather than peak dense-GPU latency | Host RAM bandwidth/PCIe, or different accelerator backends | Quantization optional and separate | High; useful when weights do not fit, not automatically faster when they do |
| Kernel selection/autotuning | [FlashInfer Autotuner v2](https://flashinfer.ai/2026/09/22/autotuner-v2.html) | September 2026: execution-mode-aware selection, persistent environment keys, cross-rank cache handling | Each candidate backend has its own SM/toolchain support | Neither for exact candidates | Medium; offers methodology without importing a whole framework |
| Quantization / sparse/hybrid models | FlashInfer low-precision kernels, FA4, framework model support | Rapid 2025-26 progress, especially Blackwell/MoE/hybrid attention | NVFP4 features are not an Ada BF16 capability | Often conversion/calibration or a different pretrained architecture | High correctness/model-contract impact; keep separate from runtime-only transfer |

## What is actually new, and what is not

- New papers in this scan: DFlash (2026-02), FA4 (2026-03), Blink (2026-04),
  ATSInfer (2026-07). These dates describe public papers, not deployment maturity.
- Major 2026 engineering evolution: MRV2's async-first state layout; newer
  piecewise/breakable graph support; LMCache multiprocess separation;
  FlashInfer execution-mode-aware tuning. A March launch blog is not evidence
  that all limitations described then still hold in September.
- Paging, prefix caching, speculative verification, CUDA Graph and overlap are
  not 2026 inventions. Their composition, safety contracts and low-overhead
  implementations are the active frontier. A technique can be mature yet be
  the best minimal transfer; novelty alone will not decide the recommendation.

## Research maturity and claim discipline

Merged files establish that a path exists, not that every GPU/model works.
RFCs establish intent, not implementation. Papers and official benchmark posts
establish author-reported results on their hardware, not expected project gains.
No external speedup is used as a project performance forecast. Full-stack
alternatives remain in this map even when later hardware/risk screening rejects
them. The candidate decision is deferred until the local audit.

## Later source-deepening supplement

The initial independent map was saved at 13:29:45 UTC. The following discoveries
were added while checking exact source paths and checkpoint compatibility;
they are not retroactively represented as an initial recommendation.

- [DeepSpec](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2)
  publishes **Qwen3-14B** DFlash and DSpark block-7 checkpoints. This corrects
  the incomplete impression from the original DFlash checkpoint list alone.
  [DSpark](https://arxiv.org/abs/2607.05147v1), July 2026, combines a parallel
  backbone, lightweight sequential dependency modeling and confidence-based
  verification selection. Architecture availability is not runtime qualification.
- Metadata-only inspection finds five draft layers, five target feature taps,
  full target vocabulary, and about 6.22/6.36 GiB BF16 weight payload for the
  DFlash/DSpark variants respectively. No weights were downloaded. Configs name
  Transformers 5.10.2 and do not pin the exact target training revision. Public
  weights avoid mandatory retraining; reference recipes assume eight GPUs and
  warn about non-thinking training-domain transfer. See D2-D4, HF1-HF2.
- SGLang also has a **compiler-free breakable graph** implementation (S1), not
  only a torch.compile piecewise path. This creates a smaller potential
  execution-stage transfer than importing a framework compiler stack. Its
  stream, tensor-lifetime and model-boundary contracts still require auditing.
- Document path drift matters: the initial web-indexed SGLang PCG page used
  `docs_new/docs/...`; the pinned September tree uses `docs/docs/...`.
  The ledger resolves the immutable source, not the mutable search URL.
