# Frontier Candidate Pool

As of 2026-09-24. External facts resolve through `source-ledger.json`; fit,
priority, risk and learning value are **our inferences**, not author claims.
Compatibility below means a credible design path, not a newly tested backend.
Every candidate preserves historical results. The final selection is only C1
and conditional C2; remaining entries explain exclusions, not parallel plans.

## C1. Context-independent compute-region replay [PRIMARY]

- Technology/status/source: 2026 active segmented/breakable graph execution;
  SGLang S1/S2 and vLLM V3. CUDA Graph itself is mature, modular execution is
  the relevant frontier transfer.
- Problem/local fit: repeated target dispatch and rank arrival gaps (P43/P44A),
  without whole-target exact-context cache churn (P44B/P44C).
- Reuse/changes/minimal transfer: keep verification/KV/TP/draft/acceptance;
  one post-attention region per layer, optional Qwen hook, bounded registry.
- Training/checkpoint: neither. Hardware: existing CUDA/NCCL; RTX4090 plausible
  and full fixed-shape replay already demonstrated. TP2 requires identical
  collective capture/order; Qwen3-14B boundary audited, no new model backend.
- Difficulty/risk: medium-high; static aliases, residual semantics, 120 graphs
  per rank, capture memory and boundary overhead. No padding required.
- Benchmark/MVP: exact-stage then five-repeat serving; one or two phases;
  compare to literal eager, charge startup. Evidence: strongest local mechanism
  evidence, no serving gain established for this proposed region.
- Learning/interview: graph lifecycle, distributed replay, why smaller capture
  contracts may outperform a broad dynamic cache; falsifiable design, not hype.

## C2. Incremental GPU-native input metadata [SECONDARY, gated]

- Technology/status/source: MRV2 stable request rows, staged updates and gather,
  active 2026 (V1/V2). Addresses CPU bookkeeping, not model arithmetic.
- Fit/reuse/change: local lists/tensors rebuilt every step (N2/N4), but isolated
  cost unknown. Replace only verification input preparation; reuse owner IDs,
  pages, generation, RPC, model and CPU acceptance.
- Training/checkpoint: neither. Hardware: pinned copies/device indexing; Ada,
  TP2 and Qwen dimensions naturally supported in design, not yet validated.
- Difficulty/risk: medium; stale slot/page ownership, buffer lifetime and
  host-copy synchronization. No async scheduler or acceptance rewrite.
- Benchmark/MVP: exact integer oracle plus H2D/CPU/whole-serving comparison;
  one stage and one phase. Stop if attribution shows <5% serving opportunity.
  Evidence: strong external implementation, weak local cost attribution.
- Learning/interview: independent persistent state and packed batch order,
  reducing per-step work without weakening ownership.

## C3. Whole async runtime / overlap scheduler

- Technology/status/source: TRT overlap loop, MRV2 async-first (T1/V1), active
  2026 integration of an older latency-hiding principle.
- Fit/reuse/change: host gaps relevant, but per-layer launch gaps are not all
  schedulable host work. Double-buffered inputs and one-step-ahead result
  handling touch coordinator, lifetime, cancellation, pending tokens and RPC.
- Training/checkpoint: neither. Ada/TP2/Qwen possible in principle; no NVLink
  requirement for the idea. Frozen single-step ownership would need extension.
- Difficulty/risk: high; dependent speculative drafts and post-acceptance page
  rollback prevent blindly scheduling another target step.
- Benchmark/MVP: copy overlap alone is bounded but duplicates C2; whole runner
  not a small migration. Evidence external strong, local overlap slack unknown.
- Learning/interview: events/ownership excellent; inferior scope-to-evidence
  ratio now. Defer, do not call a whole rewrite the backup.

## C4. Parallel DFlash / semi-autoregressive DSpark drafter

- Technology/status/source: DFlash Feb/May 2026, DSpark July 2026; published
  DeepSpec Qwen3-14B weights (D1-D4, HF1/HF2, PDS/PDF). Genuine new architecture,
  not an inference-only replacement for the EAGLE autoregressive layer.
- Fit: serial draft fraction grows to 24.32% at c4, but target still 61.10%.
  Keep transactional target verification; replace draft model/state adapter,
  feature taps, masks and checkpoint loader. DSpark confidence policy would
  also reopen a frozen K-selection question and is not part of a fixed-K port.
- Training/checkpoint: new checkpoint yes; mandatory training no with public
  weights. Domain-specific retraining would be a separate expensive project.
- Hardware/compatibility: Qwen3-14B named by public weights, five-layer draft,
  hidden5120/vocab151936, taps[1,10,19,28,37] after layers, block7. Existing EAGLE
  pre-layer taps/reduced vocab are different. Transformers5.10.2 config is not
  a qualification of our environment. Exact target training revision unknown.
  About 6.22/6.36 GiB weight payload must replace, not simply add to, the old
  draft footprint; no proof it fits rank0 at useful KV capacity. TP2 target plus
  rank0 draft needs new communications/features audit, not new target shards.
- Difficulty/risk: high, numerical/state and architecture mask semantics;
  reproducible before/after possible, acceptance and memory must be measured.
  MVP possible only after independent memory/source compatibility gate.
- Learning/interview: excellent new-model/runtime co-design, but less direct
  bottleneck coverage and higher risk than C1. Evidence: real reference/weights,
  zero local performance evidence. No weights downloaded in Phase5.0.

## C5. Exact-key selective capture / larger residency

- Technology/status/source: active graph admission/cache engineering; our
  P44B/P44C provide stronger evidence than borrowing a framework name.
- Fit/reuse/change: long horizons do repeat keys, but cap4 replay is 1-3% and
  enabled eligible misses regress. Reuse old cache, replace policy/residency.
- Training/checkpoint: neither; Ada/TP2/Qwen already correctness-tested for
  old exact policy. Larger pools consume scarce rank0 memory.
- Difficulty/risk: medium, key agreement and eviction lifecycle. Easy paired
  benchmark, but new policy may hide rather than explain miss regression.
- MVP/learning/value/evidence: feasible cache-only MVP with strong negative
  history, moderate incremental learning. **Not selected** until operator-level
  miss cost is explained; no promise that frequency alone produces net benefit.

## C6. Whole-target context bucketing / padded graph

- Technology/status/source: framework shape planning and graph modes S2/V3,
  active but padding is not intrinsically a new 2026 algorithm.
- Fit/reuse/change: reduces max_k fragmentation; replaces verification shape
  preparation and attention/graph contract, retaining outer transactions.
- Training/checkpoint: neither. Ada/TP2 possible, Qwen causal/RoPE/page semantics
  need qualification. Dummy rows and page safety are substantial correctness work.
- Difficulty/risk: high; masking, invalid KV addresses, finite checks, target
  feature selection and altered BF16 paths. Measurable with padded controls.
- MVP/value/evidence: safety audit feasible, full serving benefit unknown;
  strong systems learning. C1 removes context dependence only from operators
  that never consume it, so has the cleaner initial safety boundary.

## C7. FlashInfer paged-attention backend / reusable plan

- Technology/status/source: F1 and 2025 paper PF; active 2026 API. Plan once,
  reuse metadata across layers and select a compatible attention backend.
- Fit/reuse/change: page adapter around existing target attention, preserve
  storage/state/TP; replace one backend. Attention sampled only 0.53-1.10ms.
- Training/checkpoint: neither. Candidate FA2-style backend on SM89 must be
  qualified with CUDA12.8/Torch2.8; not all FlashInfer kernels support all GPUs.
  TP2 local heads and Qwen head_dim128 are plausible, not tested here.
- Difficulty/risk: medium-high; paged layout, causal alignment and numerical
  disagreement diagnostics. Great isolated backend A/B and long-context tests.
- MVP/value/evidence: feasible, useful adapter engineering, poor direct
  opportunity at measured short contexts. Do not confuse FA4 Blackwell results
  with this Ada-compatible candidate or use it to claim broad speedup.

## C8. MLP/GEMM backend selection and narrow fusion

- Technology/status/source: execution-aware autotuning F2, compiler/fusion MP1;
  2026 emphasis on actual eager/replay mode, not stale microbench choices.
- Fit/reuse/change: MLP ~69% sampled non-NCCL GPU service (P43). Replace one
  gate/up/activation backend, keep target/state architecture and row all_reduce.
- Training/checkpoint: neither for BF16; quantization is a different candidate.
  Ada support depends on selected kernel, TP2 shard sizes and Qwen numerics need
  controls; Blackwell-only kernels fail hardware gate.
- Difficulty/risk: high without counters; skinny GEMV/GEMM already optimized by
  libraries. Benchmarkability high, expected gain unknown.
- MVP/value/evidence: operator-only feasible but may optimize the wrong wall
  fraction. Good kernel learning; first establish post-dispatch critical path.

## C9. Persistent megakernel / GPU-native entire serving stack

- Technology/status/source: MPK v2 June2026 and Blink Apr2026 (MP1/PMP/PBL).
  More radical removal of CPU control; real frontier beyond CUDA Graph.
- Fit/reuse/change: targets launches directly, but replaces model lowering,
  communication execution and possibly scheduling/KV. Little frozen runner reuse.
- Training/checkpoint: normally neither if architecture mapped. Blink needs
  SmartNIC/RDMA not present. MPK source has SM90/100 specialization and native
  fallback; that is **not proof Ada is impossible**, nor proof two4090 TP works.
- Difficulty/risk: very high, kernel/runtime/communication correctness. Whole
  Qwen3-14B TP2 serving not a bounded one/two-phase MVP here. Single-op prototype
  loses the paper's whole-runtime objective.
- Benchmark/value/evidence: high research value, mismatched cost/hardware;
  published results not a prediction for this machine. Reject current scope.

## C10. TP overlap / communication fusion; EP transport

- Technology/status/source: T1, MP1, DeepEP DEP; 2025-26 active particularly MoE.
- Fit/reuse/change: 84 collective calls/verify observed, but peer waiting and
  launch drift inflate residency. Dense TP overlap would alter projections or
  collective scheduling; EP dispatch/combine is not a dense all_reduce drop-in.
- Training/checkpoint: no for dense TP; adopting EP requires a different MoE
  model, outside fixed Qwen3-14B. Ada SYS lacks NVLink, backend support varies.
- Difficulty/risk: high, order/deadlocks and unproven transport bottleneck.
  Isolated matched collectives are benchmarkable; safe minimal transport port
  unavailable from DeepEP for this dense model.
- MVP/value/evidence: useful distributed reasoning, weak case for immediate
  implementation. First separate useful transfer from rank-arrival waiting.

## C11. Tiered/distributed KV and prefix reuse

- Technology/status/source: LMCache multiprocess, Mooncake transfer/store,
  active2026 (L1/L2/M1/M2). Independent cache lifetime and scalable reuse.
- Fit/reuse/change: add connector below cache storage; speculative prefix
  identity/ownership needs new safety design. Current16/88 peak pages at c4
  and isolated prefixes show no capacity/reuse bottleneck in tested regime.
- Training/checkpoint: neither. Ada plus hostRAM/SSD local tier feasible, TP2
  needs coordinated shards; remote benefits require transport and shared users.
- Difficulty/risk: high, stale prefix/model identity, async eviction and privacy.
  Benchmark requires declared reusable-prefix/capacity workloads, not rewriting
  existing workload to present an artificial win.
- MVP/value/evidence: local-tier MVP possible; teaches ownership/storage, but
  solves an unmeasured problem. Not selected as target-latency optimization.

## C12. P/D disaggregation and distributed routing

- Technology/status/source: Mooncake, NIXL/Dynamo, active2026 (M1/M2/NIXL).
- Fit/reuse/change: target engine could be a worker, but new service protocols,
  KV transfer, remote ownership, router and recovery dominate change scope.
- Training/checkpoint: neither. A14B BF16 target occupies both24GB GPUs;
  independent prefill/decode replicas do not fit one GPU each. A toy small model
  demo changes the deployment question rather than accelerating this target.
- Difficulty/risk: very high; request migration and transport failure. Requires
  extra capacity/network to benchmark realistic separation.
- MVP/value/evidence: excellent distributed-service learning, not a reasonable
  minimal transfer for fixed hardware. Current engine is not cluster-ready.

## C13. GPU acceptance / GPU-side sampling

- Technology/status/source: V4/V1, active2026 device-native output processing.
- Fit/reuse/change: eliminate greedy argmax CPU transfers or accepted-prefix
  calculation; retain target/draft initially but change result ownership and
  rank-status boundary. Current accept/commit aggregate only0.81-1.03% wall.
- Training/checkpoint: neither. Ada/TP2/Qwen compatible in principle; greedy
  tie handling and frozen CPU result/rollback semantics remain mandatory.
- Difficulty/risk: medium-high; removing a sync without a consumer redesign
  merely moves it. Stochastic sampler migration would exceed current scope.
- MVP/value/evidence: small kernel possible, good device-control lesson,
  negligible proven wall opportunity. Do not remove safety checks for speed.

## C14. Quantization / heterogeneous offload

- Technology/status/source: FlashInfer low-precision kernels, ATSInfer2026,
  SuperInfer2026 (F2/PATS/PSUP). Improves capacity or compute efficiency.
- Fit/reuse/change: current BF16 target fits and low page occupancy is measured;
  changes weights/arithmetic, offload policy or placement, not just target launch.
- Training/checkpoint: conversion/calibration or pretrained quantized model
  usually needed for lowprecision; no mandatory training for exact offload.
  Ada supports some formats, not Blackwell NVFP4 acceleration; GH200 C2C cannot
  be assumed on SYS PCIe. TP2 quantized kernels need separate validation.
- Difficulty/risk: high, new numerical/latency contract and memory transfers.
  Benchmarkable only with separate quality/capacity baselines.
- MVP/value/evidence: valuable in another capacity-limited deployment; not
  chosen under fixed BF16 project objective. ATSInfer is genuinely new and
  consumer-focused, but hardware match alone is insufficient bottleneck match.
