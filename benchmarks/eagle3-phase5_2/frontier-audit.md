# Frontier Re-selection, 2026-09-25

This is a bounded source audit, not an exhaustive survey or an external-framework
benchmark. Selection follows the new Phase5.2 measurements. No external package,
checkpoint or new inference feature was installed/executed.

## Updated Source Pool

All nine Phase5.0 repositories were checked with `git ls-remote ... HEAD`.
Selected files were downloaded at the resulting immutable commit, without
altering the old snapshots.33 files have URL/SHA256 entries in
[source-ledger.json](source-ledger.json); three additional FlashInfer implementation
files are listed below. A remote default HEAD is not a release tag or a claim
that every branch advanced; Mirage's earlier snapshot used the mpk branch.

| project | current audit commit prefix | role in reassessment |
|---|---|---|
| SGLang | 3e7e6529002d | batched EAGLE steps, conditioning split, graph boundary, P2P guards |
| vLLM | 29468dde8b51 | stable rows, GPU metadata, graph splitting, custom collective prerequisites |
| TensorRT-LLM | c76f4a856447 | overlap execution and input-buffer ownership |
| FlashInfer | dc5b19bb7408 | attention plan/run, mode-aware tuning, BF16 backend capability gates |
| DeepSpec | 005e03b81cec | public Qwen3-14B DFlash/DSpark configurations/checkpoints |
| DFlash | 07ebd93db9f4 | parallel drafting and newly reviewed DFlash2 model coverage |
| Mirage | 97de25929c09 | persistent GPU execution scope/dependencies |
| Mooncake | 16b7ba3c7364 | KV transfer, deployment-scale mismatch |
| LMCache | 57afd4c0b9bf | cache services, capacity/reuse pressure mismatch |

## Candidate Assessment

| direction | new measured relevance | transfer/hardware risk | decision |
|---|---|---|---|
| Cross-request EAGLE draft-step batching | draft6.1/11.2/21.4ms; generation-only share9.2/13.3/16.8% | transient masks/packing and row ownership; same checkpoint, no training, rank0 only | **Primary**, generation first, serial catch-up retained |
| Mode-aware BF16 MLP tactic selection | MLP about70% of non-NCCL GPU kernel service | Ada-compatible cuBLASLt source exists; actual gain/build/numerics unqualified | **Secondary**, isolated linear gate before integration |
| More graph coverage/context buckets | verify still largest, fallback rises at c4 | previous capture lifecycle and startup costs; larger coverage is not proven profitable | not selected; do not resume graph by default |
| MRV2 stable rows/GPU metadata | input descriptor/layout/copy prep about0.15-0.23ms | ownership/mirror complexity for <1% diagnostic target time | not selected at c<=4 |
| Attention plan/run or new backend | attention only0.53-1.10ms in measured target windows | paged masking/numerical contract plus backend qualification | not selected for these context lengths |
| Custom/fused TP collectives |84 collectives remain; rank-asymmetric waiting observed | current device peer-access queries false; residency is not pure wire cost | not selected; no P2P bypass or claimed transport bottleneck |
| Async-first host scheduling | scheduler about0.01-0.02% wall; dispatch about0.04-0.05ms | speculative future state/ownership extends frozen semantics | not selected as a small fix; target dispatch differs from scheduling |
| Whole GPU-native/persistent runtime | launch/rank coordination remains material | compiler/model/KV/communication migration; Ada TP2 path not qualified | too broad for minimum transfer |
| DFlash / DSpark | can reduce draft rounds and change acceptance | new draft/checkpoint/features/memory qualification, not execution-only change | viable research, lower immediate fit than reusing current drafter |
| P-EAGLE | parallel token prediction targets sequential draft cost | different trained drafter; this exact target checkpoint not qualified | not selected |
| DFlash2 | newer parallel drafting approach | advertised model coverage does not establish Qwen3-14B compatibility | not selected |
| DPara / DeLS-Spec | overlap or better parallel draft conditioning | altered/trained backbone/head and dependencies on different drafters | newly reviewed, not a drop-in MVP |
| KV tiers/P-D separation | peak16/88 blocks, cleanup zero | no measured capacity pressure; no spare full BF16 target replica on these two cards | not selected |

Primary is **not** justified by claiming draft became the largest component:
target remains60-76% of serving wall time. It is a risk-adjusted next test of
across-request repeated work, with no new weights and a falsifiable limited
adapter. Secondary directly addresses the largest target compute category but
has no measured Ada kernel advantage yet. This ordering is our engineering
judgment, not a result from benchmarking the two unimplemented candidates.

## Source Facts vs Our Design

- Current SGLang uses batch-shaped inputs inside the autoregressive speculative
  step loop and separates draft extension. That supports the execution pattern,
  not importing its KV-retention contract. See
  [`draft_forward`](https://github.com/sgl-project/sglang/blob/3e7e6529002db8125967967e0aba394986eec27d/python/sglang/srt/speculative/eagle_worker_v2.py#L782).
- vLLM MRV2 describes stable request rows and staged/GPU-native preparation.
  Our measured metadata cost is too small to prioritize it here. See its
  [pinned design](https://github.com/vllm-project/vllm/blob/29468dde8b515031dc6d4d9d06bf0a2fa0442098/docs/design/model_runner_v2.md).
- FlashInfer's BF16 cuBLASLt gate explicitly includes SM89, while its low-M
  specialized backends have different restrictions. Its
  [backend source](https://github.com/flashinfer-ai/flashinfer/blob/dc5b19bb74084a9829ca204d7da5930a36bd01a4/flashinfer/gemm/gemm_base.py#L372)
  and [Autotuner v2](https://flashinfer.ai/2026/09/22/autotuner-v2.html) support a
  mode-aware selection experiment, not a predicted4090 speedup.
- Current vLLM custom all-reduce requires a working peer-access check. Our
  [hardware query](hardware-check.json) is false in both directions; no custom
  IPC transfer test or bypass was attempted. See the
  [source guard](https://github.com/vllm-project/vllm/blob/29468dde8b515031dc6d4d9d06bf0a2fa0442098/vllm/distributed/device_communicators/custom_all_reduce.py#L271).
- DeepSpec publishes Qwen3-14B drafter links. Availability does not guarantee
  exact target revision, existing feature-tap compatibility or24GiB fit. The
  inspected DFlash14B configuration uses five target feature taps and five draft
  layers, unlike the current three-tap integration. See the
  [configuration](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dflash/dflash_qwen3_14b.py)
  and [checkpoint table](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md).
- [P-EAGLE](https://arxiv.org/abs/2602.01469) changes trained token prediction.
  It is not equivalent to batching independent requests using our frozen EAGLE.
- The updated [DFlash README](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/README.md)
  lists DFlash2 checkpoints for Muse-Glimmer-30B and Qwen3.8-27B. This audit did
  not establish a matching DFlash2 Qwen3-14B checkpoint.
- Newly reviewed [DPara, September23](https://arxiv.org/html/2609.27396v1)
  separates outcome-independent backbone work from an outcome-conditioned head,
  but adapts the backbone by finetuning. Its multi-anchor/overlap machinery is
  not an execution-only change to our checkpoint. Author speedups are not
  transferred to this project.
- [DeLS-Spec](https://arxiv.org/abs/2607.07409v1) combines a fixed DFlash backbone
  with a separately trained short-context head. That modularity is interesting,
  but still requires a different drafter/head qualification here.

No candidate is rejected merely for being unfamiliar or accepted for being new.
Attention, MoE/EP, offload and P/D techniques solve important problems elsewhere;
the new measurements and this dense two-GPU envelope do not make them the
minimum useful next intervention.

## Supplemental FlashInfer Source Provenance

All three files at commit `dc5b19bb74084a9829ca204d7da5930a36bd01a4`, fetched
September25; no compilation or execution:

| path | SHA256 |
|---|---|
| `flashinfer/jit/gemm/core.py` | `439fb5301f581f5994be06504940d5df7a52981bde3b4c3145de41e71ff80182` |
| `csrc/mm_bf16_cublaslt.cu` | `28be5dee7b76ae5a986e47b4f52030dde44ae1e78360cfac29a8d3c29ce6b940` |
| `include/flashinfer/gemm/mm_bf16_cublaslt.cuh` | `86c650b9da11c96bcb9a851247ca5babf8822ef54b0579d8c99d7bc0f4de23dc` |

The JIT generator accepts architecture major8; the wrapper exposes algorithm
enumeration and cached execution; the header specifies `CUBLAS_COMPUTE_32F`.
This establishes source-level fit only. The installed Torch2.8/CUDA12.8 ABI,
workspace cost and same-input numerical behavior still need a future gate.

Full proposed ownership, correctness, benchmark and stop conditions are in
[the Phase5.2 design](../../docs/EAGLE3_PHASE5_2.md). No next phase has started.
