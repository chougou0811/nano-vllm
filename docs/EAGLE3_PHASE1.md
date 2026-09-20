# EAGLE-3 Phase 1 Audit and Contract

Scheduler checkpoint: 6695617. All three Scheduler implementations and Config
remain unchanged. Existing generate/run/sampling paths remain unchanged. The
dirty AGENTS.md is user-owned and is not overwritten; the user's new Phase 1
authorization supersedes its older stage description.

## Pinned Sources

- Target Qwen/Qwen3-14B: 40c069824f4251a91eefaf281ebe4c544efd3e18.
- Draft AngelSlim/Qwen3-14B_eagle3: 3d13517724e81cb409ddf1d4650772ec52f1e18e.
  https://huggingface.co/AngelSlim/Qwen3-14B_eagle3
- SafeAILab/EAGLE: cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b.
  https://github.com/SafeAILab/EAGLE
  cnets.py, configs.py, utils.py and modeling_qwen3_kv.py define the reference.

The draft declares hidden5120, full vocab151936, draft vocab32000, head_dim128,
40 query/8 KV heads, one draft decoder, intermediate17408 and RoPE1e6. It is a
dedicated feature-conditioned EAGLE-3 head, not a smaller Qwen language model.
Its architecture name is LlamaForCausalLMEagle3, not the target Qwen3 decoder:
the reference draft uses two normalized streams, concatenated attention input,
3H-to-H fusion and a reduced-vocabulary head with d2t offsets. Do not load it as
an ordinary LlamaForCausalLM. Strict state-dict checks are required.

No exact training target revision or independent tokenizer is published in the
downloaded draft config. Architecture compatibility is checkable; exact training
revision compatibility is unverified. Use only the pinned target tokenizer and
its embedding weights, validate draft-to-target token mapping, and report the
remaining provenance limitation. Do not claim publisher throughput as ours.

## Features and Tensor Parallelism

The pinned Qwen3 reference captures the residual stream BEFORE layers
[2, num_layers//2, num_layers-3], i.e. [2,20,37] for this 40-layer target.
Nano-vllm's fused residual representation needs float(hidden)+float(residual),
cast back to the model dtype, not the MLP output alone and not final RMSNorm.
Scoped forward pre-hooks expose these features only during explicit speculation.

Embedding and RowParallelLinear all-reduces make these full-width hidden states
replicated across TP ranks. Attention Q/K/V heads and target KV remain sharded.
Both ranks execute verification; only rank0 runs the unsharded reference draft
and takes argmax/acceptance decisions. Committed lengths and IDs are sent via
the existing ordered RPC channel. Optional tensor/rank-state audits use NCCL.
No rank0-only call to a TP embedding or LM head is allowed.

## Minimal Execution Mode

Opt-in, one request, idle engine, BF16 eager, fixed linear-chain K. No scheduler
calls are bypassed for already-enqueued requests: reject entry unless queues and
used KV are empty. This is an isolated integration path, not online concurrent
speculative scheduling. Existing generate remains the speculation-off API.

The original sampler has no greedy branch (temperature zero divides by zero).
Do not change it: use explicit argmax in this API. Correctness/performance compare
against a test-only greedy adapter on the unchanged original target path, and
separately check ordinary positive-temperature off-path parity with fixed seeds.
Stochastic acceptance, tree proposals and concurrent batching are out of scope.

The known pending target token precedes K draft proposals. Verify all K+1 inputs
in one causal target forward, retaining every position's logits. Accept exactly
the longest proposal prefix equal to target argmax, as reference
evaluate_posterior(logits_processor=None). Emit the target argmax at the first
rejection, or the bonus target token after all K accept. Clip only for EOS,
remaining output/context budget. K does not adapt to acceptance, SLO or latency.

## KV Transaction

Reserve private physical pages using unchanged BlockManager allocation APIs for
the request capacity. No hashes are published for these pages. The target KV
cursor normally trails committed tokens by one pending token. Tentative writes
extend it during verification; commit retains the pending input and accepted
proposal inputs. Explicitly zero rejected suffix slots on all ranks, lower the
valid cursor, and append only retained target features. Only accepted proposals
and target fallback/bonus tokens enter Sequence. Clear all leased pages and release references on request exit,
including exceptional host-side exits. No stale speculative prefix is shareable.

Draft KV is deliberately rebuilt every proposal from committed target features
paired with next-token embeddings (input_ids shifted by one). Subsequent draft
steps feed draft hidden output back into the same reference layer. Proposal KV
and draft hidden states are discarded after verification; the next call rebuilds
from target-verified features. This is slow but provides an explicit rollback
baseline before persistent speculative KV management.

Model weights remain resident between requests; request KV/features/proposals do
not. Account for the extra full target embedding and draft weights on rank0;
start with target cache utilization0.75 for memory headroom. No speed claim.

## Invocation and Measurement

The optional reference checkout is external, not vendored. Check out the pinned
SafeAILab commit above at the supplied reference_path; retain its LICENSE. Core
reference file digests and the draft weight SHA-256 are checked before loading.
The draft directory needs config.json, pytorch_model.bin and download_manifest.json
with sha256, verified=true, repo and revision fields. The audited download manifest
and helper are preserved under /root/autodl-tmp/benchmarks/eagle3-phase1.

```python
result = engine.generate_eagle3(
    "The capital of France is",
    draft_path="/root/autodl-tmp/models/Qwen3-14B_eagle3",
    reference_path="/root/autodl-tmp/references/eagle-pinned",
    speculative_length=3, max_tokens=32, audit=True,
)
```

Do not pass speculative settings to the original generate API and assume that
they take effect. There is no stochastic speculative sampler in Phase 1.
The optional API requires an idle eager engine and sufficient private page
capacity. It does not enqueue work in the frozen Scheduler.

All clocks are perf_counter_ns. Token timestamps are captured only after actual
Sequence commits; a multi-token commit shares one timestamp. Consequently ITL
contains zeros within bursts, not evidence of zero user-visible streaming delay.
This non-streaming API returns only at completion: TTFT is internal token-ready
latency, not HTTP/network TTFT. Model/draft loading is outside the request window;
cleanup time is separately retained. Audit collectives, feature hooks, rollback
clearing and Python overhead are inside the measured request window. These are
host-observed latencies, not isolated GPU kernel timings.

Raw records retain every proposed/target/committed token, rejected count, physical
clear status, rank cursor/digest, forward counts and step latencies. Terminal
capacity can reduce K, including a final zero-proposal verification. Acceptance
per verification includes these terminal calls; prefill is a target forward but
is not a verification. No timing samples are filtered.

Frozen-source tests compare existing method ASTs and protected modules against
6695617. GPU diagnostics use a test-only argmax sampler adapter on the original
path, then restore the original sampler. Positive-temperature fixed-seed parity
is tested independently, not inferred from the adapter.
