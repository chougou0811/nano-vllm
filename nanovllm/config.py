import os
import math
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    scheduler_policy: str = "original"
    scheduler_prefill_chunk: int = 256
    scheduler_min_prefill_chunk: int = 128
    scheduler_ttft_ms: float = 2000.0
    scheduler_tpot_ms: float = 100.0
    scheduler_initial_step_ms: float = 50.0
    scheduler_ewma_alpha: float = 0.2
    scheduler_max_prefill_steps: int = 2
    scheduler_max_decode_steps: int = 4
    scheduler_v2_min_chunk: int = 256
    scheduler_v2_max_chunk: int = 1024
    scheduler_v2_overload_chunk: int = 512
    scheduler_v2_overload: bool = True
    scheduler_v2_cost_model: str = "bucketed"

    def __post_init__(self):
        if self.scheduler_policy not in {"original", "static", "slo-aware", "slo-v1", "slo-v2"}:
            raise ValueError("Unknown scheduler policy")
        for name in ["scheduler_prefill_chunk", "scheduler_min_prefill_chunk",
                     "scheduler_max_prefill_steps", "scheduler_max_decode_steps"]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ["scheduler_ttft_ms", "scheduler_tpot_ms", "scheduler_initial_step_ms"]:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.scheduler_ewma_alpha <= 1:
            raise ValueError("scheduler_ewma_alpha must be in (0, 1]")
        if not 0 < self.scheduler_v2_min_chunk <= self.scheduler_v2_overload_chunk <= self.scheduler_v2_max_chunk:
            raise ValueError("V2 chunks must satisfy 0 < minimum <= overload <= maximum")
        if self.scheduler_v2_cost_model not in {"bucketed", "scalar"}:
            raise ValueError("Unknown V2 cost model")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
