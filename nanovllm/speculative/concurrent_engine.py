"""Opt-in Phase 4.1 engine that leaves the frozen ordinary engine unchanged."""
import atexit
from dataclasses import fields

from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.policy_scheduler import make_scheduler
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.speculative.coordinator import SpeculativeCoordinator
from nanovllm.speculative.scheduler_adapter import SpeculativeSchedulerAdapter


class ConcurrentModelRunner(ModelRunner):
    """ModelRunner with an additive batched EAGLE RPC endpoint."""

    def eagle3_batch(self, operation, payload):
        from nanovllm.speculative.batched_runtime import dispatch
        return dispatch(self, operation, payload)


class ConcurrentLLMEngine(LLMEngine):
    """Fixed-K greedy concurrent EAGLE engine selected explicitly by callers.

    SamplingParams.temperature is retained for Sequence API compatibility but is
    ignored: proposal and target acceptance are always greedy argmax operations.
    """

    def __init__(self, model, *, draft_path, reference_path,
                 speculative_length=3, audit=False, target_graph_config=None,
                 draft_batching=False, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {key: value for key, value in kwargs.items()
                         if key in config_fields}
        config = Config(model, **config_kwargs)
        if not config.enforce_eager:
            raise ValueError("Concurrent EAGLE requires enforce_eager=True")
        if config.tensor_parallel_size != 2:
            raise ValueError("Phase 4.1 concurrent EAGLE requires TP=2")
        if speculative_length != 3:
            raise ValueError("Phase 4.1 concurrent EAGLE requires fixed K=3")
        if config.scheduler_policy != "original":
            raise ValueError("Phase 4.1 supports the original Scheduler only")
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self._closed = False
        self._faulted = False
        runner_type = ConcurrentModelRunner
        if target_graph_config is not None:
            from nanovllm.speculative.graph_runner import GraphConcurrentModelRunner
            runner_type = GraphConcurrentModelRunner
        ctx = mp.get_context("spawn")
        for rank in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=runner_type,
                                  args=(config, rank, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = runner_type(config, 0, self.events)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
            config.eos = self.tokenizer.eos_token_id
            self.scheduler = make_scheduler(config)
            if type(self.scheduler) is not Scheduler:
                raise ValueError("Phase 4.1 requires the original Scheduler")
            self.speculative_coordinator = SpeculativeCoordinator(
                self,
                draft_path,
                reference_path,
                fixed_k=speculative_length,
                audit=audit,
            )
            if draft_batching:
                from nanovllm.speculative.draft_batch import DraftBatchExecutor
                self.speculative_coordinator.draft_batch_executor = DraftBatchExecutor()
            self.scheduler = SpeculativeSchedulerAdapter(
                self.scheduler, self.speculative_coordinator
            )
            if target_graph_config is not None:
                self.model_runner.call("graph_control", "init", target_graph_config)
        except Exception:
            self.model_runner.call("exit")
            del self.model_runner
            for process in self.ps:
                process.join()
            self._closed = True
            raise
        atexit.register(self.exit)

    def step(self):
        if self._faulted:
            raise RuntimeError("Concurrent EAGLE engine is faulted")
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = (sum(seq.num_scheduled_tokens for seq in seqs)
                      if is_prefill else -len(seqs))
        try:
            if is_prefill:
                token_ids = self.speculative_coordinator.prefill(seqs)
                self.scheduler.postprocess_prefill(seqs, token_ids)
            else:
                commits = self.speculative_coordinator.decode(seqs)
                self.scheduler.postprocess_decode(seqs, commits)
                num_tokens = -sum(len(item.committed_token_ids) for item in commits)
        except Exception as error:
            self._faulted = True
            self.scheduler.abort_all(error)
            raise
        outputs = [(seq.seq_id, seq.completion_token_ids)
                   for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def exit(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.scheduler.close_all()
        finally:
            self.model_runner.call("exit")
            del self.model_runner
            for process in self.ps:
                process.join()
