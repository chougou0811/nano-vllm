from dataclasses import dataclass
from time import perf_counter_ns

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import SequenceStatus


def make_scheduler(config, clock=perf_counter_ns):
    if config.scheduler_policy == "original":
        return Scheduler(config)
    if config.scheduler_policy == "slo-v2":
        from nanovllm.engine.progress_scheduler import ProgressScheduler
        return ProgressScheduler(config, clock)
    if config.scheduler_policy == "slo-v1":
        from copy import copy
        config = copy(config)
        config.scheduler_policy = "slo-aware"
    return PolicyScheduler(config, clock)


@dataclass
class RequestTiming:
    arrival_ns: int
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    last_service_step: int = 0


class PolicyScheduler(Scheduler):
    """Pure-prefill/pure-decode arbitration; original KV and commit operations."""

    def __init__(self, config, clock=perf_counter_ns):
        super().__init__(config)
        self.config = config
        self.policy = config.scheduler_policy
        self.clock = clock
        self.timings = {}
        self.step_index = 0
        self.prefill_streak = self.decode_streak = 0
        self.alpha = config.scheduler_ewma_alpha
        self.ttft_ns = config.scheduler_ttft_ms * 1e6
        self.tpot_ns = config.scheduler_tpot_ms * 1e6
        self.decode_ns = config.scheduler_initial_step_ms * 1e6
        self.prefill_ns_per_token = self.decode_ns / config.scheduler_prefill_chunk
        self.last_decision = {}
        self.max_service_gap_steps = 0

    def add(self, seq):
        if seq.max_tokens < 1 or seq.num_prompt_tokens + seq.max_tokens > self.config.max_model_len:
            raise ValueError("Request must fit configured context with a positive output limit")
        if self._completion_blocks(seq) > len(self.block_manager.blocks):
            raise ValueError("Request output limit cannot fit KV capacity under reservation policy")
        now = self.clock()
        arrival = getattr(seq, "arrival_time_ns", now)
        if not isinstance(arrival, int) or arrival > now:
            raise ValueError("arrival_time_ns must be a non-future monotonic timestamp")
        self.timings[seq.seq_id] = RequestTiming(arrival, last_service_step=self.step_index)
        super().add(seq)

    def _completion_blocks(self, seq):
        return (seq.num_prompt_tokens + seq.max_tokens + self.block_size - 1) // self.block_size

    def _reserve_needed(self):
        return sum(max(0, self._completion_blocks(s) - len(s.block_table))
                   for s in (*self.waiting, *self.running) if s.block_table)

    def _prefill_cost(self, tokens):
        return max(self.decode_ns, tokens * self.prefill_ns_per_token)

    def _decode_deadline(self, seq):
        state = self.timings[seq.seq_id]
        return state.last_token_ns + self.tpot_ns

    def _choose(self, now):
        limit = self.max_num_batched_tokens
        minimum = min(self.config.scheduler_min_prefill_chunk, limit)
        fixed = min(self.config.scheduler_prefill_chunk, limit)
        if not self.waiting:
            return False, 0, "decode-only", None, None
        if not self.running:
            return True, fixed if self.policy == "static" else limit, "prefill-only", None, None
        if self.policy == "static":
            return self.prefill_streak == 0, fixed, "static-alternation", None, None

        head = self.waiting[0]
        pending = head.num_tokens - head.num_cached_tokens
        prefill_slack = self.timings[head.seq_id].arrival_ns + self.ttft_ns - now - self._prefill_cost(pending)
        decode_slack = min(self._decode_deadline(s) for s in self.running) - now - self.decode_ns
        if self.prefill_streak >= self.config.scheduler_max_prefill_steps:
            return False, 0, "decode-fairness", prefill_slack, decode_slack
        if self.decode_streak >= self.config.scheduler_max_decode_steps:
            return True, minimum, "prefill-fairness", prefill_slack, decode_slack

        # Spend only slack left after reserving an estimated decode step.
        budgets = [minimum]
        while budgets[-1] < limit:
            budgets.append(min(limit, budgets[-1] * 2))
        fitting = [b for b in budgets if self._prefill_cost(b) <= decode_slack]
        if fitting:
            return True, max(fitting), "prefill-fits-slack", prefill_slack, decode_slack
        if prefill_slack < 0 and prefill_slack / self.ttft_ns < decode_slack / self.tpot_ns:
            return True, minimum, "ttft-pressure", prefill_slack, decode_slack
        return False, 0, "decode-deadline", prefill_slack, decode_slack

    def _prefill(self, budget):
        selected = []
        while self.waiting and len(selected) < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            if not seq.block_table:
                cached = self.block_manager.can_allocate(seq)
                if cached < 0:
                    break
                # Conservative logical reservation, not physical preallocation.
                # Count full blocks even if prefixes could be shared, so admission
                # cannot consume blocks needed to finish resident requests.
                if len(self.block_manager.free_block_ids) < self._completion_blocks(seq) + self._reserve_needed():
                    break
                self.block_manager.allocate(seq, cached)
            remaining = seq.num_tokens - seq.num_cached_tokens
            seq.num_scheduled_tokens = min(remaining, budget)
            budget -= seq.num_scheduled_tokens
            seq.is_prefill = True
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            selected.append(seq)
            if seq.status == SequenceStatus.WAITING:
                break
        return selected

    def _decode(self):
        if self.policy == "slo-aware":
            # Equal per-token SLOs make EDF equivalent to oldest-token-first;
            # it cannot repeatedly favor a just-served request under overload.
            ordered = sorted(self.running, key=lambda s: (self._decode_deadline(s),
                              self.timings[s.seq_id].last_service_step, s.seq_id))
        else:
            ordered = list(self.running)
        selected = ordered[:self.max_num_seqs]
        for seq in selected:
            if not self.block_manager.can_append(seq):
                raise RuntimeError("KV reservation invariant violated before decode")
            self.block_manager.may_append(seq)
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.running.remove(seq)
        self.running.extend(selected)
        return selected

    def schedule(self):
        start = self.clock()
        prefill, budget, reason, p_slack, d_slack = self._choose(start)
        selected = self._prefill(budget) if prefill else self._decode()
        if not selected and prefill and self.running:
            prefill, reason = False, "kv-admission-backpressure"
            selected = self._decode()
        if not selected:
            raise RuntimeError("No feasible request under KV reservation policy")
        self.step_index += 1
        for seq in selected:
            state = self.timings[seq.seq_id]
            self.max_service_gap_steps = max(self.max_service_gap_steps, self.step_index-state.last_service_step)
            state.last_service_step = self.step_index
        self.prefill_streak = self.prefill_streak + 1 if prefill else 0
        self.decode_streak = self.decode_streak + 1 if not prefill else 0
        self._step_start = start
        self._step_tokens = sum(s.num_scheduled_tokens for s in selected)
        self.last_decision = dict(policy=self.policy, reason=reason,
            prefill_budget=budget if prefill else 0, prefill_slack_ns=p_slack, decode_slack_ns=d_slack,
            predicted_decode_ns=self.decode_ns, prefill_ns_per_token=self.prefill_ns_per_token,
            reserved_growth_blocks=self._reserve_needed(), max_service_gap_steps=self.max_service_gap_steps)
        return selected, prefill

    def postprocess(self, seqs, token_ids, is_prefill):
        before = [s.num_completion_tokens for s in seqs]
        super().postprocess(seqs, token_ids, is_prefill)
        now = self.clock()
        elapsed = max(1, now - self._step_start)
        if is_prefill:
            sample = elapsed / self._step_tokens
            self.prefill_ns_per_token += self.alpha * (sample - self.prefill_ns_per_token)
        else:
            self.decode_ns += self.alpha * (elapsed - self.decode_ns)
        for seq, count in zip(seqs, before):
            state = self.timings[seq.seq_id]
            if seq.num_completion_tokens > count:
                if state.first_token_ns is None:
                    state.first_token_ns = now
                state.last_token_ns = now
            if seq.is_finished:
                del self.timings[seq.seq_id]
