"""Progress-balanced scheduling without changing model or KV operations."""
from math import ceil
from time import perf_counter_ns

from nanovllm.engine.policy_scheduler import PolicyScheduler


def bucket(value):
    return 1 << (max(1, int(value)) - 1).bit_length()


class StepCosts:
    """Host step EWMA by phase, chunk, batch and context; bounded table size."""

    def __init__(self, seed_ns, alpha):
        self.seed_ns = seed_ns
        self.alpha = alpha
        self.values = {}

    def key(self, prefill, tokens, batch, context):
        return prefill, bucket(tokens), bucket(batch), bucket(context)

    def observe(self, key, elapsed):
        old = self.values.get(key, elapsed)
        self.values[key] = old + self.alpha * (elapsed - old)

    def predict(self, prefill, tokens, batch, context):
        key = self.key(prefill, tokens, batch, context)
        if key in self.values:
            return self.values[key]
        # Unseen shapes keep a fixed-cost floor; no inverse-token feedback loop.
        candidates = [(k, v) for k, v in self.values.items() if k[0] == prefill]
        if not candidates:
            return self.seed_ns * (0.5 + tokens / 512 if prefill else 1)
        def distance(item):
            k, _ = item
            return sum(abs(a.bit_length()-b.bit_length()) for a,b in zip(k[1:], key[1:]))
        nearest, elapsed = min(candidates, key=distance)
        token_factor = 0.5 + 0.5 * key[1] / nearest[1] if prefill else 1
        context_factor = (key[3] / nearest[3]) ** 0.1
        batch_factor = (key[2] / nearest[2]) ** 0.1
        return elapsed * token_factor * context_factor * batch_factor


class ProgressScheduler(PolicyScheduler):
    """FIFO prefill, round-robin decode, one turn per class under contention."""

    def __init__(self, config, clock=perf_counter_ns):
        super().__init__(config, clock)
        self.costs = StepCosts(self.decode_ns, self.alpha)
        self.overloaded = False
        self.healthy_turns = 0
        self.progress_decision = {}

    def _prefill_shape(self, budget):
        tokens = batch = total_context = 0
        for seq in self.waiting:
            n = min(budget - tokens, seq.num_tokens - seq.num_cached_tokens)
            if n <= 0 or batch == self.max_num_seqs:
                break
            tokens += n
            batch += 1
            total_context += seq.num_cached_tokens + n
            if n < seq.num_tokens - seq.num_cached_tokens:
                break
        return max(1, tokens), max(1, batch), max(1, total_context // max(1, batch))

    def _predict_prefill(self, budget):
        tokens, batch, context = self._prefill_shape(budget)
        if self.config.scheduler_v2_cost_model == "scalar":
            return self._prefill_cost(tokens)
        return self.costs.predict(True, tokens, batch, context)

    def _predict_decode(self):
        selected = list(self.running)[:self.max_num_seqs]
        if self.config.scheduler_v2_cost_model == "scalar" or not selected:
            return self.decode_ns
        return self.costs.predict(False, len(selected), len(selected),
                                  sum(s.num_tokens for s in selected) // len(selected))

    def _choose(self, now):
        if not self.waiting:
            self.progress_decision = dict(overloaded=self.overloaded)
            return False, 0, "v2-decode-only", None, None
        lo = min(self.config.scheduler_v2_min_chunk, self.max_num_batched_tokens)
        hi = min(self.config.scheduler_v2_max_chunk, self.max_num_batched_tokens)
        budgets = [lo]
        while budgets[-1] < hi:
            budgets.append(min(hi, budgets[-1]*2))
        head = self.waiting[0]
        remaining = head.num_tokens - head.num_cached_tokens
        age = max(0, now - self.timings[head.seq_id].arrival_ns)
        slack = self.ttft_ns - age
        decode = self._predict_decode()
        rounds = ceil(len(self.running) / self.max_num_seqs)
        round_cost = rounds * decode
        backlog = sum(s.num_tokens - s.num_cached_tokens for s in self.waiting)
        cost_min = self._predict_prefill(lo)
        infeasible = (round_cost + cost_min > self.tpot_ns or
                      ceil(backlog / lo) * (cost_min + decode) > max(slack, 1))
        if self.config.scheduler_v2_overload:
            if infeasible:
                self.overloaded, self.healthy_turns = True, 0
            elif self.overloaded:
                self.healthy_turns += 1
                if self.healthy_turns >= 3:
                    self.overloaded = False
        # Completion time includes every intervening decode turn, not only one
        # hypothetical large prefill. Age and unfinished work determine urgency.
        forecasts = {b: ceil(remaining/b) * (self._predict_prefill(b) + (decode if self.running else 0))
                     for b in budgets}
        feasible = [b for b in budgets if forecasts[b] <= slack]
        desired = min(feasible) if feasible else hi
        if self.overloaded:
            desired = max(desired, min(hi, self.config.scheduler_v2_overload_chunk))
        # A soft step-cost cap limits large chunks, but never cancels a prefill
        # turn. Infeasible deadlines do not imply suppressing useful progress.
        cap = max(self.tpot_ns, 2*decode)
        affordable = [b for b in budgets if b <= desired and self._predict_prefill(b) <= cap]
        budget = max(affordable) if affordable else lo
        self.progress_decision = dict(overloaded=self.overloaded, infeasible=infeasible,
            waiting_age_ns=age, head_remaining_tokens=remaining, waiting_tokens=backlog,
            head_progress=head.num_cached_tokens / head.num_prompt_tokens,
            predicted_decode_round_ns=round_cost, predicted_prefill_ns=self._predict_prefill(budget),
            completion_forecasts_ns=forecasts, desired_budget=desired, cost_model=self.config.scheduler_v2_cost_model)
        if not self.running:
            return True, budget, "v2-prefill-only", slack, None
        if self.prefill_streak:
            return False, 0, "v2-decode-turn", slack, self.tpot_ns-round_cost
        return True, budget, "v2-overload-progress" if self.overloaded else "v2-prefill-progress", slack, self.tpot_ns-round_cost

    def schedule(self):
        selected, prefill = super().schedule()
        context = sum(s.num_cached_tokens+s.num_scheduled_tokens for s in selected) // len(selected)
        self._cost_key = self.costs.key(prefill, self._step_tokens, len(selected), context)
        self.last_decision.update(self.progress_decision)
        return selected, prefill

    def postprocess(self, seqs, token_ids, is_prefill):
        super().postprocess(seqs, token_ids, is_prefill)
        self.costs.observe(self._cost_key, max(1, self.clock()-self._step_start))
