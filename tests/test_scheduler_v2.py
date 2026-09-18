import unittest

from test_scheduler_policy import Clock, MS, config, drain, seq
from nanovllm.engine.policy_scheduler import make_scheduler
from nanovllm.engine.progress_scheduler import StepCosts


class ProgressTests(unittest.TestCase):
    def test_v1_alias_preserves_decisions(self):
        traces = []
        for policy in ["slo-aware", "slo-v1"]:
            clock = Clock()
            scheduler = make_scheduler(config(policy), clock)
            for i in range(8):
                scheduler.add(seq(257, 12, i+1))
            trace = drain(scheduler, clock)
            traces.append([(p, len(ids), tokens) for p, ids, tokens in trace])
        self.assertEqual(*traces)

    def test_both_classes_progress_under_expired_deadlines(self):
        clock = Clock()
        scheduler = make_scheduler(config("slo-v2", scheduler_ttft_ms=1, scheduler_tpot_ms=1), clock)
        requests = [seq(1536, 16, i+1) for i in range(12)]
        for request in requests:
            scheduler.add(request)
        contested = []
        while not scheduler.is_finished():
            both = bool(scheduler.waiting and scheduler.running)
            previous = scheduler.prefill_streak
            batch, prefill = scheduler.schedule()
            if both:
                self.assertEqual(prefill, previous == 0)
                if prefill:
                    self.assertGreaterEqual(scheduler.last_decision["prefill_budget"], 256)
                contested.append(prefill)
            clock.now += 100*MS
            scheduler.postprocess(batch, [7]*len(batch), prefill)
        self.assertTrue(any(contested) and not all(contested))
        self.assertTrue(all(r.num_completion_tokens == 16 for r in requests))
        self.assertFalse(scheduler.block_manager.used_block_ids)

    def test_low_memory_reservation_still_drains(self):
        clock = Clock()
        scheduler = make_scheduler(config("slo-v2", num_kvcache_blocks=4), clock)
        for i in range(8):
            scheduler.add(seq(257, 260, i+1))
        drain(scheduler, clock)

    def test_boundaries_and_cache_reuse(self):
        for length in [255,256,257,511,512,1536,2044]:
            clock = Clock()
            scheduler = make_scheduler(config("slo-v2"), clock)
            for repeat in range(2):
                request = seq(length, 4)
                scheduler.add(request)
                drain(scheduler, clock)
                self.assertEqual(request.completion_token_ids, [7]*4)

    def test_shape_costs_do_not_merge_small_and_large_chunks(self):
        costs = StepCosts(50*MS, .2)
        small = costs.key(True, 256, 1, 256)
        large = costs.key(True, 1024, 1, 1024)
        costs.observe(small, 50*MS)
        costs.observe(large, 120*MS)
        self.assertEqual(costs.predict(True, 256, 1, 256), 50*MS)
        self.assertEqual(costs.predict(True, 1024, 1, 1024), 120*MS)
        self.assertNotEqual(small, costs.key(True, 256, 2, 256))
        self.assertNotEqual(small, costs.key(True, 256, 1, 2048))

    def test_overload_disable_and_scalar_ablation(self):
        for cost in ["scalar", "bucketed"]:
            clock = Clock()
            scheduler = make_scheduler(config("slo-v2", scheduler_v2_overload=False,
                                              scheduler_v2_cost_model=cost), clock)
            for i in range(8):
                scheduler.add(seq(511, 16, i+1))
            drain(scheduler, clock)
            self.assertFalse(scheduler.overloaded)

    def test_small_configured_token_budget_is_respected(self):
        clock = Clock()
        scheduler = make_scheduler(config("slo-v2", max_num_batched_tokens=64), clock)
        scheduler.add(seq(257))
        trace = drain(scheduler, clock)
        self.assertTrue(all(sum(tokens) <= 64 for prefill, _, tokens in trace if prefill))

    def test_residents_progress_during_continuous_arrivals(self):
        clock = Clock()
        scheduler = make_scheduler(config("slo-v2"), clock)
        early = [seq(1, 100, i+1) for i in range(4)]
        for request in early:
            scheduler.add(request)
        last = {}
        for step in range(400):
            scheduler.add(seq(256, 16, step+100))
            selected, prefill = scheduler.schedule()
            for request in selected:
                if request.seq_id in last:
                    # Capacity bounds residents; newly admitted requests append
                    # behind existing residents instead of overtaking them.
                    self.assertLessEqual(step-last[request.seq_id], 34)
                last[request.seq_id] = step
            clock.now += 20*MS
            scheduler.postprocess(selected, [7]*len(selected), prefill)
        self.assertTrue(all(r.num_completion_tokens > 1 for r in early))
        drain(scheduler, clock)


if __name__ == "__main__":
    unittest.main()
