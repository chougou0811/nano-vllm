import unittest
from dataclasses import asdict

from nanovllm.speculative.graph_policy import make_key
from benchmarks.serving.eagle3_phase44c import digest
from benchmarks.serving.eagle3_phase44c_analysis import analyze, bounded, curve


class ShadowKeyTests(unittest.TestCase):
    def events(self, contexts):
        result = []
        for i,context in enumerate(contexts):
            key = make_key(dict(cu_seqlens_q=[0,4],context_lens=[context],block_tables=[[0]]),
                block_size=256,model_identity="test",dtype="bf16",device_class="sm89",
                tp_size=2,training=True,feature_layers=(2,20,37))
            result.append(dict(index=i,key_id=digest(asdict(key)),key=asdict(key),
                context_tuple=[context],waves=[i],request_ids=[i],completed_before=i))
        return result

    def model(self):
        return dict(exact={},by_m={4:dict(capture_ms=30,saving_ms=10,provenance="test")})

    def test_second_occurrence_is_capture_not_replay(self):
        events = self.events([64]*5)
        stats = analyze(events,self.model())
        self.assertEqual(stats["second_capture_replay_opportunities"],3)
        self.assertEqual(stats["empirical_transfer_projection"]["profitable_keys"],1)
        self.assertEqual(stats["empirical_transfer_projection"]["all_second_capture_net_ms"],0)
        self.assertEqual(bounded(events,4)["hits"],3)

    def test_occurrences_distances_and_lifetime(self):
        stats = analyze(self.events([64,65,64,66,64]),self.model())
        row = next(r for r in stats["records"] if r["count"]==3)
        self.assertEqual(row["occurrence_indices"],[0,2,4])
        self.assertEqual(row["reuse_distances"],[2,2])
        self.assertEqual(row["distinct_key_reuse_distances"],[1,1])
        self.assertEqual(row["lifetime_verifications"],4)
        self.assertEqual(stats["field_ablation_cardinality"]["max_k"],1)
        self.assertEqual(stats["unique_keys"],3)

    def test_unknown_savings_do_not_turn_into_zero_cost(self):
        model = self.model()
        model["by_m"][4]["saving_ms"] = None
        stats = analyze(self.events([64]*30),model)
        self.assertEqual(stats["empirical_transfer_projection"]["unknown_keys"],1)
        self.assertIsNone(stats["empirical_transfer_projection"]["full_all_second_capture_net_ms"])
        self.assertIsNone(stats["records"][0]["break_even_replays"])
        self.assertEqual(stats["threshold_scenarios"][18]["keys"],1)

    def test_capture_budget_is_frozen(self):
        events = self.events([c for c in range(40,80) for _ in range(3)])
        self.assertEqual(bounded(events,4)["captures"],16)
        self.assertEqual(bounded(events,4)["hits"],16)

    def test_curve_is_monotonic(self):
        rows = curve(self.events([64,65,64,66,65]))
        self.assertEqual([r["unique_keys"] for r in rows],[1,2,2,3,3])
        self.assertEqual([r["reusable_keys"] for r in rows],[0,0,1,1,2])

    def test_ineligible_does_not_capture(self):
        events = self.events([64,64,64])
        events[1]["key_id"] = None
        events[1]["key"] = None
        stats = analyze(events,self.model())
        self.assertEqual(stats["eligible"],2)
        self.assertEqual(bounded(events,4)["hits"],0)

    def test_extended_workload_preserves_frozen_distribution(self):
        from benchmarks.serving.eagle3_phase44c import requests_for
        from benchmarks.serving.eagle3_phase42 import build_requests, WORKLOADS, SHAPES
        class Tokenizer:
            vocab_size = 150000
            def encode(self, text, add_special_tokens=False):
                return list(range(100,2100))
        tokenizer = Tokenizer()
        for family in WORKLOADS:
            short = requests_for(tokenizer,family,32)
            long = requests_for(tokenizer,family,512)
            old = build_requests(tokenizer,family,4,0)
            self.assertEqual([(r.prompt,r.output_limit) for r in old],
                             [(r.prompt,r.output_limit) for r in long[:len(old)]])
            self.assertEqual([r.prompt for r in short],[r.prompt for r in long[:32]])
            self.assertEqual(len({tuple(r.prompt) for r in long}),512)
            self.assertEqual(set(len(r.prompt) for r in long),set(SHAPES[family][0]))
            self.assertEqual(set(r.output_limit for r in long),set(SHAPES[family][1]))

    def test_empty_window(self):
        stats = analyze([],self.model())
        self.assertEqual(stats["unique_keys"],0)
        self.assertEqual(stats["second_capture_hit_rate"],0)

    def test_negative_savings_are_retained(self):
        model = self.model()
        model["by_m"][4]["saving_ms"] = -10
        stats = analyze(self.events([64]*5),model)
        self.assertIsNone(stats["records"][0]["break_even_replays"])
        self.assertEqual(stats["empirical_transfer_projection"]["all_second_capture_net_ms"],-60)
        self.assertEqual(stats["empirical_transfer_projection"]["profitable_keys"],0)


if __name__ == "__main__":
    unittest.main()
