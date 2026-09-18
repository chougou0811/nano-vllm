import unittest
from collections import Counter

from test_scheduler_policy import Clock, MS, config, seq
from benchmarks.serving.scheduler_finalization import (
    POLICIES, PROFILES, attach_diagnostics, build_plans, policy_config,
)
from benchmarks.serving.workload import IsolatedWorkloads
from benchmarks.serving.scheduler_finalization_report import overload_stats, paired_stats
from nanovllm.engine.policy_scheduler import make_scheduler


class FinalizationTests(unittest.TestCase):
    def test_overload_retention_is_not_an_evaluated_predicate(self):
        steps = [dict(step_latency_ms=2,policy_decision=dict(overloaded=True,
                     overload_evaluated=True,overload_transition=1,overload_decode_predicate=True)),
                 dict(step_latency_ms=6,policy_decision=dict(overloaded=True,
                     overload_evaluated=False,overload_transition=0)),
                 dict(step_latency_ms=2,policy_decision=dict(overloaded=False,
                     overload_evaluated=True,overload_transition=1))]
        actual = overload_stats(steps)
        self.assertAlmostEqual(actual["recorded_fraction"],2/3)
        self.assertEqual(actual["evaluated_fraction"],.5)
        self.assertEqual(actual["recorded_step_time_fraction"],.8)
        self.assertEqual(actual["retained_on_without_waiting"],1)
        self.assertEqual(actual["predicates"],dict(decode=1,neither=1))

    def test_paired_bootstrap_constant_difference(self):
        result = paired_stats([2]*5)
        self.assertEqual(result["mean"],2)
        self.assertEqual(result["bootstrap_mean_ci95"],[2,2])
        self.assertEqual(result["bootstrap_samples"],3125)
        larger = paired_stats([2]*8)
        self.assertEqual(larger["bootstrap_samples"],10000)
        self.assertEqual(larger["bootstrap_mean_ci95"],[2,2])

    def test_paired_independent_plan_and_balanced_order(self):
        plans = build_plans(IsolatedWorkloads(range(1000)))
        self.assertEqual(len(plans),150)
        firsts = set()
        for i in range(0,len(plans),5):
            group = plans[i:i+5]
            self.assertTrue(all(p["requests"] == group[0]["requests"] for p in group))
            tokens = {s.prompt_token_ids[0] for s in group[0]["requests"]}
            self.assertFalse(tokens & firsts)
            firsts |= tokens
            self.assertTrue(all(len(s.prompt_token_ids)+s.output_length <= 2048 for s in group[0]["requests"]))
        counts = Counter((p["stage"],p["interval"],p["policy"]) for p in plans)
        self.assertTrue(all(n == 5 for n in counts.values()))
        for stage,(_,intervals) in PROFILES.items():
            for interval in intervals:
                order = [p["policy"] for p in plans if p["stage"]==stage and p["interval"]==interval]
                for position in range(5):
                    self.assertEqual(set(order[position::5]),set(POLICIES))
        with self.assertRaises(ValueError):
            build_plans(IsolatedWorkloads(range(1000)),4)

    def test_config_is_copy_and_frozen(self):
        base = config("original")
        self.assertEqual(policy_config(base,"static-512").scheduler_prefill_chunk,512)
        self.assertFalse(policy_config(base,"v2-no-overload").scheduler_v2_overload)
        self.assertEqual(base.scheduler_policy,"original")
        with self.assertRaises(ValueError):
            policy_config(base,"unknown")

    def test_diagnostics_do_not_change_choices_or_costs(self):
        for name in ["frozen-v2","v2-no-overload"]:
            outputs = []
            for observe in [False,True]:
                clock = Clock()
                scheduler = make_scheduler(policy_config(config("original"),name),clock)
                if observe:
                    attach_diagnostics(scheduler)
                for i,length in enumerate([255,257,511,1856]):
                    scheduler.add(seq(length,12,i+1))
                trace = []
                while not scheduler.is_finished():
                    batch,prefill = scheduler.schedule()
                    d = scheduler.last_decision
                    trace.append((prefill,[s.num_scheduled_tokens for s in batch],d["reason"],d["prefill_budget"]))
                    if observe and d["overload_evaluated"]:
                        self.assertEqual(d["infeasible"],d["overload_decode_predicate"] or d["overload_backlog_predicate"])
                    clock.now += 40*MS
                    scheduler.postprocess(batch,[7]*len(batch),prefill)
                self.assertFalse(scheduler.block_manager.used_block_ids)
                outputs.append((trace,scheduler.costs.values))
            self.assertEqual(*outputs)


if __name__ == "__main__":
    unittest.main()
