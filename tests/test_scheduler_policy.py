from dataclasses import fields
from types import SimpleNamespace
import pickle
import random
import unittest

from nanovllm.config import Config
from nanovllm.engine.policy_scheduler import PolicyScheduler, make_scheduler
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

MS = 1_000_000


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


def config(policy="slo-aware", **overrides):
    values = {f.name: f.default for f in fields(Config)}
    values.update(model="unused", scheduler_policy=policy, num_kvcache_blocks=64,
                  max_model_len=2048, max_num_seqs=4, max_num_batched_tokens=512,
                  eos=999, scheduler_prefill_chunk=128, scheduler_min_prefill_chunk=64)
    values.update(overrides)
    return SimpleNamespace(**values)


def seq(n=128, output=4, token=1, eos=False):
    return Sequence([token]*n, SamplingParams(max_tokens=output, ignore_eos=not eos))


def drain(scheduler, clock, max_steps=10000):
    trace = []
    while not scheduler.is_finished():
        selected, prefill = scheduler.schedule()
        trace.append((prefill, [s.seq_id for s in selected], [s.num_scheduled_tokens for s in selected]))
        clock.now += (10*MS + sum(s.num_scheduled_tokens for s in selected)*100_000) if prefill else 20*MS
        scheduler.postprocess(selected, [7]*len(selected), prefill)
        if len(trace) > max_steps:
            raise AssertionError("Workload did not drain")
    assert not scheduler.block_manager.used_block_ids
    assert not getattr(scheduler, "timings", {})
    return trace


class PolicyTests(unittest.TestCase):
    def test_original_factory_is_exact_class(self):
        def forbidden_clock():
            raise AssertionError("original path must not use policy clock")
        self.assertIs(type(make_scheduler(config("original"), forbidden_clock)), Scheduler)

    def test_original_trace_parity(self):
        traces, outputs = [], []
        for factory in [Scheduler, make_scheduler]:
            scheduler = factory(config("original", num_kvcache_blocks=16))
            requests = [seq(n, 4, token=i+1) for i,n in enumerate([255,256,257,511,512,16])]
            ids = {s.seq_id:i for i,s in enumerate(requests)}
            for s in requests:
                scheduler.add(s)
            trace = drain(scheduler, Clock())
            traces.append([(p,[ids[i] for i in selected],n) for p,selected,n in trace])
            outputs.append([s.completion_token_ids for s in requests])
        self.assertEqual(traces[0],traces[1])
        self.assertEqual(outputs[0],outputs[1])

    def test_original_preemption_parity(self):
        traces = []
        for factory in [Scheduler, make_scheduler]:
            scheduler = factory(config("original", num_kvcache_blocks=6))
            requests = [seq(255, 300, token=i+1) for i in range(3)]
            ids = {s.seq_id: i for i, s in enumerate(requests)}
            allocations = []
            allocate = scheduler.block_manager.allocate

            def counted_allocate(request, cached):
                allocations.append(request.seq_id)
                return allocate(request, cached)

            scheduler.block_manager.allocate = counted_allocate
            for request in requests:
                scheduler.add(request)
            trace = drain(scheduler, Clock())
            self.assertGreater(len(allocations), len(requests))
            self.assertTrue(all(s.completion_token_ids == [7]*300 for s in requests))
            traces.append([(p, [ids[i] for i in selected], n) for p, selected, n in trace])
        self.assertEqual(traces[0], traces[1])

    def test_chunk_boundaries_and_eos(self):
        for policy in ["static", "slo-aware"]:
            for length in [1,255,256,257,511,512,513]:
                clock=Clock(); scheduler=make_scheduler(config(policy),clock)
                request=seq(length,4)
                scheduler.add(request)
                drain(scheduler,clock)
                self.assertEqual(request.completion_token_ids,[7]*4)
                self.assertFalse(request.block_table)
            scheduler=make_scheduler(config(policy),Clock()); request=seq(16,eos=True)
            scheduler.add(request)
            batch,prefill=scheduler.schedule(); scheduler.postprocess(batch,[999],prefill)
            self.assertTrue(request.is_finished)
            self.assertFalse(scheduler.block_manager.used_block_ids)

    def test_discarded_chunk_does_not_set_token_deadline(self):
        clock=Clock(); scheduler=make_scheduler(config("static"),clock)
        request=seq(257); scheduler.add(request)
        batch,prefill=scheduler.schedule(); clock.now+=MS
        scheduler.postprocess(batch,[123],prefill)
        self.assertEqual(request.num_completion_tokens,0)
        self.assertIsNone(scheduler.timings[request.seq_id].last_token_ns)
        drain(scheduler,clock)

    def test_decode_round_robin_and_edf(self):
        for policy in ["static","slo-aware"]:
            clock=Clock(); scheduler=make_scheduler(config(policy,max_num_seqs=2),clock)
            requests=[seq(4,8,token=i+1) for i in range(6)]
            for s in requests: scheduler.add(s)
            while scheduler.waiting:
                scheduler.prefill_streak=0
                scheduler.decode_streak=scheduler.config.scheduler_max_decode_steps
                batch,prefill=scheduler.schedule()
                self.assertTrue(prefill)
                clock.now+=10*MS
                scheduler.postprocess(batch,[7]*len(batch),prefill)
            trace=drain(scheduler,clock)
            decodes=[ids for p,ids,_ in trace if not p]
            first_three=set(i for ids in decodes[:3] for i in ids)
            self.assertEqual(len(first_three),6)

    def test_low_memory_admission_and_progress(self):
        for policy in ["static","slo-aware"]:
            clock=Clock(); scheduler=make_scheduler(config(policy,num_kvcache_blocks=4),clock)
            requests=[seq(257,260,token=i+1) for i in range(3)]
            for s in requests: scheduler.add(s)
            drain(scheduler,clock)
            self.assertTrue(all(s.num_completion_tokens==260 for s in requests))

    def test_oversized_request_rejected_without_leak(self):
        scheduler=make_scheduler(config(num_kvcache_blocks=1))
        with self.assertRaises(ValueError): scheduler.add(seq(257))
        self.assertTrue(scheduler.is_finished())
        self.assertFalse(scheduler.timings)

    def test_arrival_timestamp_and_serialization(self):
        clock=Clock();clock.now=100*MS
        scheduler=make_scheduler(config(),clock);request=seq()
        request.arrival_time_ns=20*MS;scheduler.add(request)
        self.assertEqual(scheduler.timings[request.seq_id].arrival_ns,20*MS)
        copy=pickle.loads(pickle.dumps(request))
        self.assertFalse(hasattr(copy,"arrival_time_ns"))
        self.assertEqual(copy.token_ids,request.token_ids)

    def test_future_timestamp_rejected(self):
        scheduler=make_scheduler(config(),Clock());request=seq();request.arrival_time_ns=1
        with self.assertRaises(ValueError):scheduler.add(request)

    def ready(self):
        clock=Clock(); scheduler=make_scheduler(config(),clock)
        request=seq(1,64);scheduler.add(request)
        batch,prefill=scheduler.schedule();clock.now+=20*MS
        scheduler.postprocess(batch,[7],prefill)
        scheduler.decode_ns=20*MS;scheduler.prefill_ns_per_token=100_000
        scheduler.add(seq(512,4,token=2))
        return scheduler,clock

    def test_dynamic_budget_spends_only_slack(self):
        scheduler,clock=self.ready()
        prefill,budget,*_=scheduler._choose(clock.now)
        self.assertTrue(prefill)
        self.assertLessEqual(scheduler._prefill_cost(budget)+scheduler.decode_ns,scheduler.tpot_ns)
        clock.now+=90*MS
        self.assertFalse(scheduler._choose(clock.now)[0])

    def test_budget_charges_whole_batch_not_only_head(self):
        scheduler, clock = self.ready()
        scheduler.waiting[0].num_cached_tokens = 511
        scheduler.prefill_ns_per_token = 300_000
        _, budget, *_ = scheduler._choose(clock.now)
        self.assertEqual(budget, 256)
        self.assertLessEqual(scheduler._prefill_cost(budget), 80*MS)

    def test_expired_ttft_can_override_decode(self):
        scheduler, clock = self.ready()
        waiting = scheduler.waiting[0]
        scheduler.timings[waiting.seq_id].arrival_ns = -10_000*MS
        clock.now += 90*MS
        decision = scheduler._choose(clock.now)
        self.assertTrue(decision[0])
        self.assertEqual(decision[1], scheduler.config.scheduler_min_prefill_chunk)
        self.assertEqual(decision[2], "ttft-pressure")

    def test_class_starvation_guards(self):
        scheduler,clock=self.ready()
        scheduler.decode_streak=scheduler.config.scheduler_max_decode_steps
        clock.now+=500*MS
        self.assertEqual(scheduler._choose(clock.now)[2],"prefill-fairness")
        scheduler.prefill_streak=scheduler.config.scheduler_max_prefill_steps
        self.assertEqual(scheduler._choose(clock.now)[2],"decode-fairness")

    def test_reservation_not_exhausted_by_new_prompt(self):
        clock=Clock(); scheduler=make_scheduler(config(num_kvcache_blocks=4),clock)
        first=seq(255,300);second=seq(257,4,token=2)
        scheduler.add(first);scheduler.add(second)
        trace=drain(scheduler,clock)
        self.assertEqual(first.num_completion_tokens,300)
        self.assertEqual(second.num_completion_tokens,4)

    def test_prefix_cache_still_supported(self):
        clock=Clock();scheduler=make_scheduler(config("static"),clock)
        scheduler.add(seq(257));drain(scheduler,clock)
        request=seq(257);scheduler.add(request)
        batch,prefill=scheduler.schedule()
        self.assertEqual(request.num_cached_tokens,256)
        scheduler.postprocess(batch,[7],prefill);drain(scheduler,clock)

    def test_randomized_finite_workloads_drain(self):
        for policy in ["static","slo-aware"]:
            for seed in range(5):
                rng=random.Random(seed);clock=Clock()
                scheduler=make_scheduler(config(policy,num_kvcache_blocks=12),clock)
                requests=[seq(rng.randint(1,700),rng.randint(1,100),token=i+1) for i in range(12)]
                for s in requests:scheduler.add(s)
                drain(scheduler,clock)
                self.assertTrue(all(s.is_finished for s in requests))


if __name__=="__main__":
    unittest.main()
