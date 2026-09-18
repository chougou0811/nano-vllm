import csv
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.serving.adapter import ObservedScheduler, RequestRecord, RunResult, run_workload
from benchmarks.serving.metrics import distribution, request_metrics, summarize
from benchmarks.serving.report import write_artifacts
from benchmarks.serving.workload import RequestSpec, deterministic_workload, IsolatedWorkloads


MS = 1_000_000


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class Sequence:
    def __init__(self, sid, prompt, params):
        self.seq_id = sid
        self.tokens = list(prompt)
        self.num_prompt_tokens = len(prompt)
        self.max_tokens = params.max_tokens
        self.ignore_eos = True
        self.is_finished = False
        self.num_scheduled_tokens = len(prompt)

    @property
    def num_completion_tokens(self):
        return len(self.tokens) - self.num_prompt_tokens

    def __getitem__(self, item):
        return self.tokens[item]


class Scheduler:
    eos = -1

    def __init__(self):
        self.active = []
        self.commit_count = 1

    def add(self, seq):
        self.active.append(seq)

    def schedule(self):
        seqs = list(self.active)
        prefill = any(s.num_completion_tokens == 0 for s in seqs)
        for seq in seqs:
            seq.num_scheduled_tokens = seq.num_prompt_tokens if prefill else 1
        self.last_value = (seqs, prefill)
        return self.last_value

    def postprocess(self, seqs, proposals, is_prefill):
        for seq in seqs:
            seq.tokens.extend([7] * min(self.commit_count, seq.max_tokens - seq.num_completion_tokens))
            if seq.num_completion_tokens == seq.max_tokens:
                seq.is_finished = True
                self.active.remove(seq)


class Engine:
    def __init__(self, clock):
        self.scheduler = Scheduler()
        self.clock = clock
        self.next_id = 0

    def is_finished(self):
        return not self.scheduler.active

    def add_request(self, prompt, params):
        self.scheduler.add(Sequence(self.next_id, prompt, params))
        self.next_id += 1

    def step(self):
        seqs, prefill = self.scheduler.schedule()
        self.clock.now += 10 * MS
        self.scheduler.postprocess(seqs, [999] * len(seqs), prefill)
        return [(s.seq_id, s[s.num_prompt_tokens:]) for s in seqs if s.is_finished], 0


def params(spec):
    return SimpleNamespace(max_tokens=spec.output_length)


class MetricsTests(unittest.TestCase):
    def record(self):
        return RequestRecord(0, 4, 3, 0, admitted_ns=2*MS, first_scheduled_ns=5*MS,
                             first_token_ns=10*MS, finish_ns=75*MS,
                             token_times_ns=[10*MS, 30*MS, 70*MS],
                             output_token_ids=[1, 2, 3], status="completed")

    def test_metrics(self):
        row = request_metrics(self.record())
        for name, value in dict(queue_delay_ms=5, ingress_queue_delay_ms=2,
                                scheduler_queue_delay_ms=3, ttft_ms=10, tpot_ms=30, e2e_ms=75).items():
            self.assertEqual(row[name], value)
        self.assertEqual(row["itl_ms"], [20, 40])
        self.assertIsNone(row["slo_met"])

    def test_single_token_and_simultaneous_commits(self):
        record = self.record()
        record.token_times_ns = [10*MS]
        record.output_token_ids = [1]
        self.assertIsNone(request_metrics(record)["tpot_ms"])
        self.assertTrue(request_metrics(record, tpot_slo_ms=1)["slo_met"])
        record.token_times_ns *= 2
        record.output_token_ids *= 2
        self.assertEqual(request_metrics(record)["itl_ms"], [0])

    def test_percentiles(self):
        self.assertEqual(distribution([0, 100])["p95"], 95)
        self.assertIsNone(distribution([])["p50"])

    def test_token_vs_request_weighting(self):
        first, second = self.record(), self.record()
        second.request_id = 1
        second.token_times_ns = [10*MS, 110*MS]
        second.output_token_ids = [1, 2]
        second.finish_ns = 110*MS
        summary, _ = summarize(RunResult([first, second], window_end_ns=110*MS, complete=True))
        self.assertEqual(summary["latency"]["tpot_ms"]["mean"], 65)
        self.assertAlmostEqual(summary["latency"]["itl_ms"]["mean"], 160/3)
        self.assertEqual(summary["output_tokens"], 5)
        self.assertAlmostEqual(summary["request_throughput_rps"], 2/.11)

    def test_slo_failures_and_future_arrivals(self):
        records = [self.record(), RequestRecord(1, 4, 3, 0, status="timeout"),
                   RequestRecord(2, 4, 3, 200*MS, status="not_arrived")]
        summary, _ = summarize(RunResult(records, window_end_ns=100*MS), 20, 40)
        self.assertEqual(summary["arrived_requests"], 2)
        self.assertEqual(summary["slo_violation_rate"], .5)
        self.assertEqual(summary["slo_goodput_rps"], 10)

    def test_artifact_roundtrip(self):
        result = RunResult([self.record()], window_end_ns=75*MS, complete=True)
        summary, rows = summarize(result)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_artifacts(path, result, summary, rows, {})
            self.assertEqual(json.loads((path / "requests.json").read_text())[0]["token_times_ns"],
                             self.record().token_times_ns)
            with (path / "tokens.csv").open() as stream:
                tokens = list(csv.DictReader(stream))
            self.assertEqual(len(tokens), 3)
            self.assertEqual(tokens[1]["itl_ns"], str(20*MS))
            self.assertIn("queue_delay_ms", (path / "report.md").read_text())


class AdapterTests(unittest.TestCase):
    def test_concurrency_does_not_move_arrivals(self):
        clock = Clock()
        engine = Engine(clock)
        original = engine.scheduler
        specs = [RequestSpec(0, [1], 2, 0), RequestSpec(1, [2], 2, 5*MS)]
        result = run_workload(engine, specs, params, 1, 100*MS, clock, clock.sleep)
        self.assertTrue(result.complete)
        self.assertIs(engine.scheduler, original)
        second = result.records[1]
        self.assertEqual(second.arrival_ns, 5*MS)
        self.assertEqual(second.admitted_ns, 20*MS)
        self.assertEqual(request_metrics(second)["queue_delay_ms"], 15)
        self.assertEqual((second.num_schedule_steps, second.num_prefill_steps, second.num_decode_steps), (2, 1, 1))

    def test_discards_proposals_and_records_multi_commit(self):
        clock = Clock()
        original = Scheduler()
        record = RequestRecord(0, 1, 2, 0)
        result = RunResult([record])
        proxy = ObservedScheduler(original, result, 0, clock)
        proxy.current_request = record
        seq = Sequence(9, [1], SimpleNamespace(max_tokens=2))
        proxy.add(seq)
        original.commit_count = 0
        value = proxy.schedule()
        self.assertIs(value, original.last_value)
        proxy.postprocess(*value[:1], [999], value[1])
        self.assertEqual(record.token_times_ns, [])
        original.commit_count = 2
        clock.now = 10*MS
        seqs, prefill = proxy.schedule()
        proxy.postprocess(seqs, [999, 888, 777], prefill)
        self.assertEqual(record.output_token_ids, [7, 7])
        self.assertEqual(record.token_times_ns, [10*MS, 10*MS])
        self.assertEqual(record.num_prefill_steps, 2)

    def test_idle_arrival_and_timeout(self):
        clock = Clock()
        engine = Engine(clock)
        result = run_workload(engine, [RequestSpec(0, [1], 1, 50*MS)], params, 1, 100*MS, clock, clock.sleep)
        self.assertEqual(result.records[0].admitted_ns, 50*MS)
        clock = Clock()
        result = run_workload(Engine(clock), [RequestSpec(0, [1], 1, 50*MS)], params, 1, 20*MS, clock, clock.sleep)
        self.assertFalse(result.complete)
        self.assertEqual(result.records[0].status, "not_arrived")

    def test_failure_restores_original(self):
        clock = Clock()
        engine = Engine(clock)
        original = engine.scheduler
        def fail():
            raise RuntimeError("test failure")
        engine.step = fail
        result = run_workload(engine, [RequestSpec(0, [1], 1, 0)], params, 1, 20*MS, clock, clock.sleep)
        self.assertIs(engine.scheduler, original)
        self.assertEqual(result.records[0].status, "failed")
        self.assertIn("test failure", result.error)

    def test_batch(self):
        clock = Clock()
        specs = [RequestSpec(i, [i], 3, 0) for i in range(8)]
        result = run_workload(Engine(clock), specs, params, 4, 100*MS, clock, clock.sleep)
        summary, _ = summarize(result)
        self.assertEqual(summary["completed_requests"], 8)
        self.assertEqual(summary["max_decode_batch"], 4)
        self.assertEqual(summary["latency"]["itl_ms"]["count"], 16)


class WorkloadTests(unittest.TestCase):
    def test_reproducibility_and_rng_isolation(self):
        before = random.getstate()
        first = deterministic_workload(range(100), 8, 4, 3, 100, 42)
        self.assertEqual(first, deterministic_workload(range(100), 8, 4, 3, 100, 42))
        self.assertEqual(before, random.getstate())
        self.assertEqual([s.arrival_ns for s in first], list(range(0, 800, 100)))

    def test_cli_invalid_values(self):
        from benchmarks.serving.__main__ import parse_args
        base = ["--model", "/tmp/model", "--model-revision", "a"*40, "--output-dir", "/tmp/out"]
        for extra in [["--arrival-interval-ms", "nan"], ["--concurrency", "0"],
                      ["--prompt-length", "2048"], ["--ttft-slo-ms", "-1"],
                      ["--scheduler-prefill-chunk", "0"], ["--scheduler-min-prefill-chunk", "-1"],
                      ["--scheduler-ttft-ms", "nan"], ["--scheduler-tpot-ms", "inf"]]:
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                parse_args(base + extra)

    def test_isolated_workloads(self):
        factory = IsolatedWorkloads(range(1, 100))
        runs = [factory.build(8, 257, 4, 100, 42) for _ in range(4)]
        first = [s.prompt_token_ids[0] for run in runs for s in run]
        self.assertEqual(len(first), len(set(first)))
        for run in runs:
            self.assertEqual([len(s.prompt_token_ids) for s in run], [257]*8)
            self.assertEqual([s.arrival_ns for s in run], list(range(0, 800, 100)))

    def test_isolation_exhaustion(self):
        factory = IsolatedWorkloads([1, 2])
        factory.build(2, 3, 2, 0, 42)
        with self.assertRaises(ValueError):
            factory.build(1, 3, 2, 0, 42)

    def test_warmup_plan(self):
        from benchmarks.serving.warmup import warmup_plan
        args = SimpleNamespace(warmup_mode="decode-shapes", prompt_length=257,
                               output_length=32, max_num_seqs=4, seed=42)
        plan = warmup_plan(IsolatedWorkloads(range(1, 100)), args)
        self.assertEqual([len(specs) for _, specs in plan], [1, 2, 3, 4])
        self.assertTrue(all(s.output_length == 32 for _, specs in plan for s in specs))


class RealSchedulerTests(unittest.TestCase):
    """Actual production scheduling and block management, with CPU-only token commits."""

    def engine(self, clock, token_budget=1024):
        from nanovllm.engine.scheduler import Scheduler as ActualScheduler
        from nanovllm.engine.sequence import Sequence as ActualSequence
        from nanovllm.sampling_params import SamplingParams

        class ActualEngine(Engine):
            def add_request(self, prompt, params):
                self.scheduler.add(ActualSequence(prompt, SamplingParams(max_tokens=params.max_tokens, ignore_eos=True)))

        engine = ActualEngine(clock)
        engine.scheduler = ActualScheduler(SimpleNamespace(max_num_seqs=4,
            max_num_batched_tokens=token_budget, eos=-1, kvcache_block_size=256, num_kvcache_blocks=64))
        engine.is_finished = lambda: engine.scheduler.is_finished()
        return engine

    def run_specs(self, engine, specs, mode="open-loop", concurrency=4):
        return run_workload(engine, specs, params, concurrency, 10_000*MS,
                            engine.clock, engine.clock.sleep, arrival_mode=mode)

    def test_open_loop_queues_inside_engine(self):
        specs = [RequestSpec(i, [i+1]*8, 3, 0) for i in range(12)]
        result = self.run_specs(self.engine(Clock()), specs, concurrency=1)
        self.assertTrue(result.complete, result.error)
        self.assertEqual(result.steps[0]["queue_before"]["waiting"], 12)
        self.assertEqual(result.steps[0]["queue_after_schedule"]["waiting"], 8)
        self.assertTrue(all(r.admitted_ns == 0 for r in result.records))
        self.assertGreater(result.records[-1].first_scheduled_ns, 0)

    def test_closed_retains_gate(self):
        specs = [RequestSpec(i, [i+1]*8, 3, 0) for i in range(8)]
        result = self.run_specs(self.engine(Clock()), specs, "concurrency-gated", 1)
        self.assertTrue(result.complete, result.error)
        self.assertEqual(max(s["queue_before"]["waiting"] for s in result.steps), 1)
        self.assertEqual(result.records[1].admitted_ns, 30*MS)

    def test_step_boundary_lag_not_engine_wait(self):
        specs = [RequestSpec(0, [1]*8, 3, 0), RequestSpec(1, [2]*8, 3, 5*MS)]
        result = self.run_specs(self.engine(Clock()), specs, concurrency=1)
        row = request_metrics(result.records[1])
        self.assertEqual(result.records[1].admitted_ns, 10*MS)
        self.assertEqual(row["ingress_queue_delay_ms"], 5)
        self.assertEqual(row["engine_queue_delay_ms"], 0)

    def test_positive_prefix_hit_control(self):
        engine = self.engine(Clock())
        original = engine.scheduler.block_manager
        specs = [RequestSpec(0, [3]*257, 2, 0)]
        first = self.run_specs(engine, specs)
        second = self.run_specs(engine, specs)
        self.assertTrue(first.complete and second.complete)
        self.assertEqual(first.records[0].prefix_cache_blocks, 0)
        self.assertEqual(second.records[0].initial_prefix_cache_blocks, 1)
        self.assertIs(engine.scheduler.block_manager, original)
        self.assertFalse(original.used_block_ids)

    def test_isolation_with_cacheable_prompts(self):
        factory = IsolatedWorkloads(range(1, 100))
        engine = self.engine(Clock())
        for _ in range(3):
            result = self.run_specs(engine, factory.build(4, 257, 4, 0, 42))
            self.assertTrue(result.complete, result.error)
            self.assertEqual(sum(r.prefix_cache_blocks for r in result.records), 0)
            self.assertEqual(sum(r.num_kv_allocations for r in result.records), 4)

    def test_chunked_prefill_commits_only_valid_tokens(self):
        result = self.run_specs(self.engine(Clock(), 128), [RequestSpec(0, [1]*257, 3, 0)])
        record = result.records[0]
        self.assertTrue(result.complete, result.error)
        self.assertEqual(record.num_prefill_steps, 3)
        self.assertEqual(record.num_decode_steps, 2)
        self.assertEqual(record.token_times_ns, [30*MS, 40*MS, 50*MS])

    def test_submission_breakdown(self):
        clock = Clock()
        engine = self.engine(clock)
        original_add = engine.add_request
        def delayed_add(prompt, params):
            clock.now += MS
            original_add(prompt, params)
        engine.add_request = delayed_add
        record = self.run_specs(engine, [RequestSpec(0, [1], 1, 0)]).records[0]
        metrics = request_metrics(record)
        self.assertEqual(metrics["submission_ms"], 1)
        self.assertEqual(metrics["engine_queue_delay_ms"], 0)
        self.assertEqual(metrics["scheduler_queue_delay_ms"], 1)


if __name__ == "__main__":
    unittest.main()
