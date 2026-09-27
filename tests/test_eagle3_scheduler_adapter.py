from collections import deque
from types import SimpleNamespace
import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.speculative.batch import SpeculativeCommit
from nanovllm.speculative.scheduler_adapter import SpeculativeSchedulerAdapter


def config(blocks=16, max_seqs=4):
    return SimpleNamespace(max_num_seqs=max_seqs,max_num_batched_tokens=64,
                           eos=99,kvcache_block_size=256,num_kvcache_blocks=blocks)


class FakeCoordinator:
    def __init__(self):
        self.live = set()
        self.preempted = []
        self.closed = []
        self.requests = {}

    def owns(self, seq_id):
        return seq_id in self.live

    def register(self, seq):
        self.live.add(seq.seq_id)
        self.requests[seq.seq_id] = SimpleNamespace(fail=lambda error: None)

    def preempt(self, seq):
        self.preempted.append(seq.seq_id)

    def close_request(self, seq_id):
        self.closed.append(seq_id)
        self.live.discard(seq_id)
        self.requests.pop(seq_id,None)


class SchedulerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.original = Scheduler(config())
        self.coordinator = FakeCoordinator()
        self.scheduler = SpeculativeSchedulerAdapter(self.original,self.coordinator)

    def seq(self, length=16, output=8):
        return Sequence([1]*length,SamplingParams(max_tokens=output,ignore_eos=True))

    def test_active_concurrency_limit(self):
        scheduler = SpeculativeSchedulerAdapter(Scheduler(config(max_seqs=2)),FakeCoordinator())
        scheduler.add(self.seq()); scheduler.add(self.seq())
        with self.assertRaisesRegex(ValueError,"max_num_seqs"):
            scheduler.add(self.seq())

    def test_decode_defers_requests_above_target_token_budget(self):
        original = Scheduler(config(max_seqs=4))
        original.max_num_batched_tokens = 2
        scheduler = SpeculativeSchedulerAdapter(original, FakeCoordinator())
        requests = [self.seq() for _ in range(4)]
        for request in requests:
            scheduler.add(request)
            original.waiting.remove(request)
            original.running.append(request)
            request.status = SequenceStatus.RUNNING
            original.block_manager.allocate(request, 0)
        selected, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(selected, requests[:2])
        self.assertEqual([seq.num_scheduled_tokens for seq in requests], [1, 1, 0, 0])
        self.assertEqual(list(original.running), requests)

    def test_speculative_prefix_cache_disabled_ordinary_cache_unchanged(self):
        manager = BlockManager(8,256)
        seed = Sequence([7]*256)
        manager.allocate(seed,0); seed.num_scheduled_tokens=256
        manager.hash_blocks(seed); shared_id=seed.block_table[0]; manager.deallocate(seed)
        shared = Sequence([7]*257)
        self.assertEqual(manager.can_allocate(shared),1)

        original = Scheduler(config(blocks=8)); original.block_manager = manager
        coordinator = FakeCoordinator(); scheduler = SpeculativeSchedulerAdapter(original,coordinator)
        request = Sequence([7]*257,SamplingParams(max_tokens=2,ignore_eos=True))
        scheduler.add(request)
        selected,prefill = scheduler.schedule()
        self.assertTrue(prefill)
        self.assertEqual(selected,[request])
        self.assertEqual(request.num_cached_tokens,0)
        self.assertNotEqual(request.block_table[0],shared_id)

    def test_multi_token_postprocess_and_independent_finish(self):
        first,second = self.seq(output=3),self.seq(output=8)
        for seq in (first,second):
            self.scheduler.add(seq)
            self.original.waiting.remove(seq)
            self.original.running.append(seq)
            seq.status=SequenceStatus.RUNNING
            self.original.block_manager.allocate(seq,0)
            seq.num_cached_tokens=len(seq)-1
        commits = [
            SpeculativeCommit(first.seq_id,0,15,18,(2,3,4),(2,3,4,5),2,
                              (2,3,8),8,True,None,3,3),
            SpeculativeCommit(second.seq_id,0,15,16,(5,6,7),(9,8,7,6),0,
                              (9,),9,False,None,3,3),
        ]
        self.scheduler.postprocess_decode([first,second],commits)
        self.assertTrue(first.is_finished)
        self.assertFalse(first.block_table)
        self.assertNotIn(first,self.original.running)
        self.assertEqual(first.completion_token_ids,[2,3,8])
        self.assertFalse(second.is_finished)
        self.assertEqual(second.num_cached_tokens,16)
        self.assertEqual(second.completion_token_ids,[9])
        self.assertIn(second,self.original.running)

    def test_preemption_hook_runs_before_deallocation(self):
        request = self.seq(257)
        self.scheduler.add(request)
        self.original.block_manager.allocate(request,0)
        request.status=SequenceStatus.WAITING
        self.scheduler.block_manager.deallocate(request)
        self.assertEqual(self.coordinator.preempted,[request.seq_id])
        self.assertFalse(request.block_table)

    def test_abort_cleans_queues_and_blocks(self):
        requests=[self.seq(),self.seq()]
        for request in requests:
            self.scheduler.add(request)
            self.original.block_manager.allocate(request,0)
        self.scheduler.abort_all(RuntimeError("boom"))
        self.assertFalse(self.original.waiting)
        self.assertFalse(self.original.running)
        self.assertFalse(self.original.block_manager.original.used_block_ids)
        self.assertFalse(self.coordinator.live)
        self.assertTrue(all(request.is_finished for request in requests))


if __name__ == "__main__":
    unittest.main()
