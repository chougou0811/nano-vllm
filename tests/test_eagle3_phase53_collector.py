import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.serving.eagle3_phase53_serving import acceptance_check, Collector, base


class Phase53CollectorTests(unittest.TestCase):
    def test_audit_wrapper_does_not_rebind_draft_callable(self):
        executor = SimpleNamespace(propose=lambda items:[[1]],
            last_metrics=dict(conditioning_ns=10,draft_forwards=2))
        collector = Collector.__new__(Collector)
        collector.current, collector.originals = {}, []
        collector.runner = SimpleNamespace(call=lambda *args:'rpc-result')
        coordinator = SimpleNamespace(draft_batch_executor=executor,drafts={})
        collector.engine = SimpleNamespace(speculative_coordinator=coordinator,phase53_audit=True)
        with patch.object(base.TimingCollector,'_install',lambda self:None):
            collector._install()
        item = SimpleNamespace(state=SimpleNamespace(last_metrics=dict(draft_tokens_processed=3)))
        self.assertEqual(executor.propose([item]),[[1]])
        self.assertEqual(collector.runner.call('unrelated'),'rpc-result')
        self.assertEqual(collector.current['draft_forward_count'],2)

    def test_live_tuple_and_json_list_commits(self):
        for kind in (tuple,list):
            run = dict(requests=[dict(sequence_id=7,output_limit=4)],steps=[dict(
                selected_sequence_ids=[7],commits=[dict(proposed_token_ids=kind([3,4,5]),
                    target_token_ids=kind([3,4,6,7]),accepted_length=2,
                    committed_token_ids=kind([3,4,6]),fallback_token=6,finished=True,
                    old_cursor=12,new_cursor=15)])])
            self.assertTrue(acceptance_check(run,99))


if __name__=='__main__':
    unittest.main()
