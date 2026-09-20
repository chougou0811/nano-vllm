import ast
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from nanovllm.speculative.acceptance import accept_greedy
from nanovllm.speculative.session import check_ranks
from nanovllm.speculative.runtime import _zero
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class EagleTests(unittest.TestCase):
    def test_reference_greedy_prefix_semantics(self):
        root = Path('/root/autodl-tmp/references/eagle-pinned')
        if not root.exists():
            self.skipTest('Pinned external reference unavailable')
        sys.path.insert(0,str(root))
        from eagle.model.utils import evaluate_posterior
        for k in range(1,7):
            proposals = [1]*k
            for accepted in range(k+1):
                target = [1]*accepted+[2]*(k+1-accepted)
                logits = torch.zeros(1,k+1,4)
                logits[0,torch.arange(k+1),torch.tensor(target)] = 10
                candidates = torch.tensor([[3]+proposals])
                _,length,next_logits = evaluate_posterior(logits,candidates,None)
                ours = accept_greedy(proposals,target,k+2)
                self.assertEqual(ours.matched,int(length))
                self.assertEqual(ours.accepted,accepted)
                self.assertEqual(ours.fallback,int(next_logits.argmax()))
                self.assertEqual(ours.tokens,[1]*accepted+[2])

    def test_eos_and_output_limit(self):
        r = accept_greedy([1,2,3],[1,2,3,4],8,eos=2)
        self.assertEqual(r.tokens,[1,2])
        self.assertEqual(r.accepted,2)
        self.assertIsNone(r.fallback)
        self.assertTrue(r.finished)
        r = accept_greedy([1,2],[1,2,3],1)
        self.assertEqual(r.tokens,[1])
        self.assertEqual(r.accepted,1)
        self.assertTrue(r.finished)
        r = accept_greedy([1],[2,3],8,eos=2)
        self.assertTrue(r.finished)
        self.assertEqual(r.accepted,0)
        self.assertEqual(r.fallback,2)
        r = accept_greedy([], [9],1)
        self.assertEqual(r.tokens,[9])
        self.assertTrue(r.finished)

    def test_invalid_acceptance_input(self):
        for p,t,r in [([1],[2],1),([],[1],0)]:
            with self.assertRaises(ValueError):
                accept_greedy(p,t,r)

    def test_rank_audit_rejects_drift_and_dirty_suffix(self):
        check_ranks([[0,4,5,4,77,1],[1,4,5,4,77,1]])
        for rows in [[],[[0,4,5,4,77,0]],[[0,4,5,4,77,1],[1,3,5,4,77,1]]]:
            with self.assertRaises(RuntimeError):
                check_ranks(rows)

    def test_suffix_zero_uses_physical_slot_mapping(self):
        cache = torch.ones(2,2,4,256,1,2)
        runner = SimpleNamespace(kv_cache=cache,block_size=256)
        self.assertTrue(_zero(runner,dict(blocks=[2,0]),255,258))
        self.assertEqual(torch.count_nonzero(cache[:,:,2,255]).item(),0)
        self.assertEqual(torch.count_nonzero(cache[:,:,0,:2]).item(),0)
        self.assertTrue(torch.all(cache[:,:,1] == 1))
        self.assertTrue(torch.all(cache[:,:,2,:255] == 1))
        self.assertTrue(torch.all(cache[:,:,0,2:] == 1))

    def test_private_lease_does_not_publish_prefix(self):
        bm = BlockManager(4,256)
        for _ in range(20):
            lease = Sequence([0]*513)
            bm.allocate(lease,0)
            self.assertEqual(len(bm.used_block_ids),3)
            self.assertFalse(bm.hash_to_block_id)
            bm.deallocate(lease)
            self.assertFalse(bm.used_block_ids)
            self.assertEqual(len(bm.free_block_ids),4)
            self.assertTrue(all(b.ref_count == 0 for b in bm.blocks))

    def test_benchmark_does_not_pass_numerical_divergence(self):
        from benchmarks.serving.eagle3_phase1 import require_parity
        result = dict(all_equal=True,off_parity=True,forced_rejection_equal=True)
        require_parity(result)
        for key in result:
            with self.assertRaisesRegex(RuntimeError,'parity failed'):
                require_parity(dict(result,**{key:False}))

    def test_original_paths_and_scheduler_frozen(self):
        root = Path(__file__).resolve().parents[1]
        def saved(path):
            return subprocess.check_output(['git','show',f'6695617:{path}'],cwd=root,text=True)
        for path in ['nanovllm/engine/scheduler.py','nanovllm/engine/policy_scheduler.py',
                     'nanovllm/engine/progress_scheduler.py','nanovllm/config.py',
                     'nanovllm/engine/block_manager.py','nanovllm/engine/sequence.py',
                     'nanovllm/layers/sampler.py','nanovllm/layers/attention.py','nanovllm/models/qwen3.py']:
            self.assertEqual(saved(path),(root/path).read_text())
        for path in ['nanovllm/engine/model_runner.py','nanovllm/engine/llm_engine.py']:
            def methods(code):
                return {n.name:ast.dump(n) for c in ast.parse(code).body if isinstance(c,ast.ClassDef)
                        for n in c.body if isinstance(n,ast.FunctionDef)}
            old,new = methods(saved(path)),methods((root/path).read_text())
            self.assertTrue(all(new[k] == v for k,v in old.items()))


if __name__ == '__main__':
    unittest.main()
