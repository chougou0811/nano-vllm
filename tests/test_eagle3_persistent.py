import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from nanovllm.speculative.draft_state import DraftState


class FeatureModel:
    """Exact causal toy: KV depends on both target features and shifted tokens."""
    d2t = torch.zeros(4, dtype=torch.long)
    def reset(self): pass
    def reset_kv(self): pass
    def norm(self, x): return x
    def lm_head(self, x): return torch.cat([x, x+1, x+2, x+3], -1)
    def __call__(self, features, input_ids, past_key_values=None, use_cache=True):
        rows = features[..., :1] + input_ids[..., None]
        kv = rows[:, None]
        if past_key_values is not None:
            kv = torch.cat((past_key_values[0][0], kv), 2)
        return rows + 17, ((kv, kv.clone()),)


class PersistentTests(unittest.TestCase):
    def draft(self):
        return SimpleNamespace(model=FeatureModel(), device=torch.device('cpu'))

    def test_acceptance_patterns_and_boundary(self):
        for length in (3, 255, 256, 257, 1024):
            for accepts in ([0]*8, [1]*8, [3]*8, [0, 3, 1, 2]*2):
                p = DraftState(self.draft(), 'request')
                f = DraftState(self.draft(), 'request', 'full_rebuild')
                tokens = [1]*(length+1)
                features = torch.arange(length).float()[:, None]
                for iteration,a in enumerate(accepts):
                    k=(1,6,1)[iteration%3] if accepts==[0]*8 else 3
                    previous = p.past[0][0].clone() if p.past else None
                    self.assertEqual(p.propose(features,tokens,k,owner='request'),
                                     f.propose(features,tokens,k,owner='request'))
                    _, reference = f.draft.model(features[None], torch.tensor(tokens[1:])[None])
                    self.assertTrue(torch.equal(p.past[0][0],reference[0][0]))
                    if previous is not None:
                        self.assertTrue(torch.equal(previous,p.past[0][0][:,:,:previous.shape[2]]))
                    self.assertEqual(p.last_metrics['rollback_tokens'],k-1)
                    features = torch.cat((features,torch.full((1+a,1),99.)))
                    tokens.extend([3]*a+[2])
                    p.committed(len(features),tokens,owner='request')
                p.close()
                self.assertIsNone(p.past)
                self.assertEqual(p.cursor,0)

    def test_owner_changed_prefix_and_cleanup(self):
        p = DraftState(self.draft(),1)
        with self.assertRaises(RuntimeError): p.propose(torch.ones(2,1),[1,2,3],3,owner=2)
        p.propose(torch.ones(2,1),[1,2,3],3,owner=1)
        with self.assertRaises(RuntimeError): p.committed(3,[1,0,3,2],owner=1)
        with self.assertRaises(ValueError): p.propose(torch.ones(2,1),[1,2,3],3,owner=1)
        p.close()
        p.close()
        with self.assertRaises(RuntimeError): p.propose(torch.ones(2,1),[1,2,3],3,owner=1)

    def test_reference_fp32_incremental_cache(self):
        reference=Path('/root/autodl-tmp/references/eagle-pinned')
        if not reference.exists():
            self.skipTest('Pinned EAGLE reference not installed')
        sys.path.insert(0,str(reference))
        from eagle.model.cnets import Model
        from eagle.model.configs import EConfig
        torch.manual_seed(2026)
        config=EConfig(hidden_size=32,intermediate_size=64,num_attention_heads=4,
                       num_key_value_heads=2,num_hidden_layers=1,vocab_size=32,
                       draft_vocab_size=16,head_dim=8,max_position_embeddings=2048)
        model=Model(config,load_emb=False,bias=False).float().eval()
        draft=SimpleNamespace(model=model,device=torch.device('cpu'))
        p=DraftState(draft,1)
        features=torch.randn(257,96)
        tokens=torch.randint(0,32,(258,)).tolist()
        with torch.inference_mode():
            for k,a in ((1,0),(6,6),(1,1),(6,2),(1,0),(6,0)):
                proposals=p.propose(features,tokens,k,owner=1)
                f=DraftState(draft,1,'full_rebuild')
                self.assertEqual(proposals,f.propose(features,tokens,k,owner=1))
                model.reset()
                _,expected=model(features[None],input_ids=torch.tensor(tokens[1:])[None],use_cache=True)
                for actual,reference_tensor in zip(p.past[0],expected[0]):
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual,reference_tensor,rtol=1e-5,atol=1e-6)
                features=torch.cat((features,torch.randn(1+a,96)))
                tokens.extend(torch.randint(0,32,(1+a,)).tolist())
                p.committed(len(features),tokens,owner=1)
        p.close()


if __name__ == '__main__':
    unittest.main()
