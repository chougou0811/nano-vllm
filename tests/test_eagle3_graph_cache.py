from dataclasses import asdict
import unittest

from nanovllm.speculative.graph_policy import BoundedPolicy, GraphCacheConfig, agree, make_key
from benchmarks.serving.eagle3_graph_key import make_key as prototype_key


class GraphCachePolicyTests(unittest.TestCase):
    def key(self, batch=1, context=255, q=4):
        layout = dict(cu_seqlens_q=[i*q for i in range(batch+1)],
                      context_lens=[context]*batch,block_tables=[[1,2]]*batch)
        kwargs = dict(block_size=256,model_identity="frozen",dtype="bf16",
                      device_class="sm89",tp_size=2,training=True,feature_layers=(2,20,37))
        key = make_key(layout,**kwargs)
        old = prototype_key(layout,**kwargs)
        self.assertEqual(asdict(key) if key else None,asdict(old) if old else None)
        return key

    def test_exact_frozen_key_contract(self):
        for batch in (1,2,4):
            for context in (255,256,257,1024):
                self.key(batch,context)
        self.assertIsNone(self.key(3))
        self.assertIsNone(self.key(2,q=3))

    def test_second_occurrence_and_never(self):
        for policy in ("never","second"):
            p = BoundedPolicy(GraphCacheConfig(capture_policy=policy))
            key = self.key()
            p.observe(key)
            self.assertFalse(p.should_capture(key))
            p.observe(key)
            self.assertEqual(p.should_capture(key),policy=="second")

    def test_lru_capacity_and_capture_cap(self):
        p = BoundedPolicy(GraphCacheConfig(max_graph_entries=2,max_captures=2))
        keys = [self.key(context=x) for x in (255,256,257)]
        for key in keys[:2]:
            p.observe(key)
            p.observe(key)
            p.insert(key)
        p.touch(keys[0])
        self.assertEqual(p.victim(),keys[1])
        p.observe(keys[2])
        p.observe(keys[2])
        self.assertFalse(p.should_capture(keys[2]))
        with self.assertRaises(RuntimeError):
            p.insert(keys[2])

    def test_observation_metadata_bounded(self):
        p = BoundedPolicy(GraphCacheConfig(max_observed_keys=2))
        for context in range(10):
            p.observe(self.key(context=context))
        self.assertEqual(len(p.seen),2)

    def plan(self, **changes):
        return dict(key=self.key(),layout_digest="same",epoch=1,policy="second",
                    registry=[],available=False,capture=False,error=None)|changes

    def test_symmetric_miss_capture_hit(self):
        self.assertEqual(agree([self.plan()]*2),("eager","miss"))
        self.assertEqual(agree([self.plan(capture=True)]*2),("capture","second_occurrence"))
        self.assertEqual(agree([self.plan(available=True)]*2),("replay","hit"))

    def test_key_mismatch_or_rank_miss_is_eager(self):
        self.assertEqual(agree([self.plan(),self.plan(key=self.key(context=256))])[0],"eager")
        self.assertEqual(agree([self.plan(available=True),self.plan()]),("eager","rank_missing_key"))

    def test_unsafe_layout_generation_or_policy_fails_closed(self):
        for change in (dict(layout_digest="different"),dict(epoch=2),dict(policy="never"),dict(error="bad")):
            with self.assertRaises(RuntimeError):
                agree([self.plan(),self.plan(**change)])

    def test_invalid_bounds(self):
        for config in (dict(max_graph_entries=0),dict(capture_policy="first"),dict(max_captures=-1)):
            with self.assertRaises(ValueError):
                GraphCacheConfig(**config)


if __name__=="__main__":
    unittest.main()
