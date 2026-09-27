import unittest

from benchmarks.serving.eagle3_graph_key import agree, make_key


class GraphPrototypeTests(unittest.TestCase):
    def key(self, contexts=(255,), width=2, lengths=None):
        lengths = lengths or [4]*len(contexts)
        cu = [0]
        for q in lengths:
            cu.append(cu[-1]+q)
        return make_key(dict(cu_seqlens_q=cu, context_lens=contexts,
                             block_tables=[[1]*width for _ in contexts]),
                        block_size=256,model_identity="frozen",dtype="bf16",
                        device_class="cuda-sm89",tp_size=2,training=True,
                        feature_layers=(2,20,37))

    def test_three_supported_m(self):
        for batch in (1,2,4):
            self.assertEqual(self.key((255,)*batch).rows,4*batch)

    def test_m_does_not_define_key(self):
        self.assertNotEqual(self.key((255,)),self.key((256,)))
        self.assertNotEqual(self.key(width=2),self.key(width=3))

    def test_heterogeneous_metadata_same_safe_key(self):
        self.assertEqual(self.key((255,257)),self.key((256,257)))

    def test_ragged_or_unsupported_falls_back(self):
        self.assertIsNone(self.key((255,255),lengths=[3,5]))
        self.assertIsNone(self.key((255,)*3))

    def plan(self, **changes):
        return dict(key=self.key(),layout_digest="same",mode="graph",epoch=1,
                    available=True,error=None) | changes

    def test_single_rank_cache_miss_forces_both_eager(self):
        self.assertEqual(agree([self.plan(),self.plan(available=False)]),"eager")

    def test_consensus_graph_and_eager(self):
        self.assertEqual(agree([self.plan(),self.plan()]),"graph")
        self.assertEqual(agree([self.plan(mode="eager")]*2),"eager")

    def test_plan_mismatch_and_error_fail_closed(self):
        for change in (dict(key=self.key((256,))),dict(layout_digest="different"),
                       dict(epoch=2),dict(mode="eager"),dict(error="invalid")):
            with self.assertRaises(RuntimeError):
                agree([self.plan(),self.plan(**change)])


if __name__ == "__main__":
    unittest.main()
