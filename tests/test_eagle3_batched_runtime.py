import unittest

from nanovllm.speculative.batched_runtime import ordered_ids, verification_layout


class BatchedRuntimeLayoutTests(unittest.TestCase):
    def state(self, cursor, pending, generation=0):
        return dict(generation=generation,cursor=cursor,
                    tokens=list(range(cursor))+[pending])

    def test_ragged_verification_layout(self):
        states = {10:self.state(255,90), 20:self.state(512,91), 30:self.state(257,92)}
        entries = [
            dict(seq_id=10,generation=0,proposals=[1,2,3],blocks=[7,3]),
            dict(seq_id=20,generation=0,proposals=[],blocks=[8,9,4]),
            dict(seq_id=30,generation=0,proposals=[5,6],blocks=[1,6]),
        ]
        layout = verification_layout(entries,states,256)
        self.assertEqual(layout["ordered_seq_ids"],(10,20,30))
        self.assertEqual(layout["proposal_offsets"],[0,4,5,8])
        self.assertEqual(layout["cu_seqlens_q"],[0,4,5,8])
        self.assertEqual(layout["cu_seqlens_k"],[0,259,772,1032])
        self.assertEqual(layout["context_lens"],[255,512,257])
        self.assertEqual(layout["positions"],
                         [255,256,257,258,512,257,258,259])
        self.assertEqual(layout["input_ids"],
                         [90,1,2,3,91,92,5,6])
        self.assertEqual(layout["slot_mapping"],
                         [7*256+255,3*256,3*256+1,3*256+2,
                          4*256,6*256+1,6*256+2,6*256+3])
        self.assertEqual(layout["block_tables"],[[7,3,-1],[8,9,4],[1,6,-1]])

    def test_duplicate_missing_capacity_and_stale_generation(self):
        state = self.state(255,90)
        with self.assertRaises(ValueError):
            ordered_ids([dict(seq_id=1),dict(seq_id=1)])
        with self.assertRaises(KeyError):
            verification_layout([dict(seq_id=2,generation=0,proposals=[],blocks=[1])],
                                {1:state},256)
        with self.assertRaisesRegex(RuntimeError,"Stale"):
            verification_layout([dict(seq_id=1,generation=1,proposals=[],blocks=[1])],
                                {1:state},256)
        with self.assertRaisesRegex(RuntimeError,"too short"):
            verification_layout([dict(seq_id=1,generation=0,proposals=[1],blocks=[1])],
                                {1:state},256)

    def test_pending_token_invariant(self):
        state = self.state(16,90)
        state["tokens"].append(91)
        with self.assertRaisesRegex(RuntimeError,"pending"):
            verification_layout([dict(seq_id=1,generation=0,proposals=[],blocks=[1])],
                                {1:state},256)


if __name__ == "__main__":
    unittest.main()
