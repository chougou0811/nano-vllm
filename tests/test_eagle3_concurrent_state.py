import unittest

from nanovllm.speculative.batch import (
    KVBlockTransaction,
    ProposalEntry,
    SpeculativeBatch,
    SpeculativePhase,
    SpeculativeRequestState,
    TargetRequestState,
)


class ConcurrentStateTests(unittest.TestCase):
    def test_request_state_transitions_and_idempotent_close(self):
        state = SpeculativeRequestState(7, 3, 3)
        state.transition(SpeculativePhase.PREFILL, SpeculativePhase.READY)
        state.transition(SpeculativePhase.READY, SpeculativePhase.PROPOSING)
        with self.assertRaises(RuntimeError):
            state.transition(SpeculativePhase.READY, SpeculativePhase.RESERVED)
        state.phase = SpeculativePhase.FINISHED
        self.assertTrue(state.close())
        self.assertFalse(state.close())
        with self.assertRaises(RuntimeError):
            state.require(SpeculativePhase.READY)

    def test_owner_generation_guards(self):
        state = SpeculativeRequestState(4, 9, 3)
        state.validate_owner(4, 9)
        with self.assertRaisesRegex(RuntimeError, "Foreign"):
            state.validate_owner(4, 10)
        txn = KVBlockTransaction(4, 9, 255, 259, 1)
        txn.validate_owner(4, 9)
        with self.assertRaisesRegex(RuntimeError, "Foreign"):
            txn.validate_owner(5, 9)

    def test_active_transaction_prevents_close(self):
        state = SpeculativeRequestState(1, 0, 3, phase=SpeculativePhase.FAILED)
        state.transaction = KVBlockTransaction(1, 0, 0, 4, 0)
        with self.assertRaisesRegex(RuntimeError, "active KV"):
            state.close()
        state.transaction.active = False
        self.assertTrue(state.close())

    def test_target_descriptor_invariants(self):
        TargetRequestState(1, 0, cursor=8, token_count=9, feature_rows=8,
                           phase=SpeculativePhase.READY).validate()
        for state in [
            TargetRequestState(1, 0, cursor=8, token_count=8, feature_rows=8,
                               phase=SpeculativePhase.READY),
            TargetRequestState(1, 0, cursor=8, token_count=9, feature_rows=7,
                               phase=SpeculativePhase.READY),
            TargetRequestState(1, 0, cursor=8, token_count=9, feature_rows=8,
                               tentative_rows=2, phase=SpeculativePhase.READY),
        ]:
            with self.assertRaises(RuntimeError):
                state.validate()

    def entry(self, seq_id, offset, k=3, remaining=8):
        return ProposalEntry(seq_id, 0, 256 + seq_id, 255 + seq_id, remaining,
                             3, k, tuple(range(k)), offset,
                             "output-limit" if k < 3 else None)

    def test_ragged_batch_offsets_and_budget(self):
        entries = (self.entry(1, 0, 3), self.entry(2, 4, 0, 1), self.entry(3, 5, 2))
        batch = SpeculativeBatch(entries, 8)
        self.assertEqual(batch.ordered_seq_ids, (1, 2, 3))
        self.assertEqual(batch.proposal_offsets, (0, 4, 5, 8))
        self.assertEqual(batch.bounds(2), (4, 5))
        self.assertEqual(batch.total_query_tokens, 8)
        with self.assertRaises(KeyError):
            batch.bounds(99)

    def test_batch_rejects_duplicates_gaps_and_over_budget(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            SpeculativeBatch((self.entry(1, 0), self.entry(1, 4)), 8)
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            SpeculativeBatch((self.entry(1, 0), self.entry(2, 5)), 9)
        with self.assertRaisesRegex(ValueError, "budget"):
            SpeculativeBatch((self.entry(1, 0), self.entry(2, 4)), 7)

    def test_proposal_entry_validates_cursors_and_remaining(self):
        with self.assertRaises(ValueError):
            ProposalEntry(1, 0, 4, 5, 8, 3, 3, (1, 2, 3), 0)
        with self.assertRaises(ValueError):
            ProposalEntry(1, 0, 4, 3, 2, 3, 2, (1, 2), 0)
        with self.assertRaises(ValueError):
            ProposalEntry(1, 0, 4, 3, 8, 3, 2, (1,), 0)


if __name__ == "__main__":
    unittest.main()
