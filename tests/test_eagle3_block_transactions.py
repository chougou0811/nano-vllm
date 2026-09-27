import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.speculative.block_transactions import TransactionalBlockManager


class BlockTransactionTests(unittest.TestCase):
    def manager(self, blocks):
        return TransactionalBlockManager(BlockManager(blocks, 256))

    def request(self, manager, cursor):
        seq = Sequence([1] * (cursor + 1))
        manager.allocate(seq, 0)
        return seq

    def assert_consistent(self, manager):
        self.assertFalse(manager.used_block_ids & set(manager.free_block_ids))
        self.assertEqual(len(manager.used_block_ids) + len(manager.free_block_ids),
                         len(manager.blocks))
        for block in manager.blocks:
            self.assertEqual(block.block_id in manager.used_block_ids, block.ref_count > 0)

    def test_boundary_reservation_and_full_rollback(self):
        for cursor, expected_new in [(255, 1), (256, 0), (257, 0),
                                     (511, 1), (512, 0), (513, 0)]:
            with self.subTest(cursor=cursor):
                manager = self.manager(12)
                seq = self.request(manager, cursor)
                before = tuple(seq.block_table)
                transaction = manager.reserve_rows(seq, cursor, 4, generation=9)
                self.assertEqual(len(transaction.new_block_ids), expected_new)
                self.assertEqual(tuple(seq.block_table[:len(before)]), before)
                self.assertFalse(manager.hash_to_block_id)
                released = manager.rollback_rows(seq, transaction)
                self.assertEqual(set(released), set(transaction.new_block_ids))
                self.assertEqual(tuple(seq.block_table), before)
                self.assertEqual(manager.rollback_rows(seq, transaction), released)
                self.assert_consistent(manager)
                manager.deallocate(seq)
                self.assertFalse(manager.used_block_ids)

    def test_different_accept_lengths_keep_only_needed_transaction_blocks(self):
        for cursor in (255, 511):
            for accepted in range(4):
                with self.subTest(cursor=cursor, accepted=accepted):
                    manager = self.manager(12)
                    seq = self.request(manager, cursor)
                    original = len(seq.block_table)
                    transaction = manager.reserve_rows(seq, cursor, 4)
                    committed_end = cursor + 1 + accepted
                    manager.commit_rows(seq, transaction, committed_end)
                    expected = max(original, (committed_end + 255) // 256)
                    self.assertEqual(len(seq.block_table), expected)
                    self.assertNotIn(seq.seq_id, manager.transactions)
                    self.assertFalse(manager.hash_to_block_id)
                    self.assert_consistent(manager)
                    manager.deallocate(seq)

    def test_rollback_never_decrements_inherited_shared_prefix(self):
        manager = self.manager(12)
        seed = Sequence([7] * 256)
        manager.allocate(seed, 0)
        seed.num_scheduled_tokens = 256
        manager.hash_blocks(seed)
        shared_id = seed.block_table[0]
        manager.deallocate(seed)
        first = Sequence([7] * 256 + [1])
        second = Sequence([7] * 256 + [2])
        manager.allocate(first, manager.can_allocate(first))
        manager.allocate(second, manager.can_allocate(second))
        self.assertEqual(first.block_table[0], shared_id)
        self.assertEqual(second.block_table[0], shared_id)
        self.assertEqual(manager.blocks[shared_id].ref_count, 2)
        transaction = manager.reserve_rows(first, 512, 4)
        manager.rollback_rows(first, transaction)
        self.assertEqual(manager.blocks[shared_id].ref_count, 2)
        self.assertIn(shared_id, first.block_table)
        self.assertIn(shared_id, second.block_table)
        self.assert_consistent(manager)
        manager.deallocate(first)
        self.assertEqual(manager.blocks[shared_id].ref_count, 1)
        manager.deallocate(second)
        self.assertFalse(manager.used_block_ids)

    def test_capacity_failure_has_no_side_effect(self):
        manager = self.manager(1)
        seq = self.request(manager, 255)
        before = tuple(seq.block_table)
        self.assertFalse(manager.can_reserve_rows(seq, 255, 4))
        with self.assertRaises(MemoryError):
            manager.reserve_rows(seq, 255, 4)
        self.assertEqual(tuple(seq.block_table), before)
        self.assertFalse(manager.transactions)
        self.assert_consistent(manager)

    def test_deallocate_rolls_back_active_transaction_idempotently(self):
        manager = self.manager(8)
        seq = self.request(manager, 255)
        transaction = manager.reserve_rows(seq, 255, 4)
        manager.deallocate(seq)
        self.assertFalse(transaction.active)
        self.assertFalse(manager.transactions)
        self.assertFalse(manager.used_block_ids)
        self.assertFalse(seq.block_table)
        manager.deallocate(seq)
        self.assertFalse(manager.used_block_ids)

    def test_active_transaction_and_owner_guards(self):
        manager = self.manager(8)
        seq = self.request(manager, 255)
        transaction = manager.reserve_rows(seq, 255, 4, generation=2)
        with self.assertRaisesRegex(RuntimeError, "active"):
            manager.reserve_rows(seq, 255, 1)
        other = self.request(manager, 16)
        with self.assertRaisesRegex(RuntimeError, "Foreign"):
            manager.rollback_rows(other, transaction)
        manager.rollback_rows(seq, transaction)
        manager.deallocate(seq)
        manager.deallocate(other)


if __name__ == "__main__":
    unittest.main()
