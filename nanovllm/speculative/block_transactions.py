"""Transactional KV block reservations layered over the frozen BlockManager."""
from nanovllm.speculative.batch import KVBlockTransaction


class TransactionalBlockManager:
    """Add tentative multi-row ownership without changing BlockManager."""

    def __init__(self, original):
        self.original = original
        self.transactions: dict[int, KVBlockTransaction] = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _required_blocks(self, end: int) -> int:
        if end < 0:
            raise ValueError("KV row end cannot be negative")
        return (end + self.block_size - 1) // self.block_size

    def can_reserve_rows(self, seq, start_row: int, num_rows: int) -> bool:
        if start_row < 0 or num_rows < 1:
            raise ValueError("Invalid speculative KV row range")
        if seq.seq_id in self.transactions:
            raise RuntimeError("Sequence already has an active KV transaction")
        required = self._required_blocks(start_row + num_rows)
        return max(0, required - len(seq.block_table)) <= len(self.free_block_ids)

    def reserve_rows(self, seq, start_row: int, num_rows: int,
                     *, generation: int = 0) -> KVBlockTransaction:
        if not self.can_reserve_rows(seq, start_row, num_rows):
            raise MemoryError("Insufficient blocks for speculative KV transaction")
        original_count = len(seq.block_table)
        required = self._required_blocks(start_row + num_rows)
        new_blocks = tuple(
            self.original._allocate_block()
            for _ in range(max(0, required - original_count))
        )
        seq.block_table.extend(new_blocks)
        transaction = KVBlockTransaction(
            seq.seq_id,
            generation,
            start_row,
            start_row + num_rows,
            original_count,
            new_blocks,
        )
        self.transactions[seq.seq_id] = transaction
        return transaction

    def _release_suffix(self, seq, transaction, keep_blocks: int) -> tuple[int, ...]:
        keep_blocks = max(transaction.original_block_count, keep_blocks)
        if keep_blocks > len(seq.block_table):
            raise RuntimeError("Cannot retain unreserved KV blocks")
        released = []
        while len(seq.block_table) > keep_blocks:
            block_id = seq.block_table[-1]
            if block_id not in transaction.new_block_ids:
                raise RuntimeError("Rollback would release an inherited block")
            seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count != 0:
                raise RuntimeError("Transaction-owned block is unexpectedly shared")
            self.original._deallocate_block(block_id)
            released.append(block_id)
        return tuple(released)

    def commit_rows(self, seq, transaction: KVBlockTransaction,
                    committed_end: int) -> tuple[int, ...]:
        transaction.validate_owner(seq.seq_id, transaction.generation)
        if not transaction.active:
            return transaction.released_block_ids
        if self.transactions.get(seq.seq_id) is not transaction:
            raise RuntimeError("Unknown KV transaction")
        if not transaction.start_row <= committed_end <= transaction.tentative_end:
            raise ValueError("Committed KV cursor outside tentative range")
        released = self._release_suffix(
            seq, transaction, self._required_blocks(committed_end)
        )
        transaction.committed_end = committed_end
        transaction.released_block_ids = released
        transaction.active = False
        del self.transactions[seq.seq_id]
        return released

    def rollback_rows(self, seq, transaction: KVBlockTransaction) -> tuple[int, ...]:
        transaction.validate_owner(seq.seq_id, transaction.generation)
        if not transaction.active:
            return transaction.released_block_ids
        if self.transactions.get(seq.seq_id) is not transaction:
            raise RuntimeError("Unknown KV transaction")
        released = self._release_suffix(
            seq, transaction, transaction.original_block_count
        )
        transaction.released_block_ids = released
        transaction.active = False
        del self.transactions[seq.seq_id]
        return released

    def deallocate(self, seq):
        transaction = self.transactions.get(seq.seq_id)
        if transaction is not None:
            self.rollback_rows(seq, transaction)
        return self.original.deallocate(seq)
