"""Original Scheduler selection with opt-in speculative execution lifecycle."""
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.speculative.block_transactions import TransactionalBlockManager


class SpeculativeBlockManager(TransactionalBlockManager):
    def __init__(self, original, coordinator):
        super().__init__(original)
        self.coordinator = coordinator

    def can_allocate(self, seq):
        if self.coordinator.owns(seq.seq_id):
            if len(self.original.free_block_ids) < seq.num_blocks:
                return -1
            return 0
        return self.original.can_allocate(seq)

    def deallocate(self, seq):
        if self.coordinator.owns(seq.seq_id):
            self.coordinator.preempt(seq)
        return super().deallocate(seq)


class SpeculativeSchedulerAdapter:
    speculative = True

    def __init__(self, original, coordinator):
        self.original = original
        self.coordinator = coordinator
        self._block_manager = SpeculativeBlockManager(original.block_manager,coordinator)
        original.block_manager = self._block_manager
        coordinator.block_manager = self._block_manager

    def __getattr__(self, name):
        return getattr(self.original,name)

    @property
    def block_manager(self):
        return self._block_manager

    def add(self, seq):
        if len(self.original.waiting)+len(self.original.running) >= self.original.max_num_seqs:
            raise ValueError("Phase 4.1 active concurrency cannot exceed max_num_seqs")
        self.coordinator.register(seq)
        try:
            return self.original.add(seq)
        except Exception:
            self.coordinator.close_request(seq.seq_id)
            raise

    def schedule(self):
        seqs, is_prefill = self.original.schedule()
        if is_prefill or len(seqs) <= self.original.max_num_batched_tokens:
            return seqs, is_prefill
        selected = seqs[:self.original.max_num_batched_tokens]
        for seq in seqs[self.original.max_num_batched_tokens:]:
            seq.num_scheduled_tokens = 0
        return selected, False

    def postprocess_prefill(self, seqs, token_ids):
        for seq,token_id in zip(seqs,token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if ((not seq.ignore_eos and token_id == self.original.eos) or
                    seq.num_completion_tokens == seq.max_tokens):
                seq.status = SequenceStatus.FINISHED
                self.coordinator.close_request(seq.seq_id)
                self.original.block_manager.original.deallocate(seq)
                self.original.running.remove(seq)

    def postprocess_decode(self, seqs, commits):
        by_id = {result.seq_id:result for result in commits}
        if set(by_id) != {seq.seq_id for seq in seqs}:
            raise RuntimeError("Speculative commit batch does not match Scheduler batch")
        for seq in seqs:
            result = by_id[seq.seq_id]
            for token in result.committed_token_ids:
                seq.append_token(token)
            seq.num_cached_tokens = result.new_cursor
            seq.num_scheduled_tokens = 0
            if result.finished:
                seq.status = SequenceStatus.FINISHED
                self.coordinator.close_request(seq.seq_id)
                self.original.block_manager.original.deallocate(seq)
                self.original.running.remove(seq)

    def abort_all(self, error):
        for seq in list(self.original.waiting)+list(self.original.running):
            if self.coordinator.owns(seq.seq_id):
                self.coordinator.requests[seq.seq_id].fail(error)
                try:
                    self.coordinator.close_request(seq.seq_id)
                finally:
                    self.original.block_manager.original.deallocate(seq)
                seq.status = SequenceStatus.FINISHED
        self.original.waiting.clear()
        self.original.running.clear()

    def close_all(self):
        self.abort_all("engine exit")
