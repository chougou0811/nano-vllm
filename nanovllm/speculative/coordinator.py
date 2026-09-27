"""Opt-in fixed-K coordinator for concurrent greedy EAGLE requests."""
from time import perf_counter_ns

from nanovllm.speculative.acceptance import accept_greedy
from nanovllm.speculative.batch import (
    ProposalEntry,
    SpeculativeBatch,
    SpeculativeCommit,
    SpeculativePhase,
    SpeculativeRequestState,
)
from nanovllm.speculative.draft import ReferenceDraft
from nanovllm.speculative.draft_state import DraftState


class SpeculativeCoordinator:
    def __init__(self, engine, draft_path, reference_path, *, fixed_k=3, audit=False):
        if not engine.model_runner.enforce_eager:
            raise ValueError("Concurrent EAGLE requires eager execution")
        if fixed_k != 3:
            raise ValueError("Phase 4.1 supports fixed K=3 only")
        self.engine = engine
        self.runner = engine.model_runner
        self.block_manager = engine.scheduler.block_manager
        self.fixed_k = fixed_k
        self.audit = audit
        identity = (str(draft_path),str(reference_path))
        if getattr(engine,"_eagle_draft_identity",None) != identity:
            cfg = self.runner.config
            engine._eagle_draft = ReferenceDraft(draft_path,reference_path,
                cfg.model,cfg.hf_config,self.runner.kv_cache.device)
            engine._eagle_draft_identity = identity
        self.draft = engine._eagle_draft
        self.requests = {}
        self.sequences = {}
        self.drafts = {}
        self.begun = set()
        self.last_step = {}

    def owns(self, seq_id):
        return seq_id in self.requests

    def register(self, seq):
        if seq.seq_id in self.requests:
            raise RuntimeError("Duplicate speculative request")
        self.requests[seq.seq_id] = SpeculativeRequestState(seq.seq_id,0,self.fixed_k)
        self.sequences[seq.seq_id] = seq

    def _entry(self, state, **values):
        return dict(seq_id=state.seq_id,generation=state.generation,**values)

    def _check_ranks(self, ranks, expected_ids, *, close=False):
        if not ranks or any(len(rank) != len(expected_ids) for rank in ranks):
            raise RuntimeError("Missing TP rank status")
        width = 3 if close else 8
        for rank in ranks:
            if any(len(row) != width for row in rank):
                raise RuntimeError("Malformed TP rank status")
            if [row[0] for row in rank] != list(expected_ids):
                raise RuntimeError("TP request ordering disagreement")
            if any(not row[-1] for row in rank):
                raise RuntimeError("TP suffix cleanup failed")
        if any(rank != ranks[0] for rank in ranks[1:]):
            raise RuntimeError("TP request state disagreement")

    def _begin(self, seq):
        state = self.requests[seq.seq_id]
        if seq.seq_id in self.begun:
            return
        result = self.runner.call("eagle3_batch","begin",dict(entries=[
            self._entry(state,tokens=list(seq.token_ids),audit=self.audit)]))
        self._check_ranks(result,(seq.seq_id,))
        self.begun.add(seq.seq_id)

    def prefill(self, seqs):
        for seq in seqs:
            self._begin(seq)
            self.requests[seq.seq_id].require(SpeculativePhase.PREFILL)
        entries = []
        for seq in seqs:
            state = self.requests[seq.seq_id]
            entries.append(self._entry(state,tokens=list(seq.token_ids),
                blocks=list(seq.block_table),start=seq.num_cached_tokens,
                count=seq.num_scheduled_tokens))
        result = self.runner.call("eagle3_batch","prefill",dict(entries=entries))
        if not result["finite"] or not result["replicated"]:
            raise RuntimeError("Invalid batched target prefill")
        self._check_ranks(result["ranks"],tuple(seq.seq_id for seq in seqs))
        completed = []
        for seq,token in zip(seqs,result["target_ids"]):
            state = self.requests[seq.seq_id]
            state.target_cursor += seq.num_scheduled_tokens
            if seq.num_cached_tokens+seq.num_scheduled_tokens == seq.num_tokens:
                completed.append(self._entry(state,token=token))
        if completed:
            ranks = self.runner.call("eagle3_batch","seed",dict(entries=completed))
            self._check_ranks(ranks,tuple(entry["seq_id"] for entry in completed))
            for entry in completed:
                state = self.requests[entry["seq_id"]]
                state.phase = SpeculativePhase.READY
                self.drafts[state.seq_id] = DraftState(self.draft,state.seq_id,"persistent")
                self.drafts[state.seq_id].request_generation = state.generation
        self.last_step = dict(kind="prefill",batch_size=len(seqs),target_query_tokens=
                              sum(seq.num_scheduled_tokens for seq in seqs))
        return result["target_ids"]

    def _proposal_counts(self, seqs):
        if len(seqs) > self.runner.config.max_num_seqs:
            raise RuntimeError("Speculative request batch exceeds max_num_seqs")
        budget = self.runner.config.max_num_batched_tokens
        if budget < len(seqs):
            raise RuntimeError("Target token budget cannot give every request q=1")
        extra = budget-len(seqs)
        result = []
        for seq in seqs:
            remaining = seq.max_tokens-seq.num_completion_tokens
            count = min(self.fixed_k,max(0,remaining-1),extra)
            extra -= count
            reason = None
            if count < self.fixed_k:
                reason = "output-limit" if count == max(0,remaining-1) else "target-token-budget"
            result.append((remaining,count,reason))
        return result

    def decode(self, seqs):
        counts = self._proposal_counts(seqs)
        entries,transactions = [],{}
        proposal_ns = 0
        verified = False
        committed = False
        try:
            batched_proposals = {}
            executor = getattr(self, "draft_batch_executor", None)
            eligible = ([(seq, count) for seq, (_, count, _) in zip(seqs, counts) if count]
                        if executor is not None and len(seqs) > 1 else [])
            if executor is not None and len(eligible) > 1:
                from nanovllm.speculative.draft_batch import DraftBatchInput
                inputs = []
                for seq, count in eligible:
                    state = self.requests[seq.seq_id]
                    state.require(SpeculativePhase.READY)
                    target = self.runner._eagle_batch_states[seq.seq_id]
                    inputs.append(DraftBatchInput(self.drafts[seq.seq_id], state,
                        state.generation, target["features"], tuple(target["tokens"]), count))
                start = perf_counter_ns()
                proposed = executor.propose(inputs)
                proposal_ns += perf_counter_ns()-start
                batched_proposals = {seq.seq_id: tokens for (seq, _), tokens in zip(eligible, proposed)}
            offset = 0
            for seq,(remaining,count,reason) in zip(seqs,counts):
                state = self.requests[seq.seq_id]
                state.require(SpeculativePhase.READY)
                state.phase = SpeculativePhase.PROPOSING
                target = self.runner._eagle_batch_states[seq.seq_id]
                start = perf_counter_ns()
                if seq.seq_id in batched_proposals:
                    proposals = batched_proposals[seq.seq_id]
                else:
                    proposals = self.drafts[seq.seq_id].propose(
                        target["features"],target["tokens"],count,owner=seq.seq_id) if count else []
                proposal_ns += perf_counter_ns()-start
                while not self.block_manager.can_reserve_rows(seq,state.target_cursor,1+len(proposals)):
                    if not proposals:
                        raise MemoryError("Cannot reserve q=1 target row")
                    proposals.pop()
                    reason = "kv-capacity"
                transaction = self.block_manager.reserve_rows(seq,state.target_cursor,
                    1+len(proposals),generation=state.generation)
                transactions[seq.seq_id] = transaction
                state.transaction = transaction
                state.proposed_token_ids = tuple(proposals)
                state.clipping_reason = reason
                state.phase = SpeculativePhase.RESERVED
                entry = ProposalEntry(seq.seq_id,state.generation,state.target_cursor,
                    self.drafts[seq.seq_id].cursor,remaining,self.fixed_k,len(proposals),
                    tuple(proposals),offset,reason)
                entries.append(entry)
                offset = entry.q_end
            batch = SpeculativeBatch(tuple(entries),self.runner.config.max_num_batched_tokens)
            verify_entries = [dict(seq_id=e.seq_id,generation=e.generation,
                proposals=list(e.proposed_token_ids),blocks=list(self.sequences[e.seq_id].block_table))
                for e in entries]
            verify_start = perf_counter_ns()
            verification = self.runner.call("eagle3_batch","verify",dict(entries=verify_entries))
            verification_ns = perf_counter_ns()-verify_start
            verified = True
            if not verification["finite"] or not verification["replicated"]:
                raise RuntimeError("Invalid batched target verification")
            self._check_ranks(verification["ranks"],batch.ordered_seq_ids)
            results = []
            commit_entries = []
            eos = self.runner.config.eos
            for entry,target_ids in zip(entries,verification["target_ids"]):
                state = self.requests[entry.seq_id]
                state.phase = SpeculativePhase.VERIFIED
                result = accept_greedy(entry.proposed_token_ids,target_ids,entry.remaining,eos)
                commit_entries.append(dict(seq_id=entry.seq_id,generation=entry.generation,
                    accepted=result.accepted,tokens=result.tokens,finished=result.finished))
                results.append((entry,target_ids,result))
            commit_ranks = self.runner.call("eagle3_batch","commit",dict(entries=commit_entries))
            committed = True
            self._check_ranks(commit_ranks,batch.ordered_seq_ids)
            commits = []
            for entry,target_ids,result in results:
                seq = self.sequences[entry.seq_id]
                state = self.requests[entry.seq_id]
                new_cursor = entry.old_cursor+1+result.accepted
                self.block_manager.commit_rows(seq,transactions[entry.seq_id],new_cursor)
                state.transaction = None
                state.target_cursor = new_cursor
                state.accepted_length = result.accepted
                state.committed_token_ids = tuple(result.tokens)
                state.fallback_token = result.fallback
                state.phase = SpeculativePhase.FINISHED if result.finished else SpeculativePhase.READY
                target = self.runner._eagle_batch_states[entry.seq_id]
                self.drafts[entry.seq_id].committed(new_cursor,target["tokens"],owner=entry.seq_id)
                commits.append(SpeculativeCommit(entry.seq_id,entry.generation,
                    entry.old_cursor,new_cursor,entry.proposed_token_ids,tuple(target_ids),
                    result.accepted,tuple(result.tokens),result.fallback,result.finished,
                    entry.clipping_reason,entry.requested_k,entry.actual_k))
            self.last_step = dict(kind="decode",batch_size=len(seqs),
                target_query_tokens=batch.total_query_tokens,proposal_ns=proposal_ns,
                verification_ns=verification_ns,entries=commits)
            return commits
        except Exception:
            if verified and not committed:
                rollback = [self._entry(self.requests[e.seq_id]) for e in entries]
                try:
                    ranks = self.runner.call("eagle3_batch","rollback",dict(entries=rollback))
                    self._check_ranks(ranks,tuple(e.seq_id for e in entries))
                except Exception:
                    pass
            for seq_id,transaction in transactions.items():
                if transaction.active:
                    self.block_manager.rollback_rows(self.sequences[seq_id],transaction)
                state = self.requests[seq_id]
                state.transaction = None
                state.fail("decode transaction failed")
            raise

    def close_request(self, seq_id, *, retain_registration=False):
        if seq_id not in self.requests:
            return False
        state = self.requests[seq_id]
        if state.transaction is not None and state.transaction.active:
            self.block_manager.rollback_rows(self.sequences[seq_id],state.transaction)
            state.transaction = None
        if seq_id in self.begun:
            ranks = self.runner.call("eagle3_batch","close",dict(entries=[self._entry(state)]))
            self._check_ranks(ranks,(seq_id,),close=True)
            self.begun.remove(seq_id)
        draft = self.drafts.pop(seq_id,None)
        if draft is not None:
            draft.close()
        state.phase = SpeculativePhase.FINISHED if state.error is None else SpeculativePhase.FAILED
        state.close()
        if retain_registration:
            generation = state.generation+1
            self.requests[seq_id] = SpeculativeRequestState(seq_id,generation,self.fixed_k)
        else:
            del self.requests[seq_id]
            self.sequences.pop(seq_id,None)
        return True

    def preempt(self, seq):
        if seq.seq_id in self.requests:
            self.close_request(seq.seq_id,retain_registration=True)

    def close_all(self):
        for seq_id in list(self.requests):
            self.close_request(seq_id)
