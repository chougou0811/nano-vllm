"""Isolated single-request EAGLE session; the online Scheduler stays frozen."""
from time import perf_counter_ns

import torch

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from .acceptance import accept_greedy
from .draft import ReferenceDraft
from .draft_state import DraftState
from .controller import AdaptiveK


def check_ranks(rows):
    if not rows or any(row[1:] != rows[0][1:] for row in rows) or not all(row[-1] for row in rows):
        raise RuntimeError(f"Rank state / cleared-suffix disagreement: {rows}")


def generate(engine,prompt,*,draft_path,reference_path,max_tokens,k,ignore_eos,audit,
             proposal_override=None,eos_override=None,reference_probe=None,
             draft_state_mode="full_rebuild",state_probe=None,adaptive_k=False,k_controller=None,
             adaptive_prior=None):
    runner = engine.model_runner
    if not runner.enforce_eager:
        raise ValueError("Phase 1 requires eager execution")
    bm = engine.scheduler.block_manager
    if not engine.is_finished() or bm.used_block_ids:
        raise ValueError("EAGLE Phase 1 requires an idle engine and exclusive KV lease")
    tokens = engine.tokenizer.encode(prompt) if isinstance(prompt,str) else list(prompt)
    cfg = runner.config
    if adaptive_k and (k < 1 or draft_state_mode != "persistent" or k_controller is not None):
        raise ValueError("Adaptive K requires persistent state, K>0 and a private controller")
    controller = AdaptiveK() if adaptive_k else k_controller
    if adaptive_prior is not None:
        if not adaptive_k:
            raise ValueError("A calibration prior requires adaptive_k=True")
        from .prior_controller import PriorAdaptiveK
        controller = PriorAdaptiveK(adaptive_prior)
    if controller is not None and k < 1:
        raise ValueError("K controller requires an enabled draft")
    if not tokens or max_tokens < 1 or k < 0 or len(tokens)+max_tokens > cfg.max_model_len:
        raise ValueError("Invalid prompt/output/K/context lengths")
    if any(not isinstance(t,int) or t < 0 or t >= cfg.hf_config.vocab_size for t in tokens):
        raise ValueError("Invalid target token ID")
    if k:
        identity = (str(draft_path),str(reference_path))
        if getattr(engine,"_eagle_draft_identity",None) != identity:
            engine._eagle_draft = ReferenceDraft(draft_path,reference_path,cfg.model,cfg.hf_config,runner.kv_cache.device)
            engine._eagle_draft_identity = identity
    eos = None if ignore_eos else (cfg.eos if eos_override is None else eos_override)
    lease = Sequence([0]*(len(tokens)+max_tokens))
    if lease.num_blocks > len(bm.free_block_ids):
        raise ValueError("Insufficient free KV pages for exclusive request capacity")
    seq = Sequence(tokens,SamplingParams(max_tokens=max_tokens,ignore_eos=ignore_eos))
    steps,token_times = [],[]
    draft_state = DraftState(engine._eagle_draft, seq.seq_id, draft_state_mode) if k else None
    start = perf_counter_ns()
    bm.allocate(lease,0)
    begun = False
    def call(operation,payload):
        return runner.call("eagle3",operation,payload)
    close_status = None
    try:
        ranks = call("begin",dict(tokens=tokens,blocks=lease.block_table,audit=audit))
        begun = True
        check_ranks(ranks)
        first = call("prefill",{})
        if not first["finite"] or not first["replicated"]:
            raise RuntimeError("Nonfinite target or nonreplicated hidden features")
        check_ranks(first["ranks"])
        check_ranks(call("seed",dict(token=first["token"])))
        seq.append_token(first["token"])
        token_times.append(perf_counter_ns())
        while seq.num_completion_tokens < max_tokens and seq.last_token != eos:
            remaining = max_tokens-seq.num_completion_tokens
            context_length = runner._eagle_state["cached"]
            select_start = perf_counter_ns()
            requested_k = controller.select(context_length,remaining) if controller is not None else k
            if controller is not None and (type(requested_k) is not int or requested_k not in range(1,7)):
                raise ValueError("Controller must choose integer K in [1,6]")
            count = min(requested_k,remaining-1)
            selection_ns = perf_counter_ns()-select_start
            proposals = []
            p_start = perf_counter_ns()
            if count:
                state = runner._eagle_state
                proposals = draft_state.propose(state["features"],state["tokens"],count,owner=seq.seq_id)
                if state_probe is not None:
                    state_probe(draft_state,state,proposals,len(steps))
                if reference_probe is not None:
                    reference_probe(engine._eagle_draft,state,proposals)
                if proposal_override is not None:
                    proposals = proposal_override(proposals,len(steps))
                if len(proposals) != count or any(t < 0 or t >= cfg.hf_config.vocab_size for t in proposals):
                    raise ValueError("Invalid proposal")
            p_end = perf_counter_ns()
            verification = call("verify",dict(proposals=proposals,return_logits=reference_probe is not None))
            v_end = perf_counter_ns()
            if not verification["finite"] or not verification["replicated"]:
                raise RuntimeError("Invalid verification features/logits")
            check_ranks(verification["ranks"])
            result = accept_greedy(proposals,verification["target_ids"],remaining,eos)
            if reference_probe is not None:
                reference_probe(None,verification,(seq.last_token,proposals,result))
            ranks = call("commit",dict(accepted=result.accepted,tokens=result.tokens))
            check_ranks(ranks)
            if draft_state is not None:
                state = runner._eagle_state
                draft_state.committed(state["cached"],state["tokens"],owner=seq.seq_id)
            for token in result.tokens:
                seq.append_token(token)
            now = perf_counter_ns()
            token_times.extend([now]*len(result.tokens))
            conditioning_ns = draft_state.last_metrics["conditioning_ns"] if count else 0
            conditioning_rows = (draft_state.last_metrics["cursor_after"]-draft_state.last_metrics["cursor_before"]) if count else 0
            steps.append(dict(proposed_tokens=proposals,target_ids=verification["target_ids"],
                matched=result.matched,accepted=result.accepted,committed_tokens=result.tokens,
                fallback_token=result.fallback,rejection=int(result.matched < len(proposals)),
                discarded=len(proposals)-result.accepted,proposal_latency_ns=p_end-p_start,
                verification_latency_ns=v_end-p_end,commit_ns=now,ranks=ranks,
                speculator_forwards=count,
                requested_k=requested_k,actual_k=count,context_length=context_length,
                controller_selection_ns=selection_ns,
                conditioning_latency_ns=conditioning_ns,conditioning_rows=conditioning_rows,
                proposal_work_latency_ns=p_end-p_start-conditioning_ns,
                controller=dict(controller.last_decision) if controller is not None else None,
                draft_state=dict(draft_state.last_metrics) if count else {}))
            if controller is not None:
                extra = dict(conditioning_ns=conditioning_ns,conditioning_rows=conditioning_rows) if getattr(controller,"uses_catchup_cost",False) else {}
                controller.observe(context=context_length,requested_k=requested_k,proposed=count,
                    accepted=result.accepted,outputs=len(result.tokens),proposal_ns=p_end-p_start,
                    verification_ns=v_end-p_end,initial_prefix=len(steps)==1,terminal=result.finished,**extra)
            if result.finished:
                break
        seq.status = SequenceStatus.FINISHED
    finally:
        try:
            if begun:
                close_status = call("close",{})
                check_ranks(close_status)
        finally:
            if draft_state is not None:
                draft_state.close()
            bm.deallocate(lease)
    end = perf_counter_ns()
    return dict(token_ids=seq.completion_token_ids,text=engine.tokenizer.decode(seq.completion_token_ids),
        arrival_ns=start,token_times_ns=token_times,finish_ns=token_times[-1],cleanup_ns=end,
        steps=steps,k=k,target_forward_count=1+len(steps),
        speculator_forward_count=sum(s["speculator_forwards"] for s in steps),
        kv_released=not bm.used_block_ids,spec_state_released=runner._eagle_state is None,
        draft_state_released=draft_state is None or (draft_state.closed and draft_state.past is None),
        draft_state_mode=draft_state_mode,
        adaptive_k=adaptive_k,
        close_ranks=close_status,prefill_ranks=first["ranks"])
