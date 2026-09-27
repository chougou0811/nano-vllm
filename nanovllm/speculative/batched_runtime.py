"""Per-request TP target state and ragged EAGLE verification."""
import struct
import zlib

import torch
import torch.distributed as dist

from nanovllm.engine.sequence import Sequence
from nanovllm.speculative.batch import SpeculativePhase
from nanovllm.utils.context import get_context, reset_context


def ordered_ids(entries):
    ids = tuple(entry["seq_id"] for entry in entries)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Ordered request IDs must be nonempty and unique")
    return ids


def verification_layout(entries, states, block_size):
    """Pure-CPU mirror of ModelRunner.prepare_prefill for audit/tests."""
    ids = ordered_ids(entries)
    input_ids, positions, cu_q, cu_k, slots = [], [], [0], [0], []
    tables = []
    offsets = [0]
    context_lens = []
    for seq_id, entry in zip(ids, entries):
        state = states[seq_id]
        if entry["generation"] != state["generation"]:
            raise RuntimeError("Stale target request generation")
        cursor, tokens = state["cursor"], state["tokens"]
        if len(tokens) != cursor + 1:
            raise RuntimeError("Target request lacks one pending token")
        proposals = list(entry["proposals"])
        query = tokens[cursor:] + proposals
        q = len(query)
        if q != 1 + len(proposals):
            raise RuntimeError("Invalid verification query")
        blocks = list(entry["blocks"])
        required = (cursor + q + block_size - 1) // block_size
        if len(blocks) < required:
            raise RuntimeError("Verification block table is too short")
        input_ids.extend(query)
        positions.extend(range(cursor, cursor + q))
        cu_q.append(cu_q[-1] + q)
        cu_k.append(cu_k[-1] + cursor + q)
        offsets.append(offsets[-1] + q)
        context_lens.append(cursor)
        for position in range(cursor, cursor + q):
            slots.append(blocks[position // block_size] * block_size + position % block_size)
        tables.append(blocks)
    width = max(map(len, tables))
    padded = [table + [-1] * (width-len(table)) for table in tables]
    return dict(ordered_seq_ids=ids,input_ids=input_ids,positions=positions,
        cu_seqlens_q=cu_q,cu_seqlens_k=cu_k,block_tables=padded,
        slot_mapping=slots,context_lens=context_lens,proposal_offsets=offsets)


def _states(runner):
    if not hasattr(runner, "_eagle_batch_states"):
        runner._eagle_batch_states = {}
    return runner._eagle_batch_states


def _resolve(runner, entries, phases=None):
    states = _states(runner)
    result = []
    for seq_id, entry in zip(ordered_ids(entries), entries):
        if seq_id not in states:
            raise RuntimeError(f"Missing target request state: {seq_id}")
        state = states[seq_id]
        if entry["generation"] != state["generation"]:
            raise RuntimeError("Stale target request generation")
        if phases is not None and state["phase"] not in phases:
            raise RuntimeError(f"Invalid target phase for request {seq_id}")
        result.append(state)
    return result


def _zero(runner, blocks, start, end):
    if end <= start:
        return True
    positions = torch.arange(start,end,device=runner.kv_cache.device)
    table = torch.tensor(blocks,device=positions.device)
    physical,offsets = table[positions//runner.block_size],positions%runner.block_size
    runner.kv_cache[:,:,physical,offsets] = 0
    return not bool(torch.count_nonzero(runner.kv_cache[:,:,physical,offsets]).item())


def _status(runner, entries, clean=None):
    states = _resolve(runner, entries) if entries else []
    rows = []
    for index,(entry,state) in enumerate(zip(entries,states)):
        tokens = state["tokens"]
        crc = zlib.crc32(struct.pack(f"<{len(tokens)}i",*tokens)) if tokens else 0
        rows.append([entry["seq_id"],state["generation"],state["cursor"],len(tokens),
                     0 if state["features"] is None else state["features"].shape[0],
                     0 if state["tentative"] is None else state["tentative"].shape[0],
                     crc,int(True if clean is None else clean[index])])
    tensor = torch.tensor(rows,dtype=torch.int64,device=runner.kv_cache.device)
    ranks = [torch.empty_like(tensor) for _ in range(runner.world_size)]
    dist.all_gather(ranks,tensor)
    return [value.cpu().tolist() for value in ranks] if runner.rank == 0 else None


def _descriptors(entries, states, verification):
    seqs = []
    for entry,state in zip(entries,states):
        if verification:
            tokens = state["tokens"] + list(entry["proposals"])
            start = state["cursor"]
            count = 1 + len(entry["proposals"])
        else:
            tokens = list(entry["tokens"])
            start = entry["start"]
            count = entry["count"]
        seq = Sequence(tokens)
        seq.block_table = list(entry["blocks"])
        seq.num_cached_tokens = start
        seq.num_scheduled_tokens = count
        seqs.append(seq)
    return seqs


def _forward(runner, entries, states, *, verification):
    seqs = _descriptors(entries,states,verification)
    captured,handles = [],[]
    layers = runner.model.model.layers
    capture_ids = [2,len(layers)//2,len(layers)-3]
    def capture(module,args):
        _,hidden,residual = args
        captured.append(hidden.clone() if residual is None else
                        (hidden.float()+residual.float()).to(hidden.dtype))
    try:
        for index in capture_ids:
            handles.append(layers[index].register_forward_pre_hook(capture))
        input_ids,positions = runner.prepare_prefill(seqs)
        hidden = runner.model(input_ids,positions)
        if verification:
            get_context().is_prefill = False
        logits = runner.model.compute_logits(hidden)
        features = torch.cat(captured,dim=-1)
        finite = torch.isfinite(features).all()
        if runner.rank == 0:
            finite &= torch.isfinite(logits).all()
        finite = finite.to(torch.int32)
        dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        replicated = True
        if any(state["audit"] for state in states):
            reference = features.clone()
            dist.broadcast(reference,0)
            same = torch.tensor(int(torch.equal(features,reference)),device=features.device)
            dist.all_reduce(same,op=dist.ReduceOp.MIN)
            replicated = bool(same.item())
        return logits,features,bool(finite.item()),replicated
    finally:
        for handle in handles:
            handle.remove()
        reset_context()


@torch.inference_mode()
def dispatch(runner, operation, payload, *, verification_forward=None):
    entries = payload.get("entries", [])
    states = _states(runner)
    if operation == "begin":
        ids = ordered_ids(entries)
        if any(seq_id in states for seq_id in ids):
            raise RuntimeError("Duplicate live target request")
        for entry in entries:
            states[entry["seq_id"]] = dict(generation=entry["generation"],
                cursor=0,tokens=list(entry["tokens"]),features=None,tentative=None,
                blocks=[],audit=bool(entry.get("audit",False)),phase=SpeculativePhase.PREFILL)
        return _status(runner,entries)
    if operation == "prefill":
        current = _resolve(runner,entries,{SpeculativePhase.PREFILL})
        for entry,state in zip(entries,current):
            if entry["start"] != state["cursor"] or entry["tokens"] != state["tokens"]:
                raise RuntimeError("Noncontiguous target prefill")
            state["blocks"] = list(entry["blocks"])
        logits,features,finite,replicated = _forward(runner,entries,current,verification=False)
        offset = 0
        for entry,state in zip(entries,current):
            count = entry["count"]
            piece = features[offset:offset+count]
            state["features"] = piece if state["features"] is None else torch.cat((state["features"],piece),0)
            state["cursor"] += count
            offset += count
        return dict(target_ids=logits.argmax(-1).tolist() if runner.rank == 0 else None,
                    finite=finite,replicated=replicated,ranks=_status(runner,entries))
    if operation == "seed":
        current = _resolve(runner,entries,{SpeculativePhase.PREFILL})
        for entry,state in zip(entries,current):
            if state["cursor"] != len(state["tokens"]):
                raise RuntimeError("Cannot seed incomplete target prefill")
            state["tokens"].append(entry["token"])
            state["phase"] = SpeculativePhase.READY
        return _status(runner,entries)
    if operation == "verify":
        current = _resolve(runner,entries,{SpeculativePhase.READY})
        verification_layout(entries,{entry["seq_id"]:state for entry,state in zip(entries,current)},runner.block_size)
        for entry,state in zip(entries,current):
            state["blocks"] = list(entry["blocks"])
        logits,features,finite,replicated = (
            _forward(runner,entries,current,verification=True) if verification_forward is None
            else verification_forward(entries,current))
        offsets = [0]
        for entry in entries:
            offsets.append(offsets[-1]+1+len(entry["proposals"]))
        target_ids = []
        for index,state in enumerate(current):
            state["tentative"] = features[offsets[index]:offsets[index+1]]
            state["phase"] = SpeculativePhase.VERIFIED
            if runner.rank == 0:
                target_ids.append(logits[offsets[index]:offsets[index+1]].argmax(-1).tolist())
        return dict(target_ids=target_ids if runner.rank == 0 else None,
                    offsets=offsets,finite=finite,replicated=replicated,ranks=_status(runner,entries))
    if operation == "commit":
        current = _resolve(runner,entries,{SpeculativePhase.VERIFIED})
        clean = []
        for entry,state in zip(entries,current):
            accepted = entry["accepted"]
            tentative = state["tentative"]
            if not 0 <= accepted < tentative.shape[0]:
                raise RuntimeError("Invalid accepted length")
            keep = 1+accepted
            end = state["cursor"]+tentative.shape[0]
            new_end = state["cursor"]+keep
            tokens = list(entry["tokens"])
            if len(tokens) not in (accepted,accepted+1):
                raise RuntimeError("Invalid committed output length")
            clean.append(_zero(runner,state["blocks"],new_end,end))
            state["features"] = torch.cat((state["features"],tentative[:keep]),0)
            state["cursor"] = new_end
            state["tokens"].extend(tokens)
            state["tentative"] = None
            state["phase"] = SpeculativePhase.FINISHED if entry["finished"] else SpeculativePhase.READY
            if not entry["finished"] and len(state["tokens"]) != state["cursor"]+1:
                raise RuntimeError("Committed target lost pending-token invariant")
        return _status(runner,entries,clean)
    if operation == "rollback":
        current = _resolve(runner,entries,{SpeculativePhase.VERIFIED})
        clean = []
        for state in current:
            end = state["cursor"]+state["tentative"].shape[0]
            clean.append(_zero(runner,state["blocks"],state["cursor"],end))
            state["tentative"] = None
            state["phase"] = SpeculativePhase.READY
        return _status(runner,entries,clean)
    if operation == "close":
        current = _resolve(runner,entries)
        clean = []
        for entry,state in zip(entries,current):
            clean.append(_zero(runner,state["blocks"],0,len(state["blocks"])*runner.block_size))
            del states[entry["seq_id"]]
        # State no longer exists, so return a close-specific rank matrix.
        rows = torch.tensor([[entry["seq_id"],entry["generation"],int(value)]
                             for entry,value in zip(entries,clean)],dtype=torch.int64,
                            device=runner.kv_cache.device)
        ranks = [torch.empty_like(rows) for _ in range(runner.world_size)]
        dist.all_gather(ranks,rows)
        return [value.cpu().tolist() for value in ranks] if runner.rank == 0 else None
    raise ValueError(operation)
