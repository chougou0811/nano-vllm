"""Rank-synchronous target KV transactions over existing paged attention."""
import struct
import zlib

import torch
import torch.distributed as dist

from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


def _zero(runner, state, start, end):
    if end <= start:
        return True
    positions = torch.arange(start,end,device=runner.kv_cache.device)
    table = torch.tensor(state["blocks"],device=positions.device)
    blocks,offsets = table[positions//runner.block_size],positions%runner.block_size
    runner.kv_cache[:,:,blocks,offsets] = 0
    return not bool(torch.count_nonzero(runner.kv_cache[:,:,blocks,offsets]).item())


def _status(runner, state, clean=True):
    tokens = state["tokens"] if state else []
    crc = zlib.crc32(struct.pack(f"<{len(tokens)}i",*tokens)) if tokens else 0
    values = [runner.rank, state["cached"] if state else 0,len(tokens),
              state["features"].shape[0] if state and state["features"] is not None else 0,
              crc,int(clean)]
    tensor = torch.tensor(values,dtype=torch.int64,device=runner.kv_cache.device)
    ranks = [torch.empty_like(tensor) for _ in range(runner.world_size)]
    dist.all_gather(ranks,tensor)
    return [r.cpu().tolist() for r in ranks] if runner.rank == 0 else None


def _forward(runner,state,tokens,start,prefill=False):
    seq = Sequence(tokens)
    seq.block_table = state["blocks"]
    seq.num_cached_tokens = start
    seq.num_scheduled_tokens = len(tokens)-start
    captured,handles = [],[]
    layers = runner.model.model.layers
    capture_ids = [2,len(layers)//2,len(layers)-3]
    def capture(module,args):
        _,hidden,residual = args
        captured.append(hidden.clone() if residual is None else (hidden.float()+residual.float()).to(hidden.dtype))
    try:
        for index in capture_ids:
            handles.append(layers[index].register_forward_pre_hook(capture))
        ids,positions = runner.prepare_prefill([seq])
        hidden = runner.model(ids,positions)
        # LM head normally selects only final prefill positions. Verification
        # needs all rows, while attention has already consumed prefill context.
        get_context().is_prefill = False
        logits = runner.model.compute_logits(hidden[-1:] if prefill else hidden)
        features = torch.cat(captured,dim=-1)
        finite = torch.isfinite(features).all()
        if runner.rank == 0:
            finite &= torch.isfinite(logits).all()
        finite = finite.to(torch.int32)
        dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        replicated = True
        if state["audit"]:
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
def dispatch(runner,operation,payload):
    if operation == "begin":
        runner._eagle_state = dict(tokens=list(payload["tokens"]),blocks=payload["blocks"],cached=0,
                                   features=None,tentative=None,audit=payload["audit"])
        return _status(runner,runner._eagle_state)
    state = runner._eagle_state
    if operation == "prefill":
        logits,features,finite,replicated = _forward(runner,state,state["tokens"],0,True)
        state.update(cached=len(state["tokens"]),features=features)
        token = int(logits[0].argmax().item()) if runner.rank == 0 else None
        return dict(token=token,finite=finite,replicated=replicated,ranks=_status(runner,state))
    if operation == "seed":
        state["tokens"].append(payload["token"])
        return _status(runner,state)
    if operation == "verify":
        ids = state["tokens"]+payload["proposals"]
        logits,features,finite,replicated = _forward(runner,state,ids,state["cached"])
        state["tentative"] = features
        return dict(target_ids=logits.argmax(-1).tolist() if runner.rank == 0 else None,
                    logits=logits if runner.rank == 0 and payload.get("return_logits") else None,
                    finite=finite,replicated=replicated,ranks=_status(runner,state))
    if operation == "commit":
        keep = 1+payload["accepted"]
        end = state["cached"]+state["tentative"].shape[0]
        new_end = state["cached"]+keep
        clean = _zero(runner,state,new_end,end)
        state["features"] = torch.cat((state["features"],state["tentative"][:keep]),dim=0)
        state["cached"] = new_end
        state["tokens"].extend(payload["tokens"])
        state["tentative"] = None
        return _status(runner,state,clean)
    if operation == "close":
        clean = _zero(runner,state,0,len(state["blocks"])*runner.block_size)
        runner._eagle_state = None
        return _status(runner,None,clean)
    raise ValueError(operation)
