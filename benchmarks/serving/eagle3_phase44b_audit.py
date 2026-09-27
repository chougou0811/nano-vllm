"""GPU audit observer; never imported by production."""
from dataclasses import replace
import hashlib

import torch
import torch.distributed as dist

from nanovllm.speculative.graph_runner import GraphConcurrentModelRunner
from nanovllm.speculative import batched_runtime as runtime


def byte_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
        a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class AuditRunner(GraphConcurrentModelRunner):
    def __init__(self,*args,**kwargs):
        self.audit_records = []
        self.audit_states = []
        self.audit_enabled = False
        self.audit_veto = None
        super().__init__(*args,**kwargs)

    @torch.inference_mode()
    def graph_control(self,operation,payload=None):
        if operation=="audit":
            self.audit_enabled = payload.get("enabled",True)
            self.audit_records = []
            self.audit_states = []
            self.audit_veto = payload.get("veto")
            dist.barrier(group=self.control_group)
        elif operation=="audit_snapshot":
            value = self.graph_cache.consensus(dict(rank=self.rank,checks=self.audit_records,states=self.audit_states))
            dist.barrier(group=self.control_group)
            return value
        else:
            return super().graph_control(operation,payload)

    @torch.inference_mode()
    def eagle3_batch(self,operation,payload):
        if not self.audit_enabled:
            return super().eagle3_batch(operation,payload)
        cache = self.graph_cache
        original = cache.forward
        original_consensus = cache.consensus

        def consensus(value):
            if self.rank==1 and isinstance(value,dict) and "available" in value:
                value = dict(value)
                if self.audit_veto=="missing":
                    value["available"] = False
                if self.audit_veto=="key" and value["key"] is not None:
                    value["key"] = replace(value["key"],max_k=value["key"].max_k+1)
            return original_consensus(value)

        def audited(entries,states):
            blocks = torch.tensor(sorted({b for e in entries for b in e["blocks"]}),device=self.kv_cache.device)
            before = self.kv_cache.index_select(2,blocks).clone()
            eager = runtime._forward(self,entries,states,verification=True)
            expected_kv = self.kv_cache.index_select(2,blocks).clone()
            self.kv_cache.index_copy_(2,blocks,before)
            actual = original(entries,states)
            checks = dict(features=byte_equal(eager[1],actual[1]),
                kv_pages=byte_equal(expected_kv,self.kv_cache.index_select(2,blocks)),
                finite=actual[2],replicated=actual[3])
            if self.rank==0:
                checks.update(logits=byte_equal(eager[0],actual[0]),
                              target_ids=torch.equal(eager[0].argmax(-1),actual[0].argmax(-1)))
            flags = original_consensus(checks)
            self.audit_records.append(checks)
            if not all(all(row.values()) for row in flags):
                raise RuntimeError(f"Graph integration numerical/state disagreement: {flags}")
            return actual

        cache.consensus = consensus
        cache.forward = audited
        try:
            value = super().eagle3_batch(operation,payload)
        finally:
            cache.forward = original
            cache.consensus = original_consensus
        if operation in ("prefill","verify","commit","rollback"):
            rows = []
            for entry in payload["entries"]:
                state = self._eagle_batch_states[entry["seq_id"]]
                cursor = state["cursor"]
                positions = torch.arange(cursor,device=self.kv_cache.device)
                blocks = torch.tensor(state["blocks"],device=positions.device)
                valid = self.kv_cache[:,:,blocks[positions//self.block_size],positions%self.block_size]
                rows.append(dict(seq_id=entry["seq_id"],cursor=cursor,token_count=len(state["tokens"]),
                    feature_sha=sha(state["features"]),kv_sha=sha(valid),phase=state["phase"].name))
            self.audit_states.append(dict(operation=operation,rows=rows))
        return value
