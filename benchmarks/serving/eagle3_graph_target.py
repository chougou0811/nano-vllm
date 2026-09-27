"""Benchmark-only fixed-shape target graph; never imported by production."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import torch
import torch.distributed as dist

from nanovllm.speculative.concurrent_engine import ConcurrentModelRunner
import nanovllm.speculative.batched_runtime as runtime
from nanovllm.utils.context import set_context, get_context, reset_context
from benchmarks.serving.eagle3_graph_key import agree, make_key


FROZEN_FORWARD = runtime._forward


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def identical(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))


class StaticTarget:
    def __init__(self, runner, key, layout):
        self.runner = runner
        self.key = key
        self.graph = None
        # cuBLAS retains a workspace per stream. Share one engine-owned capture
        # stream so graph-key turnover cannot grow that process-lifetime cache.
        self.stream = runner.capture_stream
        self.device = runner.kv_cache.device
        self.host, self.inputs = {}, {}
        for name in ("input_ids","positions","cu_seqlens_q","cu_seqlens_k","slot_mapping","block_tables"):
            dtype = torch.int64 if name in ("input_ids","positions") else torch.int32
            example = torch.tensor(layout[name],dtype=dtype)
            self.host[name] = torch.empty_like(example,pin_memory=True)
            self.inputs[name] = torch.empty(example.shape,dtype=dtype,device=self.device)
        cfg = runner.config.hf_config
        self.hidden = torch.empty(key.rows,cfg.hidden_size,dtype=cfg.dtype,device=self.device)
        self.features = torch.empty(key.rows,3*cfg.hidden_size,dtype=cfg.dtype,device=self.device)
        self.local_logits = torch.empty(key.rows,cfg.vocab_size//runner.world_size,dtype=cfg.dtype,device=self.device)
        self.logits = (torch.empty(key.rows,cfg.vocab_size,dtype=cfg.dtype,device=self.device)
                       if runner.rank == 0 else None)
        self.capture_count = 0
        self.replays = 0
        self.addresses = self.pointers()
        self.weight_addresses = tuple(p.data_ptr() for p in runner.model.parameters())
        self.kv_address = runner.kv_cache.data_ptr()

    def pointers(self):
        return {name: value.data_ptr() for name,value in dict(
            self.inputs,hidden=self.hidden,features=self.features,
            local_logits=self.local_logits,logits=self.logits).items() if value is not None}

    def update(self, layout):
        if self.pointers() != self.addresses or self.runner.kv_cache.data_ptr()!=self.kv_address:
            raise RuntimeError("Graph static address changed")
        for name,value in self.inputs.items():
            source = torch.tensor(layout[name],dtype=value.dtype)
            if source.shape != value.shape:
                raise RuntimeError("Graph metadata shape changed")
            self.host[name].copy_(source)
            value.copy_(self.host[name],non_blocking=True)

    def _core(self):
        x = self.inputs
        set_context(True,cu_seqlens_q=x["cu_seqlens_q"],cu_seqlens_k=x["cu_seqlens_k"],
                    max_seqlen_q=self.key.max_q,max_seqlen_k=self.key.max_k,
                    slot_mapping=x["slot_mapping"],block_tables=x["block_tables"])
        captured, handles = [], []

        def capture(module, args):
            _, hidden, residual = args
            captured.append(hidden.clone() if residual is None else
                            (hidden.float()+residual.float()).to(hidden.dtype))

        original_gather = dist.gather

        def gather(tensor, *args, **kwargs):
            self.local_logits.copy_(tensor)
            return original_gather(tensor,*args,**kwargs)

        try:
            for index in self.key.feature_layers:
                handles.append(self.runner.model.model.layers[index].register_forward_pre_hook(capture))
            self.hidden.copy_(self.runner.model(x["input_ids"],x["positions"]))
            get_context().is_prefill = False
            dist.gather = gather
            logits = self.runner.model.compute_logits(self.hidden)
            if logits is not None:
                self.logits.copy_(logits)
            self.features.copy_(torch.cat(captured,dim=-1))
        finally:
            dist.gather = original_gather
            for handle in handles:
                handle.remove()
            reset_context()

    def capture(self, layout):
        self.update(layout)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self._core()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        dist.barrier(group=self.runner.control_group)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=self.stream):
            self._core()
        torch.cuda.synchronize()
        self.capture_count += 1
        if tuple(p.data_ptr() for p in self.runner.model.parameters()) != self.weight_addresses:
            raise RuntimeError("Model storage changed during capture")

    def replay(self, layout, states):
        self.update(layout)
        self.graph.replay()
        self.replays += 1
        # Static output storage must never become persistent request-owned state.
        features = self.features.clone()
        logits = self.logits
        finite = torch.isfinite(features).all()
        if self.runner.rank == 0:
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

    @torch.inference_mode()
    def release(self):
        torch.cuda.synchronize()
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
        for value in (*self.inputs.values(),self.hidden,self.features,self.local_logits,self.logits):
            if value is not None:
                value.zero_()
        torch.cuda.synchronize()
        self.runner = None


class GraphRunner(ConcurrentModelRunner):
    def __init__(self,*args,**kwargs):
        self.control_group = None
        self.fixed_graphs = {}
        self.mode = "eager"
        self.epoch = 0
        self.verify_index = 0
        self.capture_indices = {0,2}
        self.audit_records = []
        self.timing_records = []
        self.pending_events = []
        self.decisions = []
        self.trace = None
        self.measure = False
        self.mask_rank1 = False
        self.graph_keys_log = []
        super().__init__(*args,**kwargs)

    def consensus(self, local):
        values = [None]*self.world_size
        dist.all_gather_object(values,local,group=self.control_group)
        return values

    def layout_key(self, entries, states):
        layout = runtime.verification_layout(entries,
            {entry["seq_id"]:state for entry,state in zip(entries,states)},self.block_size)
        cfg = self.config.hf_config
        key = make_key(layout,block_size=self.block_size,
                       model_identity=self.model_identity,dtype=str(cfg.dtype),
                       device_class=str(torch.cuda.get_device_capability()),tp_size=self.world_size,
                       training=self.model.training,feature_layers=(2,cfg.num_hidden_layers//2,cfg.num_hidden_layers-3))
        return layout,key

    def select(self, entries, states, create=False):
        error = None
        layout = key = None
        try:
            layout,key = self.layout_key(entries,states)
        except Exception as exc:
            error = repr(exc)
        plans = self.consensus(dict(key=key,layout_digest=digest((entries,layout)),
            mode="eager" if self.mode=="coordinated-eager" else self.mode,epoch=self.epoch,error=error,
            available=key in self.fixed_graphs and not(self.mask_rank1 and self.rank==1)))
        selected = agree(plans)
        if create and key is not None and not all(p["available"] for p in plans):
            if len(self.fixed_graphs)>=2:
                raise RuntimeError("Prototype bounded graph registry exceeded")
            obj = StaticTarget(self,key,layout)
            obj.capture(layout)
            self.fixed_graphs[key] = obj
            self.graph_keys_log.append(dict(key=asdict(key),addresses=obj.addresses,
                                             capture_epoch=self.epoch))
            selected = "graph"
        self.decisions.append(dict(index=self.verify_index,epoch=self.epoch,key=asdict(key) if key else None,
                                   selected=selected,availability=[p["available"] for p in plans]))
        return layout,key,selected

    @torch.inference_mode()
    def audited_forward(self, entries, states):
        layout,key,selected = self.select(entries,states)
        create = self.verify_index in self.capture_indices and key is not None
        if selected != "graph" and not create:
            return FROZEN_FORWARD(self,entries,states,verification=True)
        blocks = torch.tensor(sorted({b for e in entries for b in e["blocks"]}),device=self.kv_cache.device)
        before = self.kv_cache.index_select(2,blocks).clone()
        local = []
        original_gather = dist.gather

        def gather(tensor,*args,**kwargs):
            local.append(tensor.clone())
            return original_gather(tensor,*args,**kwargs)

        dist.gather = gather
        try:
            eager = FROZEN_FORWARD(self,entries,states,verification=True)
        finally:
            dist.gather = original_gather
        eager_kv = self.kv_cache.index_select(2,blocks).clone()
        if create and key not in self.fixed_graphs:
            self.kv_cache.index_copy_(2,blocks,before)
            layout,key,selected = self.select(entries,states,create=True)
        obj = self.fixed_graphs[key]
        cpu_before = [(s["cursor"],tuple(s["tokens"]),tuple(s["blocks"]),s["phase"].name) for s in states]
        checks = []
        for replay in range(3):
            self.kv_cache.index_copy_(2,blocks,before)
            graph = obj.replay(layout,states)
            kv = self.kv_cache.index_select(2,blocks)
            check = dict(replay=replay,features=identical(eager[1],graph[1]),
                         local_logits=identical(local[0],obj.local_logits),
                         kv_bytes=identical(eager_kv,kv),finite=graph[2],replicated=graph[3],
                         addresses=obj.pointers()==obj.addresses,
                         feature_ownership=graph[1].data_ptr()!=obj.features.data_ptr(),
                         cpu_state=cpu_before==[(s["cursor"],tuple(s["tokens"]),tuple(s["blocks"]),s["phase"].name) for s in states])
            if self.rank == 0:
                check.update(logits=identical(eager[0],graph[0]),
                             target_ids=torch.equal(eager[0].argmax(-1),graph[0].argmax(-1)))
            flags = {k:v for k,v in check.items() if k != "replay"}
            all_checks = self.consensus(flags)
            if not all(all(row.values()) for row in all_checks):
                torch.save(dict(eager_logits=eager[0],graph_logits=graph[0],
                    eager_features=eager[1],graph_features=graph[1],
                    eager_local_logits=local[0],graph_local_logits=obj.local_logits,
                    kv_before=before,eager_kv=eager_kv,graph_kv=kv),
                    Path(self.raw,f"disagreement-rank{self.rank}-step{self.verify_index}.pt"))
                self.audit_records.append(dict(key=asdict(key),failed=all_checks))
                self.save_records()
                raise RuntimeError(f"Same-shape graph correctness disagreement: {all_checks}")
            checks.append(check)
        self.audit_records.append(dict(key=asdict(key),ordered_seq_ids=[e["seq_id"] for e in entries],
            contexts=layout["context_lens"],block_tables=layout["block_tables"],checks=checks,
            features_sha256=tensor_digest(graph[1]),local_logits_sha256=tensor_digest(obj.local_logits),
            kv_sha256=tensor_digest(eager_kv)))
        return graph

    def experimental_forward(self, entries, states):
        if self.mode == "audit":
            return self.audited_forward(entries,states)
        layout,key,selected = self.select(entries,states)
        if selected == "graph":
            return self.fixed_graphs[key].replay(layout,states)
        return FROZEN_FORWARD(self,entries,states,verification=True)

    def eagle3_batch(self, operation, payload):
        original = runtime._forward
        if operation == "verify" and self.mode != "eager":
            def forward(runner,entries,states,*,verification):
                if runner is self and verification:
                    return self.experimental_forward(entries,states)
                return original(runner,entries,states,verification=verification)
            runtime._forward = forward
        row = None
        try:
            if operation == "verify" and self.measure:
                row = dict(index=len(self.timing_records),mode=self.mode,start_ns=perf_counter_ns(),
                           rows=sum(1+len(e["proposals"]) for e in payload["entries"]))
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.profiler.record_function(f"diag.target.{row['index']}") if self.trace else nullcontext():
                    start.record()
                    value = super().eagle3_batch(operation,payload)
                    end.record()
                row["end_ns"] = perf_counter_ns()
                self.timing_records.append(row)
                self.pending_events.append((row,start,end))
            else:
                value = super().eagle3_batch(operation,payload)
            return value
        except Exception as exc:
            self.last_error = repr(exc)
            if self.rank == 0:
                raise
            # Coordinated audit exceptions must not kill the worker RPC loop.
            return None
        finally:
            runtime._forward = original
            if operation == "verify":
                self.verify_index += 1

    def save_records(self):
        if not hasattr(self,"raw"):
            return
        Path(self.raw,f"rank{self.rank}.json").write_text(json.dumps(dict(
            rank=self.rank,audits=self.audit_records,decisions=self.decisions,keys=self.graph_keys_log,
            timings=self.timing_records,error=getattr(self,"last_error",None)),indent=2,default=str))

    @torch.inference_mode()
    def graph_control(self, operation, payload=None):
        # The frozen RPC mailbox has one slot, not a queue. Acknowledge every
        # prototype control command before rank 0 may publish the next one.
        result = self._graph_control(operation,payload)
        dist.barrier(group=self.control_group)
        return result

    def _graph_control(self, operation, payload=None):
        payload = payload or {}
        if operation == "init":
            self.control_group = dist.new_group(backend="gloo",timeout=timedelta(seconds=120))
            self.capture_stream = torch.cuda.Stream()
            self.raw = payload["raw"]
            self.model_identity = digest(self.config.hf_config.to_dict())
            self.base_allocated = torch.cuda.memory_allocated()
        elif operation == "mode":
            self.mode = payload["mode"]
            self.measure = payload.get("measure",False)
            self.mask_rank1 = payload.get("mask_rank1",False)
            if payload.get("reset_index"):
                self.verify_index = 0
                self.epoch += 1
        elif operation == "prepare":
            entries = payload["entries"]
            states = runtime._resolve(self,entries)
            blocks = torch.tensor(sorted({b for e in entries for b in e["blocks"]}),device=self.kv_cache.device)
            snapshot = self.kv_cache.index_select(2,blocks).clone()
            self.select(entries,states,create=True)
            self.kv_cache.index_copy_(2,blocks,snapshot)
        elif operation == "flush":
            torch.cuda.synchronize()
            for row,start,end in self.pending_events:
                row["gpu_event_ms"] = start.elapsed_time(end)
            self.pending_events.clear()
            self.save_records()
        elif operation == "trace_start":
            torch.cuda.synchronize()
            self.trace_path = payload["path"]
            self.trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False)
            self.trace.start()
            dist.barrier(group=self.control_group)
        elif operation == "trace_stop":
            dist.barrier(group=self.control_group)
            self.trace.stop()
            self.trace.export_chrome_trace(self.trace_path+f"-rank{self.rank}.json")
            self.trace = None
        elif operation == "clear":
            torch.cuda.synchronize()
            while self.fixed_graphs:
                _,obj = self.fixed_graphs.popitem()
                obj.release()
                del obj
            gc.collect()
            torch.cuda.empty_cache()
        elif operation == "cleanup":
            result = dict(rank=self.rank,targets=len(getattr(self,"_eagle_batch_states",{})),
                          graphs=len(self.fixed_graphs),allocated=torch.cuda.memory_allocated(),
                          reserved=torch.cuda.memory_reserved())
            return self.consensus(result)
        else:
            raise ValueError(operation)

    def exit(self):
        if self.fixed_graphs:
            for obj in self.fixed_graphs.values():
                obj.release()
            self.fixed_graphs.clear()
        gc.collect()
        if self.control_group is not None:
            dist.destroy_process_group(self.control_group)
            self.control_group = None
        super().exit()
