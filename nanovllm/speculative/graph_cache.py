"""Bounded, exact-context target graphs; no request/transaction ownership."""
from collections import deque
from dataclasses import asdict
import gc
import hashlib
import json
from time import perf_counter_ns

import torch
import torch.distributed as dist

from nanovllm.speculative import batched_runtime as runtime
from nanovllm.speculative.graph_entry import GraphEntry
from nanovllm.speculative.graph_policy import BoundedPolicy, GraphCacheConfig, agree, make_key


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


class GraphCache:
    def __init__(self, runner, config):
        self.runner = runner
        self.entries = {}
        self.epoch = 0
        self.enabled = True
        self.model_identity = digest(runner.config.hf_config.to_dict())
        self.configure(config)

    def configure(self, config):
        if self.entries:
            raise RuntimeError("Release graph entries before reconfiguration")
        self.config = config if isinstance(config,GraphCacheConfig) else GraphCacheConfig(**config)
        self.policy = BoundedPolicy(self.config)
        self.events = deque(maxlen=self.config.max_event_records)
        self.dropped_events = 0
        self.lifetimes = []
        self.epoch += 1
        self.peak_allocated = self.peak_reserved = 0
        self.peak_live_bytes = self.peak_live_reserved = 0

    def memory(self):
        return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                    peak_allocated=max(self.peak_allocated,torch.cuda.max_memory_allocated()),
                    peak_reserved=max(self.peak_reserved,torch.cuda.max_memory_reserved()))

    def emit(self, event):
        if len(self.events)==self.events.maxlen:
            self.dropped_events += 1
        self.events.append(event)

    def consensus(self, value):
        values = [None]*self.runner.world_size
        dist.all_gather_object(values,value,group=self.runner.control_group)
        return values

    def layout_key(self, entries, states):
        runner = self.runner
        layout = runtime.verification_layout(entries,
            {e["seq_id"]:s for e,s in zip(entries,states)},runner.block_size)
        cfg = runner.config.hf_config
        key = make_key(layout,block_size=runner.block_size,model_identity=self.model_identity,
            dtype=str(cfg.dtype),device_class=str(torch.cuda.get_device_capability()),
            tp_size=runner.world_size,training=runner.model.training,
            feature_layers=(2,cfg.num_hidden_layers//2,cfg.num_hidden_layers-3))
        return layout,key

    def release_key(self, key, reason):
        dist.barrier(group=self.runner.control_group)
        before = self.memory()
        obj,meta = self.entries.pop(key)
        start = perf_counter_ns()
        obj.release()
        del obj
        self.policy.lru.pop(key)
        gc.collect()
        torch.cuda.empty_cache()
        meta.update(released_step=self.policy.step,release_reason=reason,
                    release_ns=perf_counter_ns()-start,memory_after_release=self.memory())
        self.emit(dict(kind="release",key=digest(asdict(key)),reason=reason,
                       before=before,after=self.memory(),entry_id=meta["entry_id"],
                       duration_ns=meta["release_ns"]))

    @torch.inference_mode()
    def clear(self):
        for key in list(self.policy.lru):
            self.release_key(key,"clear")

    def snapshot(self):
        return dict(rank=self.runner.rank,epoch=self.epoch,config=asdict(self.config),
            events=list(self.events),lifetimes=self.lifetimes,dropped_events=self.dropped_events,
            live_entries=len(self.entries),seen_keys=len(self.policy.seen),memory=self.memory(),
            peak_live_bytes=self.peak_live_bytes,peak_live_reserved=self.peak_live_reserved,
            target_states=len(getattr(self.runner,"_eagle_batch_states",{})))

    def capture(self, key, layout):
        runner = self.runner
        victim = self.policy.victim()
        if victim is not None:
            self.release_key(victim,"eviction")
        before = self.memory()
        self.peak_allocated = before["peak_allocated"]
        self.peak_reserved = before["peak_reserved"]
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter_ns()
        slots = torch.tensor(layout["slot_mapping"],dtype=torch.long,device=runner.kv_cache.device)
        physical,offsets = slots//runner.block_size,slots%runner.block_size
        saved = runner.kv_cache[:,:,physical,offsets].clone()
        obj = GraphEntry(runner,key,layout)
        obj.capture(layout)
        runner.kv_cache[:,:,physical,offsets] = saved
        torch.cuda.synchronize()
        peak = dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        self.peak_allocated = max(self.peak_allocated,peak["allocated"])
        self.peak_reserved = max(self.peak_reserved,peak["reserved"])
        del saved,slots,physical,offsets
        after = self.memory()
        meta = dict(entry_id=len(self.lifetimes),key=asdict(key),key_id=digest(asdict(key)),
            created_step=self.policy.step,last_used=self.policy.step,capture_count=1,replay_count=0,
            warmup_target_forwards=3,capture_recordings=1,
            capture_ns=perf_counter_ns()-start,allocated_delta=after["allocated"]-before["allocated"],
            reserved_delta=after["reserved"]-before["reserved"],capture_peak=peak,
            before=before,after=after,addresses=obj.addresses)
        self.entries[key] = (obj,meta)
        self.policy.insert(key)
        self.lifetimes.append(meta)
        self.peak_live_bytes = max(self.peak_live_bytes,sum(m["allocated_delta"] for _,m in self.entries.values()))
        self.peak_live_reserved = max(self.peak_live_reserved,sum(max(0,m["reserved_delta"]) for _,m in self.entries.values()))
        return meta

    @torch.inference_mode()
    def forward(self, entries, states):
        if not self.enabled:
            return runtime._forward(self.runner,entries,states,verification=True)
        start = perf_counter_ns()
        error = None
        layout = key = None
        try:
            layout,key = self.layout_key(entries,states)
        except Exception as exc:
            error = repr(exc)
        occurrence = self.policy.observe(key)
        available = key in self.entries
        plan = dict(key=key,layout_digest=digest((entries,layout)),epoch=self.epoch,
            policy=asdict(self.config),available=available,capture=self.policy.should_capture(key),
            registry=list(self.policy.lru),error=error)
        plans = self.consensus(plan)
        action,reason = agree(plans)
        agreement_ns = perf_counter_ns()-start
        event = dict(kind="verify",step=self.policy.step,seq_ids=[e["seq_id"] for e in entries],
            key=asdict(key) if key else None,key_id=digest(asdict(key)) if key else None,
            occurrence=occurrence,action=action,reason=reason,agreement_ns=agreement_ns,
            q_lengths=[1+len(e["proposals"]) for e in entries],
            contexts=layout["context_lens"],capture_ns=0,replay_ns=0,eager_ns=0)
        if action=="replay":
            obj,meta = self.entries[key]
            if obj.kv_address!=self.runner.kv_cache.data_ptr():
                raise RuntimeError("KV allocation changed while graph is live")
            begin = perf_counter_ns()
            result = obj.replay(layout,states)
            event["replay_ns"] = perf_counter_ns()-begin
            event["entry_id"] = meta["entry_id"]
            meta["replay_count"] += 1
            meta["last_used"] = self.policy.step
            self.policy.touch(key)
        else:
            begin = perf_counter_ns()
            result = runtime._forward(self.runner,entries,states,verification=True)
            event["eager_ns"] = perf_counter_ns()-begin
            if action=="capture":
                meta = self.capture(key,layout)
                event.update(capture_ns=meta["capture_ns"],entry_id=meta["entry_id"])
        event.update(total_ns=perf_counter_ns()-start,live_entries=len(self.entries),memory=self.memory())
        self.emit(event)
        return result
