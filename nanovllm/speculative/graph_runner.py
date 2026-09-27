"""Opt-in graph runner used only by ConcurrentLLMEngine."""
from datetime import timedelta

import torch
import torch.distributed as dist

from nanovllm.speculative.concurrent_engine import ConcurrentModelRunner
from nanovllm.speculative.batched_runtime import dispatch
from nanovllm.speculative.graph_cache import GraphCache


class GraphConcurrentModelRunner(ConcurrentModelRunner):
    def __init__(self,*args,**kwargs):
        self.graph_cache = None
        self.control_group = None
        super().__init__(*args,**kwargs)

    def eagle3_batch(self, operation, payload):
        forward = self.graph_cache.forward if self.graph_cache is not None else None
        return dispatch(self,operation,payload,verification_forward=forward)

    @torch.inference_mode()
    def graph_control(self, operation, payload=None):
        payload = payload or {}
        result = None
        if operation=="init":
            self.control_group = dist.new_group(backend="gloo",timeout=timedelta(seconds=120))
            self.capture_stream = torch.cuda.Stream()
            self.graph_cache = GraphCache(self,payload)
        elif operation=="reset":
            self.graph_cache.clear()
            self.graph_cache.configure(payload.get("config",self.graph_cache.config))
            self.graph_cache.enabled = payload.get("enabled",True)
        elif operation=="clear":
            self.graph_cache.clear()
        elif operation=="snapshot":
            result = self.graph_cache.consensus(self.graph_cache.snapshot())
        else:
            raise ValueError(operation)
        # The original RPC transport is a single-slot mailbox.
        dist.barrier(group=self.control_group)
        return result

    def exit(self):
        if self.graph_cache is not None:
            self.graph_cache.clear()
            self.graph_cache = None
        if self.control_group is not None:
            dist.destroy_process_group(self.control_group)
            self.control_group = None
        super().exit()
