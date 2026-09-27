"""Fixed-shape tensor core promoted from the frozen Phase 4.4A prototype."""
import torch
import torch.distributed as dist
from nanovllm.utils.context import set_context, get_context, reset_context


class GraphEntry:
    def __init__(self, runner, key, layout):
        self.model = runner.model
        self.rank = runner.rank
        self.kv_cache = runner.kv_cache
        self.control_group = runner.control_group
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
        if self.pointers() != self.addresses or self.kv_cache.data_ptr()!=self.kv_address:
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
                handles.append(self.model.model.layers[index].register_forward_pre_hook(capture))
            self.hidden.copy_(self.model(x["input_ids"],x["positions"]))
            get_context().is_prefill = False
            dist.gather = gather
            logits = self.model.compute_logits(self.hidden)
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
        dist.barrier(group=self.control_group)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=self.stream):
            self._core()
        torch.cuda.synchronize()
        self.capture_count += 1
        if tuple(p.data_ptr() for p in self.model.parameters()) != self.weight_addresses:
            raise RuntimeError("Model storage changed during capture")

    def replay(self, layout, states):
        self.update(layout)
        self.graph.replay()
        self.replays += 1
        # Static output storage must never become persistent request-owned state.
        features = self.features.clone()
        logits = self.logits
        finite = torch.isfinite(features).all()
        if self.rank == 0:
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
        self.model = None
        self.kv_cache = None
        self.control_group = None
