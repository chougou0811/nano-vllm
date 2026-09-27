"""Opt-in research runner; frozen inference modules are never edited."""
from contextlib import contextmanager, nullcontext
from datetime import timedelta
import json
from pathlib import Path
from time import perf_counter_ns

import torch
import torch.distributed as dist

from nanovllm.speculative.concurrent_engine import ConcurrentModelRunner
from nanovllm.speculative import batched_runtime as runtime
from nanovllm.speculative.graph_cache import GraphCache
from benchmarks.serving.eagle3_graph_target import identical


def eligible(entries):
    return len(entries) in (1, 2, 4) and all(len(e["proposals"]) == 3 for e in entries)


class MLPGraph:
    def __init__(self, forward, rows, width, device, dtype, stream):
        self.forward = forward
        self.input = torch.zeros((rows, width), device=device, dtype=dtype)
        self.graph = None
        self.output = None
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output = forward(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = forward(self.input)
        torch.cuda.synchronize()

    def __call__(self, value):
        if value.shape != self.input.shape or value.dtype != self.input.dtype:
            raise RuntimeError("Unexpected MLP graph input")
        self.input.copy_(value)
        self.graph.replay()
        return self.output

    def close(self):
        if self.graph is not None:
            self.graph.reset()
        self.graph = self.output = self.input = self.forward = None


class Phase51Runner(ConcurrentModelRunner):
    def __init__(self, *args, **kwargs):
        self.mode = "eager"
        self.control_group = None
        self.graph_cache = None
        self.regions = {}
        self.trace = None
        self.records = []
        self.audits = []
        self.measure = False
        self.layer_count = 0
        self.raw = None
        self.startup = []
        super().__init__(*args, **kwargs)

    def consensus(self, value):
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.control_group)
        return values

    def memory(self):
        free, total = torch.cuda.mem_get_info()
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), free=free, total=total)

    @contextmanager
    def mlp_scope(self, entries):
        originals = []
        selected = []
        if eligible(entries):
            rows = 4 * len(entries)
            for index, layer in enumerate(self.model.model.layers[:self.layer_count]):
                selected.append((layer.mlp, self.regions[(index, rows)]))
        try:
            for module, graph in selected:
                layer_forward = module.forward
                originals.append((module, layer_forward))
                module.forward = graph
            yield
        finally:
            for module, original in originals:
                module.forward = original

    def region_forward(self, entries, states):
        with self.mlp_scope(entries):
            return runtime._forward(self, entries, states, verification=True)

    def audited_forward(self, entries, states):
        blocks = torch.tensor(sorted({b for e in entries for b in e["blocks"]}),
                              device=self.kv_cache.device)
        before = self.kv_cache.index_select(2, blocks).clone()
        baseline = runtime._forward(self, entries, states, verification=True)
        after = self.kv_cache.index_select(2, blocks).clone()
        checks = []
        for repeat in range(3):
            self.kv_cache.index_copy_(2, blocks, before)
            candidate = self.region_forward(entries, states)
            row = dict(features=identical(baseline[1], candidate[1]),
                       kv=identical(after, self.kv_cache.index_select(2, blocks)),
                       finite=candidate[2], replicated=candidate[3])
            if self.rank == 0:
                row["logits"] = identical(baseline[0], candidate[0])
            ranks = self.consensus(row)
            checks.append(ranks)
            if not all(all(r.values()) for r in ranks):
                self.audits.append(dict(passed=False, ranks=ranks))
                self.save()
                raise RuntimeError("Phase51 same-shape numerical mismatch")
        self.audits.append(dict(passed=True, rows=sum(1+len(e["proposals"]) for e in entries),
                                eligible=eligible(entries), repeats=checks))
        return candidate

    def forward(self, entries, states):
        if self.mode == "audit":
            return self.audited_forward(entries, states)
        if self.mode == "mlp":
            return self.region_forward(entries, states)
        if self.mode in ("wrapper", "miss"):
            self.graph_cache.enabled = self.mode == "miss"
            return self.graph_cache.forward(entries, states)
        if self.mode == "shadow":
            self.graph_cache.layout_key(entries, states)
        if self.mode == "agreement":
            self.consensus({"mode": "agreement", "entries": entries})
        return runtime._forward(self, entries, states, verification=True)

    def eagle3_batch(self, operation, payload):
        callback = None if self.mode == "eager" else self.forward
        measured = self.measure and operation == "verify"
        if measured:
            start = perf_counter_ns()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
        try:
            with torch.profiler.record_function("phase51.verify") if self.trace else nullcontext():
                result = runtime.dispatch(self, operation, payload, verification_forward=callback)
        except RuntimeError as error:
            if "Phase51 same-shape numerical mismatch" not in str(error):
                raise
            if self.rank == 0:
                raise
            return None
        if measured:
            b.record()
            b.synchronize()
            self.records.append(dict(mode=self.mode, start_ns=start, end_ns=perf_counter_ns(),
                                     cuda_ms=a.elapsed_time(b), rank=self.rank))
        return result

    @torch.inference_mode()
    def phase51_control(self, operation, payload=None):
        payload = payload or {}
        result = None
        if operation == "init":
            self.control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
            self.capture_stream = torch.cuda.Stream()
            self.graph_cache = GraphCache(self, dict(capture_policy="never", max_captures=0))
            self.raw = Path(payload["raw"])
        elif operation == "mode":
            if payload["mode"] not in ("eager", "wrapper", "shadow", "agreement", "miss", "mlp", "audit"):
                raise ValueError("Unknown phase51 mode")
            self.mode = payload["mode"]
            self.measure = payload.get("measure", False)
            self.layer_count = payload.get("layers", len(self.model.model.layers))
        elif operation == "prepare":
            before = self.memory()
            start = perf_counter_ns()
            cfg = self.config.hf_config
            for index, layer in enumerate(self.model.model.layers[:payload.get("layers", len(self.model.model.layers))]):
                for rows in (4, 8, 16):
                    if (index, rows) not in self.regions:
                        self.regions[index, rows] = MLPGraph(layer.mlp.forward, rows, cfg.hidden_size,
                            self.kv_cache.device, self.kv_cache.dtype, self.capture_stream)
            result = self.consensus(dict(rank=self.rank, before=before, after=self.memory(),
                                         duration_ns=perf_counter_ns()-start, entries=len(self.regions)))
            self.startup.append(result)
        elif operation == "old_capture":
            entries = payload["entries"]
            states = runtime._resolve(self, entries)
            layout, key = self.graph_cache.layout_key(entries, states)
            if key is None or self.graph_cache.entries:
                raise RuntimeError("Lifecycle diagnostic requires one eligible, empty cache")
            self.graph_cache.capture(key, layout)
        elif operation == "old_release":
            self.graph_cache.clear()
        elif operation == "snapshot":
            result = self.consensus(dict(rank=self.rank, memory=self.memory(), entries=len(self.regions),
                targets=len(getattr(self, "_eagle_batch_states", {})), records=self.records,
                audits=self.audits, startup=self.startup, graph_events=list(self.graph_cache.events)))
        elif operation == "trace_start":
            self.trace_path = payload["path"]
            self.trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
            self.trace.__enter__()
        elif operation == "trace_stop":
            self.trace.__exit__(None, None, None)
            self.trace.export_chrome_trace(f"{self.trace_path}-rank{self.rank}.json")
            self.trace = None
        elif operation == "clear":
            torch.cuda.synchronize()
            for entry in self.regions.values():
                entry.close()
            self.regions.clear()
        else:
            raise ValueError(operation)
        self.save()
        dist.barrier(group=self.control_group)
        return result

    def save(self):
        if self.raw is not None:
            (self.raw/f"rank{self.rank}.json").write_text(json.dumps(dict(
                timings=self.records, audits=self.audits, startup=self.startup), indent=2))

    def exit(self):
        torch.cuda.synchronize()
        for entry in self.regions.values():
            entry.close()
        self.regions.clear()
        if self.graph_cache is not None:
            self.graph_cache.clear()
        if self.control_group is not None:
            dist.destroy_process_group(self.control_group)
        super().exit()
