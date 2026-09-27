"""Diagnostic-only dual-rank target profiling; no production file changes."""
import argparse
import atexit
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter_ns
import traceback

import torch
import torch.distributed as dist

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.speculative.concurrent_engine import ConcurrentModelRunner
from benchmarks.serving.eagle3_phase42 import (
    TARGET, DRAFT, REFERENCE, build_requests, cleanup_state,
)


def category(name, module):
    if name == "lm_head":
        return "lm_head"
    if name.endswith("embed_tokens"):
        return "embedding"
    for suffix, label in (("qkv_proj", "qkv"), ("o_proj", "output_projection"),
                          ("gate_up_proj", "mlp_gate_up"), ("down_proj", "mlp_down"),
                          ("rotary_emb", "rope"), ("act_fn", "norm_elementwise")):
        if name.endswith(suffix):
            return label
    if type(module).__name__ == "RMSNorm":
        return "norm_elementwise"
    return None


class DiagnosticRunner(ConcurrentModelRunner):
    def __init__(self, *args, **kwargs):
        self.diag_active = False
        self.in_target = False
        self.prof = None
        self.hooks = []
        self.patches = []
        self.module_stack = []
        super().__init__(*args, **kwargs)

    def span(self, label):
        return torch.profiler.record_function(label) if self.prof else nullcontext()

    def _patch(self, obj, name, replacement):
        self.patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def _install_hooks(self):
        for name, module in self.model.named_modules():
            label = category(name, module)
            if not label:
                continue

            def before(mod, inputs, _name=name, _label=label):
                if not self.in_target:
                    return
                ctx = self.span(f"diag.module::{_label}::{_name}")
                ctx.__enter__()
                self.module_stack.append((_name, ctx))
                x = next((x for x in inputs if isinstance(x, torch.Tensor)), None)
                weight = getattr(mod, "weight", None)
                if _label in ("qkv", "output_projection", "mlp_gate_up", "mlp_down", "lm_head"):
                    self.shapes.append(dict(target=self.target_index, name=_name,
                                            category=_label, input=list(x.shape),
                                            weight=list(weight.shape), dtype=str(x.dtype)))

            def after(mod, inputs, output):
                if self.in_target:
                    _, ctx = self.module_stack.pop()
                    ctx.__exit__(None, None, None)

            self.hooks.append(module.register_forward_pre_hook(before))
            self.hooks.append(module.register_forward_hook(after, always_call=True))

        import nanovllm.layers.attention as attention
        for name, label in (("store_kvcache", "kv_store"),
                            ("flash_attn_varlen_func", "attention_varlen"),
                            ("flash_attn_with_kvcache", "attention_decode")):
            original = getattr(attention, name)

            def wrapped(*args, _fn=original, _label=label, **kwargs):
                with self.span(f"diag.module::{_label}::{_label}") if self.in_target else nullcontext():
                    return _fn(*args, **kwargs)

            self._patch(attention, name, wrapped)

        for name in ("all_reduce", "gather", "all_gather"):
            original = getattr(dist, name)

            def collective(*args, _fn=original, _name=name, **kwargs):
                if not self.in_target:
                    return _fn(*args, **kwargs)
                tensor = args[0] if _name != "all_gather" else args[1]
                row = dict(type=_name, target=self.target_index,
                           owner=self.module_stack[-1][0] if self.module_stack else "runtime",
                           shape=list(tensor.shape), bytes=tensor.numel() * tensor.element_size(),
                           dtype=str(tensor.dtype), op=str(kwargs.get("op", "SUM/default")),
                           start_ns=perf_counter_ns())
                with self.span(f"diag.collective::{_name}"):
                    value = _fn(*args, **kwargs)
                row["end_ns"] = perf_counter_ns()
                self.collectives.append(row)
                return value

            self._patch(dist, name, collective)

    def diagnostic(self, operation, config=None):
        if operation == "start":
            self.diag_config = config
            self.targets, self.events, self.rpc, self.shapes, self.collectives = [], [], [], [], []
            self.collective_events = []
            self.target_index = -1
            self.diag_active = True
            if config["profile"]:
                self.prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=config.get("rich", True), profile_memory=False, with_stack=False,
                )
                self.prof.start()
                if config.get("rich", True):
                    self._install_hooks()
            if config.get("collective_events"):
                self._install_collective_events()
            dist.barrier()
            torch.cuda.synchronize()
        elif operation == "stop":
            dist.barrier()
            torch.cuda.synchronize()
            self.diag_active = False
            for row, (start, end) in zip(self.targets, self.events):
                row["gpu_event_ms"] = start.elapsed_time(end)
            for row, start, end in self.collective_events:
                row["stream_elapsed_ms"] = start.elapsed_time(end)
            path = Path(self.diag_config["path"])
            if self.prof:
                self.prof.stop()
                self.prof.export_chrome_trace(str(path) + f"-rank{self.rank}.trace.json")
                self.prof = None
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            for obj, name, original in reversed(self.patches):
                setattr(obj, name, original)
            self.patches.clear()
            data = dict(rank=self.rank, gpu=torch.cuda.current_device(), targets=self.targets,
                        rpc=self.rpc, shapes=self.shapes, collectives=self.collectives,
                        allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
            Path(str(path) + f"-rank{self.rank}.json").write_text(json.dumps(data, indent=2))
            dist.barrier()
        elif operation == "cleanup":
            state = torch.tensor([len(getattr(self, "_eagle_batch_states", {}))], device="cuda")
            states = [torch.empty_like(state) for _ in range(self.world_size)]
            dist.all_gather(states, state)
            return [int(x.item()) for x in states]
        else:
            raise ValueError(operation)

    def _install_collective_events(self):
        # Events bracket stream dependencies, not pure network transfer time.
        for name in ("all_reduce", "gather", "all_gather"):
            original = getattr(dist, name)

            def measured(*args, _fn=original, _name=name, **kwargs):
                if not self.in_target:
                    return _fn(*args, **kwargs)
                tensor = args[0] if _name != "all_gather" else args[1]
                row = dict(type=_name, target=self.target_index, shape=list(tensor.shape),
                           bytes=tensor.numel() * tensor.element_size(), start_ns=perf_counter_ns())
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                value = _fn(*args, **kwargs)
                end.record()
                row["end_ns"] = perf_counter_ns()
                self.collectives.append(row)
                self.collective_events.append((row, start, end))
                return value

            self._patch(dist, name, measured)

    def _target(self, fn, rows, contexts):
        if not self.diag_active:
            return fn()
        self.target_index += 1
        row = dict(index=self.target_index, rows=rows, contexts=contexts,
                   start_ns=perf_counter_ns())
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.in_target = True
        try:
            with self.span(f"diag.target.{self.target_index}"):
                start.record()
                value = fn()
                end.record()
        finally:
            self.in_target = False
        row["end_ns"] = perf_counter_ns()
        self.targets.append(row)
        self.events.append((start, end))
        return value

    def eagle3_batch(self, operation, payload):
        if operation != "verify":
            return super().eagle3_batch(operation, payload)
        return self._target(
            lambda: super(DiagnosticRunner, self).eagle3_batch(operation, payload),
            sum(1 + len(item["proposals"]) for item in payload["entries"]),
            [self._eagle_batch_states[item["seq_id"]]["cursor"] for item in payload["entries"]],
        )

    def run(self, seqs, is_prefill):
        if is_prefill:
            return super().run(seqs, is_prefill)
        return self._target(lambda: super(DiagnosticRunner, self).run(seqs, is_prefill),
                            len(seqs), [len(seq) - 1 for seq in seqs])

    def write_shm(self, method_name, *args):
        start = perf_counter_ns()
        value = super().write_shm(method_name, *args)
        if self.diag_active:
            self.rpc.append(dict(kind="pack_dispatch", method=method_name,
                                 operation=args[0] if method_name == "eagle3_batch" else None,
                                 start_ns=start, end_ns=perf_counter_ns()))
        return value

    def read_shm(self):
        start = perf_counter_ns()
        value = super().read_shm()
        if self.diag_active:
            method, args = value
            self.rpc.append(dict(kind="wait_read_unpack", method=method,
                                 operation=args[0] if method == "eagle3_batch" else None,
                                 start_ns=start, end_ns=perf_counter_ns()))
        return value

    def prepare_prefill(self, seqs):
        with self.span("diag.module::input_preparation::prepare_prefill") if self.in_target else nullcontext():
            return super().prepare_prefill(seqs)

    def prepare_decode(self, seqs):
        with self.span("diag.module::input_preparation::prepare_decode") if self.in_target else nullcontext():
            return super().prepare_decode(seqs)


def source_hashes(root):
    paths = list((root / "nanovllm").rglob("*.py")) + list((root / "tests").glob("test_*.py"))
    paths += list((root / "benchmarks/eagle3-phase4").rglob("*"))
    paths += list((root / "benchmarks/eagle3-phase4_2").rglob("*"))
    paths += [root / "benchmarks/serving/eagle3_phase42.py",
              root / "benchmarks/serving/eagle3_phase42_summary.py"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file() and "__pycache__" not in str(path)}


def trial(engine, system, workload, concurrency, mode, raw, steps):
    if not engine.is_finished():
        raise RuntimeError("Trial must start idle")
    if system == "ordinary":
        old = engine.scheduler.block_manager
        if old.used_block_ids:
            raise RuntimeError("Cannot reset a live cache")
        engine.scheduler.block_manager = BlockManager(len(old.blocks), old.block_size)
    torch.manual_seed(4300)
    requests = build_requests(engine.tokenizer, workload, concurrency, 0)[:concurrency]
    seqs = []
    for request in requests:
        engine.add_request(request.prompt, SamplingParams(
            temperature=1e-9, max_tokens=request.output_limit, ignore_eos=True))
        seqs.append(engine.scheduler.waiting[-1])
    # c4 x 768 exceeds the unchanged 2048-token prefill budget: allow two
    # original prefill steps, then capture when the complete batch is active.
    while engine.scheduler.waiting:
        engine.step()
    if engine.scheduler.waiting or len(engine.scheduler.running) != concurrency:
        raise RuntimeError("Capture requires all requests active after prefill")
    profiled = mode in ("timing", "profile", "minimal", "collective-events")
    path = raw / f"{system}-c{concurrency}-{workload}-{mode}"
    if profiled:
        engine.model_runner.call("diagnostic", "start", dict(
            path=str(path), profile=mode in ("profile", "minimal"), rich=mode == "profile",
            collective_events=mode == "collective-events"))
    samples = []
    for _ in range(steps):
        start = perf_counter_ns()
        engine.step()
        samples.append(dict(host_step_ns=perf_counter_ns() - start,
                            outputs=[list(seq.completion_token_ids) for seq in seqs]))
    if profiled:
        engine.model_runner.call("diagnostic", "stop")
    while not engine.is_finished():
        engine.step()
    cleanup = cleanup_state(engine, system)
    rank_states = engine.model_runner.call("diagnostic", "cleanup")
    if any(cleanup.values()) or any(rank_states):
        raise RuntimeError(f"Cleanup failure: {cleanup}, {rank_states}")
    outputs = [list(seq.completion_token_ids) for seq in seqs]
    if any(len(out) != req.output_limit for out, req in zip(outputs, requests)):
        raise RuntimeError("Unexpected output length")
    return dict(mode=mode, workload=workload, concurrency=concurrency, samples=samples,
                inputs=[req.prompt for req in requests], output_limits=[req.output_limit for req in requests],
                outputs=outputs, cleanup=cleanup, rank_target_states=rank_states)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--system", choices=["ordinary", "speculative"], required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--workloads", nargs="+", default=["mixed-output", "long-short"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=["warmup", "control", "timing", "profile"],
                        choices=["warmup", "control", "timing", "profile", "minimal", "collective-events"])
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    out = args.raw / f"{args.system}-manifest.json"
    if out.exists():
        raise FileExistsError(out)
    root = Path.cwd()
    hashes = source_hashes(root)
    config = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=1024,
                  max_num_batched_tokens=2048, max_num_seqs=4,
                  gpu_memory_utilization=.70, scheduler_policy="original")
    result = dict(diagnostic_only=True, command=[sys.executable, *sys.argv], config=config,
                  system=args.system, target=TARGET, draft=DRAFT, reference=REFERENCE,
                  target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
                  draft_revision="3d13517724e81cb409ddf1d4650772ec52f1e18e",
                  versions={name: importlib.metadata.version(name)
                            for name in ("torch", "triton", "flash-attn", "transformers")},
                  cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
                  profiler_activities=str(torch.profiler.supported_activities()),
                  nsight={name: shutil.which(name) for name in ("nsys", "ncu")},
                  gpu=subprocess.check_output(["nvidia-smi"], text=True),
                  topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
                  git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  git_status=subprocess.check_output(["git", "status", "--short"], text=True),
                  source_hashes_before=hashes, trials=[], normal_exit=False)
    result["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    shutil.copyfile(__file__, args.raw / f"{args.system}-diagnostic-source.py")
    engine = None
    try:
        if args.system == "speculative":
            import nanovllm.speculative.concurrent_engine as mod
            mod.ConcurrentModelRunner = DiagnosticRunner
            engine = mod.ConcurrentLLMEngine(TARGET, draft_path=DRAFT, reference_path=REFERENCE,
                                             speculative_length=3, audit=False, **config)
        else:
            import nanovllm.engine.llm_engine as mod
            mod.ModelRunner = DiagnosticRunner
            engine = LLM(TARGET, **config)
        for concurrency in args.concurrency:
            for workload in args.workloads:
                runs = []
                for mode in args.modes:
                    run = trial(engine, args.system, workload, concurrency, mode, args.raw, args.steps)
                    runs.append(run)
                    result["trials"].append(run)
                    print(json.dumps(dict(system=args.system, concurrency=concurrency,
                                          workload=workload, mode=mode, cleanup=run["cleanup"])), flush=True)
                    out.write_text(json.dumps(result, indent=2))
                if any(run["outputs"] != runs[0]["outputs"] for run in runs[1:]):
                    raise RuntimeError("Profiling changed complete outputs")
                for step in range(args.steps):
                    if any(run["samples"][step]["outputs"] != runs[0]["samples"][step]["outputs"] for run in runs[1:]):
                        raise RuntimeError("Profiling changed captured step outputs")
        result["normal_exit"] = True
    except Exception:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()
        result["source_hashes_after"] = source_hashes(root)
        result["source_regression"] = hashes != result["source_hashes_after"]
        out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
