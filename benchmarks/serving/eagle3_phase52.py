"""Phase5.2 profiling only: frozen Phase5.1 runtime, no new inference feature."""
import argparse
import atexit
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
import traceback

import torch
import torch.distributed as dist

import nanovllm.speculative.concurrent_engine as engine_module
from nanovllm.speculative import batched_runtime as runtime
from benchmarks.serving.eagle3_phase51_runtime import Phase51Runner
from benchmarks.serving.eagle3_phase51 import clean, set_mode, heldout
from benchmarks.serving.eagle3_phase42 import (
    TARGET, DRAFT, REFERENCE, WORKLOADS, build_requests, run_closed_loop)
from benchmarks.serving.eagle3_phase43 import DiagnosticRunner, trial


def protected():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("nanovllm", "tests", "docs", "benchmarks")
            for p in sorted(Path(folder).rglob("*"))
            if p.is_file() and p.suffix in (".py", ".md", ".json", ".csv")
            and "__pycache__" not in str(p) and "phase52" not in str(p)
            and "phase5_2" not in str(p) and "PHASE5_2" not in str(p)}


class ReprofileRunner(Phase51Runner):
    def __init__(self, *args, **kwargs):
        self.diag_active = False
        self.in_target = False
        self.prof = None
        self.hooks, self.patches, self.module_stack = [], [], []
        super().__init__(*args, **kwargs)

    span = DiagnosticRunner.span
    _patch = DiagnosticRunner._patch

    def write_shm(self, method_name, *args):
        start = perf_counter_ns()
        result = super().write_shm(method_name, *args)
        if self.diag_active:
            self.rpc.append(dict(kind="pack_dispatch", method=method_name,
                operation=args[0] if method_name == "eagle3_batch" else None,
                start_ns=start, end_ns=perf_counter_ns()))
        return result

    def read_shm(self):
        start = perf_counter_ns()
        result = super().read_shm()
        if self.diag_active:
            method, args = result
            self.rpc.append(dict(kind="wait_read_unpack", method=method,
                operation=args[0] if method == "eagle3_batch" else None,
                start_ns=start, end_ns=perf_counter_ns()))
        return result

    def _metadata(self):
        for name in ("_descriptors", "verification_layout", "_status", "_zero"):
            original = getattr(runtime, name)
            def wrapped(*args, _fn=original, _name=name, **kwargs):
                start = perf_counter_ns()
                with self.span(f"diag.module::runtime_{_name}::{_name}"):
                    result = _fn(*args, **kwargs)
                self.metadata.append(dict(name=_name, target=self.target_index,
                    start_ns=start, end_ns=perf_counter_ns()))
                return result
            self._patch(runtime, name, wrapped)

    def prepare_prefill(self, seqs):
        if not self.diag_active:
            return super().prepare_prefill(seqs)
        start = perf_counter_ns()
        with self.span("diag.module::input_preparation::prepare_prefill"):
            result = super().prepare_prefill(seqs)
        self.metadata.append(dict(name="prepare_prefill", target=self.target_index,
                                  start_ns=start, end_ns=perf_counter_ns()))
        return result

    def eagle3_batch(self, operation, payload):
        if not self.diag_active or operation not in ("verify", "prefill"):
            return super().eagle3_batch(operation, payload)
        self.target_index += 1
        entries = payload["entries"]
        row = dict(index=self.target_index, operation=operation,
            rows=sum(1+len(e["proposals"]) if operation == "verify" else e["count"] for e in entries),
            batch=len(entries), start_ns=perf_counter_ns())
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.in_target = True
        try:
            with self.span(f"diag.target.{self.target_index}"):
                start.record()
                result = super().eagle3_batch(operation, payload)
                end.record()
        finally:
            self.in_target = False
        row["end_ns"] = perf_counter_ns()
        self.targets.append(row)
        self.events.append((start, end))
        return result

    def diagnostic(self, operation, config=None):
        if operation == "cleanup":
            return DiagnosticRunner.diagnostic(self, operation)
        if operation == "start":
            self.diag_config = config
            self.targets, self.events, self.rpc = [], [], []
            self.shapes, self.collectives, self.metadata = [], [], []
            self.target_index = -1
            self.diag_active = True
            if config["profile"]:
                self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
                self.prof.start()
                if config.get("rich"):
                    DiagnosticRunner._install_hooks(self)
                    # Captured MLP internals have no new Python module calls.
                    # Attribute their replay kernels to the outer graph-launch scope.
                    for index, layer in enumerate(self.model.model.layers):
                        def before(module, inputs, _index=index):
                            if self.in_target:
                                ctx = self.span(f"diag.module::mlp_region::layer{_index}")
                                ctx.__enter__()
                                self.module_stack.append((f"mlp{_index}", ctx))
                        def after(module, inputs, output):
                            if self.in_target:
                                self.module_stack.pop()[1].__exit__(None, None, None)
                        self.hooks.append(layer.mlp.register_forward_pre_hook(before))
                        self.hooks.append(layer.mlp.register_forward_hook(after, always_call=True))
            self._metadata()
            dist.barrier()
            torch.cuda.synchronize()
        elif operation == "stop":
            dist.barrier()
            torch.cuda.synchronize()
            self.diag_active = False
            for row, (start, end) in zip(self.targets, self.events):
                row["gpu_event_ms"] = start.elapsed_time(end)
            path = Path(self.diag_config["path"])
            if self.prof:
                self.prof.stop()
                self.prof.export_chrome_trace(f"{path}-rank{self.rank}.trace.json")
                self.prof = None
            for handle in self.hooks:
                handle.remove()
            self.hooks.clear()
            for obj, name, original in reversed(self.patches):
                setattr(obj, name, original)
            self.patches.clear()
            Path(f"{path}-rank{self.rank}.json").write_text(json.dumps(dict(
                rank=self.rank, targets=self.targets, rpc=self.rpc, shapes=self.shapes,
                collectives=self.collectives, metadata=self.metadata), indent=2))
            dist.barrier()
        else:
            raise ValueError(operation)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mode", choices=("eager", "mlp"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=2048,
        max_num_batched_tokens=2048, max_num_seqs=4, gpu_memory_utilization=.7,
        scheduler_policy="original")
    data = dict(phase="5.2", mode=args.mode, command=[sys.executable, *sys.argv],
        config=config, target=TARGET, draft=DRAFT, reference=REFERENCE,
        target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        draft_revision="3d13517724e81cb409ddf1d4650772ec52f1e18e",
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        gpu=subprocess.check_output(["nvidia-smi"], text=True),
        topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        versions={n: importlib.metadata.version(n) for n in ("torch", "flash-attn", "triton", "transformers")},
        environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "NCCL_DEBUG", "NCCL_GRAPH_MIXING_SUPPORT")},
        protected_before=protected(), source=Path(__file__).read_text(),
        warmups=[], trials=[], diagnostics=[], full_profiles=[], normal_exit=False)
    def save():
        (args.output/"manifest.json").write_text(json.dumps(data, indent=2))
    save()
    engine = None
    try:
        engine_module.ConcurrentModelRunner = ReprofileRunner
        engine = engine_module.ConcurrentLLMEngine(TARGET, draft_path=DRAFT, reference_path=REFERENCE,
            speculative_length=3, audit=False, **config)
        runner = engine.model_runner
        runner.call("phase51_control", "init", dict(raw=str(args.output)))
        if args.mode == "mlp":
            data["startup"] = runner.call("phase51_control", "prepare")
        set_mode(runner, args.mode)
        families = list(WORKLOADS)+["heldout-mixed"]
        cs = [1,2,4] if args.mode == "eager" else [4,2,1]
        def build(family, c, repeat):
            return heldout(engine.tokenizer,c,repeat) if family == "heldout-mixed" else build_requests(engine.tokenizer,family,c,repeat)
        # Finish all unprofiled serving before attaching profiler hooks.
        for c in cs:
            for family in families:
                rows = build(family,c,100)
                for row in rows:
                    row.output_limit = min(8,row.output_limit)
                run = run_closed_loop(engine,"speculative",rows,c)
                data["warmups"].append(dict(concurrency=c,workload=family,summary=run["summary"]))
            for repeat in range(args.repeats):
                order=families[repeat:]+families[:repeat]
                if repeat % 2:
                    order.reverse()
                for family in order:
                    torch.manual_seed(5100+repeat)
                    run=run_closed_loop(engine,"speculative",build(family,c,repeat),c)
                    run.update(concurrency=c,workload=family,repeat=repeat)
                    run["rank_cleanup"]=clean(engine)
                    data["trials"].append(run)
                    save()
                    print(json.dumps(dict(mode=args.mode,c=c,family=family,repeat=repeat,
                        tok_s=run["summary"]["output_tokens_per_s"])),flush=True)
        for c in cs:
            for family in ("mixed-output","long-short"):
                runs=[]
                for mode in ("timing","minimal","profile"):
                    run=trial(engine,"speculative",family,c,mode,args.output,2)
                    runs.append(run)
                    data["diagnostics"].append(run)
                    save()
                if any(r["outputs"]!=runs[0]["outputs"] for r in runs):
                    raise RuntimeError("Profiling changed outputs")
            # Include prefill, replacement waves, clipped q and cleanup in a trace.
            references=[]
            for profile in (False,True):
                rows=build("heldout-mixed",c,0)
                for row in rows:
                    row.output_limit=12
                if profile:
                    runner.call("diagnostic","start",dict(path=str(args.output/f"full-c{c}"),profile=True,rich=True))
                run=run_closed_loop(engine,"speculative",rows,c)
                if profile:
                    runner.call("diagnostic","stop")
                run.update(concurrency=c,profile=profile)
                references.append([r["output_token_ids"] for r in run["requests"]])
                data["full_profiles"].append(run)
                clean(engine)
                save()
            if references[0]!=references[1]:
                raise RuntimeError("Full serving profiler changed outputs")
        data["final_cleanup"]=clean(engine)
        runner.call("phase51_control","clear")
    except Exception:
        data["failure"]=traceback.format_exc()
        raise
    finally:
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()
        data["normal_exit"]="failure" not in data
        after=protected()
        data["protected_changed"]=[p for p,h in data["protected_before"].items() if after.get(p)!=h]
        save()


if __name__ == "__main__":
    main()
