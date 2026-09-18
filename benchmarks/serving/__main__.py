import argparse
import atexit
from dataclasses import fields
import math
from pathlib import Path
import re
import sys
import traceback

from .adapter import run_workload
from .metrics import summarize
from .report import capture_manifest, command, write_artifacts, write_json
from .verification import verify_observer
from .workload import IsolatedWorkloads, workload_data
from .warmup import warmup_plan, execute_warmup


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic in-process serving benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    for name, default in [("tensor-parallel-size", 2), ("max-model-len", 2048),
                          ("max-num-batched-tokens", 2048), ("max-num-seqs", 4),
                          ("num-requests", 12), ("prompt-length", 128),
                          ("output-length", 32), ("concurrency", 4)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85)
    parser.add_argument("--arrival-interval-ms", type=float, default=100)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--temperature", type=float, default=.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--length-mode", choices=["exact", "eos"], default="exact")
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--verify-observer", action="store_true")
    parser.add_argument("--arrival-mode", choices=["concurrency-gated", "open-loop"], default="concurrency-gated")
    parser.add_argument("--warmup-mode", choices=["legacy", "decode-shapes"], default="legacy")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--prefix-probe", action="store_true")
    parser.add_argument("--burst-probe", action="store_true")
    args = parser.parse_args(argv)
    for name in ["tensor_parallel_size", "max_model_len", "max_num_batched_tokens",
                 "max_num_seqs", "num_requests", "prompt_length", "output_length", "concurrency", "repeats"]:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    for name in ["timeout_s", "temperature", "gpu_memory_utilization", "ttft_slo_ms", "tpot_slo_ms"]:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.arrival_interval_ms) or args.arrival_interval_ms < 0:
        parser.error("arrival interval must be finite and nonnegative")
    if args.warmup_requests < 0 or args.gpu_memory_utilization > 1 or args.tensor_parallel_size > 8:
        parser.error("Invalid warmup, memory utilization or TP size")
    if args.temperature <= 1e-10:
        parser.error("Original sampler requires temperature > 1e-10")
    if args.prompt_length + args.output_length > args.max_model_len:
        parser.error("prompt + output exceeds max model length")
    if args.warmup_mode == "decode-shapes" and args.prompt_length + max(2, args.output_length) > args.max_model_len:
        parser.error("Decode shape warmup requires room for at least two output tokens")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.model_revision):
        parser.error("model revision must be an explicit 40-character commit SHA")
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = None
    result = None
    diagnostics = None
    trials = []
    manifest = dict(status="preparing", arguments=vars(args))
    exit_code = 1
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from nanovllm import LLM, SamplingParams

        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        dtype = str(getattr(config, "dtype", None) or config.torch_dtype).removeprefix("torch.")
        if dtype != args.dtype:
            raise ValueError(f"Production runtime uses model config dtype {dtype}, requested {args.dtype}")
        if args.prompt_length + args.output_length > min(args.max_model_len, config.max_position_embeddings):
            raise ValueError("Workload exceeds effective model context length")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        special = set(tokenizer.all_special_ids)
        vocab = sorted({i for i in tokenizer.get_vocab().values() if i < config.vocab_size and i not in special})
        factory = IsolatedWorkloads(vocab)
        workloads = [factory.build(args.num_requests, args.prompt_length, args.output_length,
                     round(args.arrival_interval_ms * 1e6), args.seed + repeat) for repeat in range(args.repeats)]
        specs = workloads[0]
        warmups = warmup_plan(factory, args)
        engine_config = {name: getattr(args, name) for name in [
            "tensor_parallel_size", "max_model_len", "max_num_batched_tokens", "max_num_seqs",
            "gpu_memory_utilization", "enforce_eager"]}
        write_json(output / "workload.json", [workload_data(w) for w in workloads])
        print("Verifying model checksums and capturing reproducibility metadata...", flush=True)
        manifest = capture_manifest(args, engine_config, [workload_data(w) for w in workloads], output, argv)
        manifest.update(arrival_mode=args.arrival_mode, concurrency_effective=(args.concurrency
                        if args.arrival_mode == "concurrency-gated" else None),
                        warmup_mode=args.warmup_mode, repeats=args.repeats, diagnostics=args.diagnostics,
                        prefix_isolation="Unique first token for every request across repeats and warmup; no cache clearing")
        write_json(output / "manifest.json", manifest)
        print("Loading original nano-vllm runtime...", flush=True)
        engine = LLM(args.model, **engine_config)
        effective = engine.model_runner.config
        manifest["effective_engine_config"] = {f.name: getattr(effective, f.name)
                                                for f in fields(effective) if f.name != "hf_config"}
        manifest["hf_config"] = config.to_dict()
        manifest["gpu_after_initialization"] = command(["nvidia-smi"])

        def reset_seed():
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        def sampling(spec, exact=None):
            return SamplingParams(temperature=args.temperature, max_tokens=spec.output_length,
                                  ignore_eos=args.length_mode == "exact" if exact is None else exact)

        if args.diagnostics:
            from .diagnostics import Diagnostics
            diagnostics = Diagnostics(engine, output).start()
        write_json(output / "warmup.json", [{"name": name, "requests": workload_data(w)} for name, w in warmups])
        reset_seed()
        manifest["warmup_coverage"] = []
        for name, warm in warmups:
            if diagnostics:
                diagnostics.mark(name)
            warm_result, coverage = execute_warmup(engine, warm, lambda s: sampling(s, True), round(args.timeout_s*1e9))
            coverage["name"] = name
            manifest["warmup_coverage"].append(coverage)
            warm_dir = output / "warmup" / name
            warm_dir.mkdir(parents=True)
            warm_summary, warm_rows = summarize(warm_result)
            write_artifacts(warm_dir, warm_result, warm_summary, warm_rows, manifest)
            if not coverage["complete"] or not coverage["kv_released"] or coverage["prefix_cache_blocks"]:
                raise RuntimeError(f"Warmup validation failed: {coverage}")
            if args.warmup_mode == "decode-shapes" and len(warm) not in coverage["decode_batches"]:
                raise RuntimeError(f"Warmup did not cover requested decode batch: {name}")
        if args.verify_observer:
            print("Checking original vs observed schedule and committed outputs...", flush=True)
            if diagnostics:
                diagnostics.mark("observer-verification")
            probes = factory.build(2, min(8, args.prompt_length), 4, 0, args.seed + 20000)
            manifest["observer_verification"] = verify_observer(
                engine, probes, lambda s: sampling(s, True), reset_seed)
        for repeat, workload in enumerate(workloads, 1):
            reset_seed()
            name = f"repeat-{repeat:02d}"
            trial_dir = output / name
            trial_dir.mkdir()
            write_json(trial_dir / "workload.json", workload_data(workload))
            if diagnostics:
                diagnostics.mark(name)
            print(f"Starting {name}: {args.arrival_mode}", flush=True)
            result = run_workload(engine, workload, sampling, args.concurrency, round(args.timeout_s * 1e9),
                                  arrival_mode=args.arrival_mode)
            released = not engine.scheduler.block_manager.used_block_ids
            summary, rows = summarize(result, args.ttft_slo_ms, args.tpot_slo_ms)
            trial_manifest = dict(manifest, repeat=repeat, kv_released_after_measurement=released)
            write_artifacts(trial_dir, result, summary, rows, trial_manifest)
            trials.append(dict(name=name, summary=summary, kv_released=released))
            write_json(output / "summary.json", trials)
            if not result.complete or not released or summary["initial_prefix_cache_blocks"]:
                raise RuntimeError(f"{name} failed correctness or prefix isolation")
        manifest["probes"] = []
        probe_plans = []
        if args.burst_probe:
            probe_plans.append(("burst-probe", factory.build(max(args.num_requests, args.max_num_seqs*3),
                                args.prompt_length, min(4, args.output_length), 0, args.seed + 30000)))
        if args.prefix_probe:
            length = effective.kvcache_block_size + 1
            if length + 4 > effective.max_model_len:
                raise ValueError("Context too short for non-vacuous prefix probe")
            for i in range(2):
                probe_plans.append((f"prefix-probe-{i+1}", factory.build(4, length, 4, 0, args.seed + 40000+i)))
        for name, probe in probe_plans:
            reset_seed()
            if diagnostics:
                diagnostics.mark(name)
            probe_dir = output / name
            probe_dir.mkdir()
            write_json(probe_dir / "workload.json", workload_data(probe))
            probe_result = run_workload(engine, probe, lambda s: sampling(s, True), args.concurrency,
                                        round(args.timeout_s*1e9), arrival_mode="open-loop")
            probe_summary, probe_rows = summarize(probe_result)
            write_artifacts(probe_dir, probe_result, probe_summary, probe_rows, manifest)
            manifest["probes"].append(dict(name=name, summary=probe_summary))
            if not probe_result.complete or probe_summary["prefix_cache_blocks"]:
                raise RuntimeError(f"Probe failed: {name}")
        manifest["kv_released_after_measurement"] = not engine.scheduler.block_manager.used_block_ids
        manifest["gpu_after_measurement"] = command(["nvidia-smi"])
        manifest["status"] = "measured" if result.complete else "failed"
        exit_code = 0 if result.complete and manifest["kv_released_after_measurement"] else 1
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "error.json", dict(error=manifest["error"], traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        if diagnostics is not None:
            diagnostics.close()
        if engine is not None:
            atexit.unregister(engine.exit)
            try:
                engine.exit()
                manifest["normal_exit"] = all(p.exitcode == 0 for p in engine.ps)
                if not manifest["normal_exit"]:
                    exit_code = 1
            except Exception as error:
                manifest["normal_exit"] = False
                manifest["cleanup_error"] = repr(error)
                exit_code = 1
        manifest["status"] = "completed" if exit_code == 0 else "failed"
        write_json(output / "manifest.json", manifest)
        if trials:
            from .report import write_repeat_report
            write_repeat_report(output, trials, manifest)
    print(f"Artifacts: {output}; exit status: {exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
