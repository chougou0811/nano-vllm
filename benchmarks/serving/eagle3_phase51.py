"""Bounded Phase5.1 diagnostics, opt-in prototype and serving validation."""
import argparse
import atexit
from dataclasses import asdict
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

from nanovllm import SamplingParams
import nanovllm.speculative.concurrent_engine as engine_module
import nanovllm.speculative.coordinator as coordinator_module
from benchmarks.serving.eagle3_phase51_runtime import Phase51Runner
from benchmarks.serving.eagle3_phase44a import live_verification, verify_rollback, admit
from benchmarks.serving.eagle3_phase42 import (
    TARGET, DRAFT, REFERENCE, WORKLOADS, Request, build_requests,
    run_closed_loop, cleanup_state,
)


def protected_hashes():
    paths = []
    for directory in ("nanovllm", "tests", "docs", "benchmarks"):
        paths.extend(p for p in Path(directory).rglob("*") if p.is_file()
                     and p.suffix in (".py", ".md", ".json", ".csv")
                     and "__pycache__" not in str(p) and "phase51" not in str(p)
                     and "phase5_1" not in str(p) and "PHASE5_1" not in str(p))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def clean(engine):
    host = cleanup_state(engine, "speculative")
    ranks = engine.model_runner.call("phase51_control", "snapshot")
    if any(host.values()) or any(row["targets"] for row in ranks):
        raise RuntimeError("Phase51 state leak")
    return dict(host=host, ranks=[dict(rank=r["rank"], targets=r["targets"],
                                     entries=r["entries"], memory=r["memory"]) for r in ranks])


def set_mode(runner, mode, **kwargs):
    runner.call("phase51_control", "mode", dict(mode=mode, **kwargs))


def fixed(engine, args, result, save, modes, layers=None):
    runner = engine.model_runner
    for batch in args.batches:
        for context in (256, 768):
            set_mode(runner, "eager")
            entries = live_verification(engine, [context] * batch)
            label = getattr(args, "label", "default")
            cell = dict(batch=batch, context=context, layers=layers, label=label, samples=[])
            reference = None
            for mode in modes:
                set_mode(runner, mode, **({"layers": layers} if layers else {}))
                for _ in range(3):
                    value = verify_rollback(runner, entries)
                    if reference is None:
                        reference = value
                    if value != reference:
                        raise RuntimeError("Fixed-prefix output/rank disagreement")
            for repeat in range(5):
                order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
                if repeat % 2:
                    order = list(reversed(order))
                for mode in order:
                    set_mode(runner, mode, measure=True, **({"layers": layers} if layers else {}))
                    for sample in range(args.samples):
                        index = len(runner.records)
                        start = perf_counter_ns()
                        value = runner.call("eagle3_batch", "verify", dict(entries=entries))
                        elapsed = perf_counter_ns() - start
                        runner.call("eagle3_batch", "rollback", dict(entries=[
                            dict(seq_id=e["seq_id"], generation=e["generation"]) for e in entries]))
                        if value != reference:
                            raise RuntimeError("Timed fixed-prefix disagreement")
                        cell["samples"].append(dict(mode=mode, repeat=repeat, sample=sample,
                                                    timing_index=index, host_ns=elapsed))
            for mode in modes:
                set_mode(runner, mode, measure=True, **({"layers": layers} if layers else {}))
                prefix = args.output/f"{label}-b{batch}-ctx{context}-layers{layers}-{mode}"
                runner.call("phase51_control", "trace_start", dict(path=str(prefix)))
                for _ in range(2):
                    verify_rollback(runner, entries)
                runner.call("phase51_control", "trace_stop")
            engine.scheduler.close_all()
            cell["cleanup"] = clean(engine)
            result["fixed"].append(cell)
            save()
            print(json.dumps(dict(stage=args.stage, batch=batch, context=context, layers=layers,
                                  completed=True)), flush=True)


def generation(engine, contexts, mode, output=16):
    runner = engine.model_runner
    set_mode(runner, mode)
    seqs = admit(engine, contexts, output)
    ids = {seq.seq_id: i for i, seq in enumerate(seqs)}
    commits = []
    while not engine.is_finished():
        engine.step()
        last = engine.speculative_coordinator.last_step
        if last.get("kind") == "decode":
            rows = []
            for entry in last["entries"]:
                row = asdict(entry)
                row["seq_id"] = ids[row["seq_id"]]
                rows.append(row)
            commits.append(rows)
    return dict(outputs=[list(s.completion_token_ids) for s in seqs],
                commits=commits, cleanup=clean(engine))


def correctness(engine, args, result, save):
    runner = engine.model_runner
    result["startup"] = runner.call("phase51_control", "prepare")
    for batch in args.batches:
        for context in (255, 256, 257, 1024):
            contexts = [context] * batch
            eager = generation(engine, contexts, "eager")
            candidate = generation(engine, contexts, "audit")
            passed = all(eager[k] == candidate[k] for k in ("outputs", "commits"))
            result["correctness"].append(dict(contexts=contexts, passed=passed,
                                              eager=eager, candidate=candidate))
            save()
            if not passed:
                raise RuntimeError("Generation proposal/acceptance disagreement")
    # Heterogeneous q lengths and replacement waves are covered by serving;
    # explicit short limits exercise eager fallback and termination here.
    for limit in (1, 2, 4, 5):
        a = generation(engine, [255, 256, 257, 1024], "eager", limit)
        b = generation(engine, [255, 256, 257, 1024], "audit", limit)
        passed = all(a[k] == b[k] for k in ("outputs", "commits"))
        result["correctness"].append(dict(output_limit=limit, passed=passed, eager=a, candidate=b))
        if not passed:
            raise RuntimeError("Output limit mismatch")
    probe = generation(engine, [255], "eager", 8)
    original_eos = runner.config.eos
    runner.config.eos = probe["outputs"][0][2]
    try:
        a = generation(engine, [255], "eager", 16)
        b = generation(engine, [255], "audit", 16)
        passed = all(a[k] == b[k] for k in ("outputs", "commits")) and len(a["outputs"][0]) < 16
        result["eos_test"] = dict(passed=passed, injected_eos=runner.config.eos, eager=a, candidate=b)
        if not passed:
            raise RuntimeError("EOS mismatch")
    finally:
        runner.config.eos = original_eos
    set_mode(runner, "mlp")
    admit(engine, [255])
    while engine.scheduler.waiting:
        engine.step()
    old = coordinator_module.accept_greedy
    def fail(*args, **kwargs):
        raise RuntimeError("Phase51 intentional acceptance failure")
    coordinator_module.accept_greedy = fail
    try:
        engine.step()
        raise AssertionError("Expected intentional failure")
    except RuntimeError as exc:
        if "Phase51 intentional acceptance failure" not in str(exc):
            raise
        result["exception_cleanup"] = clean(engine)
    finally:
        coordinator_module.accept_greedy = old
    result["correctness_passed"] = True
    save()


def heldout(tokenizer, concurrency, repeat):
    text = "An archive stores checksums and versioned records. Recovery validates each restored record before publishing a consistent snapshot. "
    stream = tokenizer.encode(text * 160, add_special_tokens=False)
    rows = []
    for i in range(max(4, 2 * concurrency)):
        length = (127, 383, 895, 511)[i % 4]
        offset = (repeat * 43 + i * 29) % len(stream)
        tokens = ((stream[offset:] + stream[:offset]) * 4)[:length]
        tokens[0] = 7000 + repeat * 41 + i * 13
        rows.append(Request(i, tokens, (48, 96, 40, 80)[i % 4]))
    return rows


def serving(engine, args, result, save):
    runner = engine.model_runner
    if args.serving_mode != "literal":
        result["startup"] = runner.call("phase51_control", "prepare")
    else:
        result["startup"] = []
    selected_modes = {"paired": ("eager", "mlp"), "literal": ("eager",),
                      "optimized": ("mlp",)}[args.serving_mode]
    result["serving_mode"] = args.serving_mode
    families = list(WORKLOADS) + ["heldout-mixed"]
    def build(family, repeat):
        return (heldout(engine.tokenizer, args.concurrency, repeat) if family == "heldout-mixed"
                else build_requests(engine.tokenizer, family, args.concurrency, repeat))
    for mode in selected_modes:
        set_mode(runner, mode)
        for family in families:
            rows = build(family, 100)
            for row in rows:
                row.output_limit = min(8, row.output_limit)
            warm = run_closed_loop(engine, "speculative", rows, args.concurrency)
            result["warmups"].append(dict(mode=mode, family=family, summary=warm["summary"]))
    for repeat in range(5):
        order = families[repeat:] + families[:repeat]
        if repeat % 2:
            order.reverse()
        for family in order:
            pair = []
            modes = selected_modes if repeat % 2 == 0 else tuple(reversed(selected_modes))
            for mode in modes:
                set_mode(runner, mode)
                torch.manual_seed(5100 + repeat)
                run = run_closed_loop(engine, "speculative", build(family, repeat), args.concurrency)
                run.update(mode=mode, workload=family, repeat=repeat, concurrency=args.concurrency)
                run["rank_memory"] = clean(engine)
                result["trials"].append(run)
                pair.append(run)
                save()
                print(json.dumps(dict(mode=mode, workload=family, repeat=repeat,
                    concurrency=args.concurrency, tok_s=run["summary"]["output_tokens_per_s"])), flush=True)
            if len(pair) == 2 and [r["output_token_ids"] for r in pair[0]["requests"]] != [r["output_token_ids"] for r in pair[1]["requests"]]:
                raise RuntimeError("Serving output parity failure")
            if len(pair) == 2 and pair[0]["summary"]["speculation"] != pair[1]["summary"]["speculation"]:
                raise RuntimeError("Serving acceptance/progress parity failure")
            if any(any(run["summary"]["cleanup"].values()) for run in pair):
                raise RuntimeError("Serving cleanup failure")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", choices=("diagnostic", "lifecycle", "prototype", "correctness", "serving"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--serving-mode", choices=("paired", "literal", "optimized"), default="paired")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=2048,
                  max_num_batched_tokens=2048, max_num_seqs=4, gpu_memory_utilization=.70,
                  scheduler_policy="original")
    result = dict(command=[sys.executable, *sys.argv], config=config, target=TARGET, draft=DRAFT,
        reference=REFERENCE, target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        draft_revision="3d13517724e81cb409ddf1d4650772ec52f1e18e", stage=args.stage,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions={n: importlib.metadata.version(n) for n in ("torch", "flash-attn", "triton", "transformers")},
        gpu=subprocess.check_output(["nvidia-smi"], text=True),
        topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        environment={k: os.environ.get(k) for k in ("NCCL_DEBUG", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "NCCL_GRAPH_MIXING_SUPPORT")},
        protected_before=protected_hashes(), fixed=[], correctness=[], trials=[], warmups=[], normal_exit=False)
    sources = {str(p): p.read_text() for p in Path("benchmarks/serving").glob("eagle3_phase51*.py")}
    result["sources"] = sources
    result["source_hashes"] = {p: hashlib.sha256(t.encode()).hexdigest() for p, t in sources.items()}
    def save():
        (args.output/"manifest.json").write_text(json.dumps(result, indent=2, default=str))
    save()
    engine = None
    try:
        engine_module.ConcurrentModelRunner = Phase51Runner
        engine = engine_module.ConcurrentLLMEngine(TARGET, draft_path=DRAFT, reference_path=REFERENCE,
                                                   speculative_length=3, audit=False, **config)
        engine.model_runner.call("phase51_control", "init", dict(raw=str(args.output)))
        if args.stage == "diagnostic":
            fixed(engine, args, result, save, ["eager", "wrapper", "shadow", "agreement", "miss"])
        elif args.stage == "lifecycle":
            args.label = "before-capture"
            fixed(engine, args, result, save, ["eager", "miss"])
            set_mode(engine.model_runner, "eager")
            entries = live_verification(engine, [127])
            engine.model_runner.call("phase51_control", "old_capture", dict(entries=entries))
            engine.scheduler.close_all()
            args.label = "live-graph-no-replay"
            fixed(engine, args, result, save, ["eager", "miss"])
            engine.model_runner.call("phase51_control", "old_release")
            args.label = "after-release"
            fixed(engine, args, result, save, ["eager", "miss"])
        elif args.stage == "prototype":
            result["one_layer_startup"] = engine.model_runner.call("phase51_control", "prepare", dict(layers=1))
            fixed(engine, args, result, save, ["eager", "mlp"], layers=1)
            result["all_layer_startup"] = engine.model_runner.call("phase51_control", "prepare")
            fixed(engine, args, result, save, ["eager", "mlp"])
        elif args.stage == "correctness":
            correctness(engine, args, result, save)
        else:
            serving(engine, args, result, save)
        result["final_cleanup"] = clean(engine)
        result["ranks"] = engine.model_runner.call("phase51_control", "snapshot")
        engine.model_runner.call("phase51_control", "clear")
        result["after_release"] = clean(engine)
    except Exception:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        try:
            if engine is not None:
                atexit.unregister(engine.exit)
                engine.exit()
            result["normal_exit"] = True
        finally:
            result["protected_after"] = protected_hashes()
            result["protected_changed"] = [p for p,h in result["protected_before"].items()
                                            if result["protected_after"].get(p) != h]
            save()


if __name__ == "__main__":
    main()
