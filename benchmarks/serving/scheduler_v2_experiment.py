"""Tuning, frozen held-out load sweep, and scheduler mechanism ablations."""
import argparse
import atexit
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import traceback

from .adapter import run_workload
from .metrics import summarize
from .report import capture_manifest, command, write_artifacts, write_json
from .workload import IsolatedWorkloads, RequestSpec, workload_data


MAIN_POLICIES = ["original", "static", "slo-v1", "slo-v2"]
ABLATIONS = ["v2-no-overload", "v2-scalar", "static512", "static-head", "static-no-reserve"]


def candidate_gate(report):
    """Require candidate/static acceptance; keep narrowly scoped V1 warnings."""
    if not all(report.get(k) for k in ["finite", "coverage", "original_exact_parity", "high_precision_pass"]):
        return False
    for row in report["rows"]:
        if row["accepted"]:
            continue
        if row["policy"] != "slo-v1" or not row["control_valid"]:
            return False
        if row["top1"] != row["reference_top1"]:
            return False
        if any(row[m] > report["rules"]["hard_"+m] for m in row["limits"]):
            return False
    return True


def workload(factory, pattern, interval, seed, shuffle=False):
    pattern = list(pattern)
    if shuffle:
        random.Random(seed).shuffle(pattern)
    result = []
    for i, (prompt, output) in enumerate(pattern):
        item = factory.build(1, prompt, output, 0, seed+i)[0]
        result.append(RequestSpec(i, item.prompt_token_ids, output, i*interval*1_000_000))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acceptance-14b", required=True)
    parser.add_argument("--acceptance-small", required=True)
    args = parser.parse_args()
    # Do not turn numerical failure into an implicit exploratory bypass.
    acceptance = []
    for path in [args.acceptance_14b, args.acceptance_small]:
        report = json.loads(Path(path).read_text())
        if not candidate_gate(report):
            raise ValueError(f"Numerical acceptance not passed: {path}")
        acceptance.append(dict(path=path, all_policies_passed=report["passed"], candidate_gate_passed=True,
                               baseline_warnings=report["failures"]))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    engine, trials = None, []
    manifest = dict(status="initializing", arguments=vars(args))
    try:
        import torch
        from nanovllm import LLM, SamplingParams
        from nanovllm.engine.policy_scheduler import make_scheduler
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        factory = IsolatedWorkloads(sorted(set(tokenizer.get_vocab().values())-set(tokenizer.all_special_ids)))
        cfg = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=2048,
                   max_num_batched_tokens=2048, max_num_seqs=4, gpu_memory_utilization=.85,
                   scheduler_ttft_ms=2000., scheduler_tpot_ms=100.)
        plans = []
        tuning = [(128,64),(1536,16)]*6
        held = [(64,96),(384,32),(1024,48),(1792,8)]*3
        for repeat in range(2):
            for interval in [350,100]:
                w = workload(factory, tuning, interval, 7000+repeat*100+interval)
                for candidate in ([256,512] if repeat == 0 else [512,256]):
                    plans.append(dict(stage="tuning", repeat=repeat+1, interval=interval,
                                      policy=f"tune-{candidate}", requests=w))
        for repeat in range(3):
            for interval in [800,350,200,100,0]:
                w = workload(factory, held, interval, 23000+repeat*1000+interval, True)
                order = MAIN_POLICIES[repeat:]+MAIN_POLICIES[:repeat]
                for policy in order:
                    plans.append(dict(stage="held-out", repeat=repeat+1, interval=interval, policy=policy, requests=w))
        for repeat in range(2):
            for interval in [350,100]:
                w = workload(factory, held, interval, 44000+repeat*1000+interval, True)
                order = ABLATIONS[repeat:]+ABLATIONS[:repeat]
                # Paired references separate arbitration, granularity and shared
                # reservation/rotation, rather than using original as sole control.
                for policy in ["static", "slo-v2", *order]:
                    plans.append(dict(stage="ablation", repeat=repeat+1, interval=interval, policy=policy, requests=w))
        w = workload(factory, held*4, 150, 59000, True)
        for policy in MAIN_POLICIES:
            plans.append(dict(stage="sustained", repeat=1, interval=150, policy=policy, requests=w))
        serial = [dict(p, requests=workload_data(p["requests"])) for p in plans]
        write_json(output/"planned-workloads.json", serial)
        manifest = capture_manifest(args, cfg, serial, output, sys.argv[1:])
        manifest["command"] = manifest["command"].replace("-m benchmarks.serving ", "-m benchmarks.serving.scheduler_v2_experiment ", 1)
        manifest.update(status="running", prefix_cache="Fresh Scheduler/BlockManager metadata before every trial; warmup unique first tokens",
            numerical_acceptance=acceptance,
            tuning_rule="Choose highest pooled request goodput across two rates/two repeats; within 5% prefer smaller minimum chunk. Freeze before held-out.",
            held_out_digest=hashlib.sha256(json.dumps([x for x in serial if x["stage"]=="held-out"], sort_keys=True).encode()).hexdigest(),
            targets=dict(ttft_ms=2000, tpot_ms=100), observations="No sample filtering. Not an HTTP benchmark.")
        write_json(output/"manifest.json", manifest)
        engine = LLM(args.model, **cfg)
        base = engine.model_runner.config
        from .scheduler_v2_correctness import POLICIES, capture, compare_phase
        reference_dir = Path(args.acceptance_14b).parent
        runtime_reference = output/"runtime-correctness"
        runtime_reference.mkdir()
        for filename in ["inputs.json", "hf.pt", "probability-rules.json"]:
            shutil.copy2(reference_dir/filename, runtime_reference/filename)
        cases = json.loads((reference_dir/"inputs.json").read_text())
        numerical_policies = POLICIES[:4]+[f"original-batch{i}" for i in [1,2,3]]+POLICIES[4:]
        write_json(runtime_reference/"manifest.json", dict(captured_policies=numerical_policies,
                                                           independent_reference=args.acceptance_14b))
        for policy in numerical_policies:
            print("Runtime correctness", policy, flush=True)
            torch.save(capture(engine, cases, policy), runtime_reference/f"{policy}.pt")
        compare_phase(runtime_reference, probability=True)
        if not candidate_gate(json.loads((runtime_reference/"acceptance-probability.json").read_text())):
            raise RuntimeError("Runtime shape-control acceptance failed")

        def reset(name, minimum=256):
            assert engine.is_finished() and not engine.scheduler.block_manager.used_block_ids
            c = copy.copy(base)
            c.scheduler_policy = name if name in MAIN_POLICIES else "slo-v2"
            c.scheduler_v2_min_chunk = minimum
            if name.startswith("tune-"):
                c.scheduler_v2_min_chunk = int(name.split("-")[-1])
            if name == "v2-no-overload":
                c.scheduler_v2_overload = False
            if name == "v2-scalar":
                c.scheduler_v2_cost_model = "scalar"
            if name.startswith("static"):
                c.scheduler_policy = "static"
            if name == "static512":
                c.scheduler_prefill_chunk = 512
            scheduler = make_scheduler(c)
            if name == "static-head":
                decode = scheduler._decode
                def head_decode():
                    selected = decode()
                    for seq in selected:
                        scheduler.running.remove(seq)
                    scheduler.running.extendleft(reversed(selected))
                    return selected
                scheduler._decode = head_decode
            if name == "static-no-reserve":
                # Test-only ablation, allowed only when aggregate worst-case
                # capacity for the entire workload fits (checked before use).
                scheduler._completion_blocks = lambda seq: seq.num_blocks
                scheduler._reserve_needed = lambda: 0
            engine.scheduler = scheduler
            return {name: getattr(c, name) for name in c.__dataclass_fields__
                    if name.startswith("scheduler_") or name == "num_kvcache_blocks"}

        def sampling(s):
            return SamplingParams(temperature=.6, max_tokens=s.output_length, ignore_eos=True)

        def run(w):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            result = run_workload(engine, w, sampling, 4, 600_000_000_000,
                                  arrival_mode="open-loop", propagate_arrival=True)
            if not result.complete:
                raise RuntimeError(result.error)
            return result

        # Shared kernel warmup. No measured window includes these calls.
        for length in [128,1024,1792]:
            for batch in range(1,5):
                reset("original")
                run(factory.build(batch, length, 8, 0, 1000+length+batch))
        for budget in [256,512,1024]:
            reset("original")
            engine.scheduler.max_num_batched_tokens = budget
            run(factory.build(4, 1536, 8, 0, budget))
        selected_min = None
        for index, plan in enumerate(plans):
            if plan["stage"] != "tuning" and selected_min is None:
                scores = {}
                for candidate in [256,512]:
                    group = [t["summary"] for t in trials if t["policy"] == f"tune-{candidate}"]
                    scores[candidate] = sum(s["slo_good_requests"] for s in group)/sum(s["duration_s"] for s in group)
                best = max(scores.values())
                selected_min = min(c for c in scores if scores[c] >= best*.95)
                freeze = dict(selected_min_chunk=selected_min, scores=scores,
                              config=dict(scheduler_v2_min_chunk=selected_min, scheduler_v2_max_chunk=1024,
                                          scheduler_v2_overload_chunk=512),
                              rule=manifest["tuning_rule"], held_out_digest=manifest["held_out_digest"])
                write_json(output/"frozen-config.json", freeze)
                print("FROZEN", freeze, flush=True)
                from .scheduler_v2_correctness import capture, scores
                reference_dir = runtime_reference
                cases = json.loads((reference_dir/"inputs.json").read_text())
                captured = capture(engine, cases, "slo-v2", freeze["config"])
                torch.save(captured, output/"frozen-correctness.pt")
                reference = torch.load(reference_dir/"hf.pt", weights_only=True)
                calibration = json.loads((runtime_reference/"acceptance-probability.json").read_text())
                checks = []
                for rule in calibration["rows"]:
                    if rule["policy"] != "slo-v2":
                        continue
                    key = tuple(rule["position"])
                    ref = reference[key]
                    actual = scores(captured[key], ref)
                    passed = rule["control_valid"] and all(actual[m] <= min(limit, calibration["rules"]["hard_"+m])
                                                          for m,limit in rule["limits"].items())
                    passed &= actual["top1"] in ref.topk(10).indices.tolist()
                    passed &= (ref.max()-ref[actual["top1"]]).item() <= rule["tie_band"]
                    checks.append(dict(position=list(key), passed=bool(passed), **actual))
                accepted = len(checks) == len(captured) and all(c["passed"] for c in checks)
                write_json(output/"frozen-acceptance.json", dict(passed=accepted, checks=checks,
                    calibration=str(runtime_reference/"acceptance-probability.json"), config=freeze["config"]))
                if not accepted:
                    raise RuntimeError("Frozen candidate numerical acceptance failed; held-out timing not started")
            configured = reset(plan["policy"], selected_min or 256)
            w = plan["requests"]
            if plan["policy"] == "static-no-reserve":
                block_size = base.kvcache_block_size
                needed = sum((len(s.prompt_token_ids)+s.output_length+block_size-1)//block_size for s in w)
                assert needed <= base.num_kvcache_blocks, "Unsafe reservation ablation rejected"
            trial_name = f"{plan['stage']}/arrival-{plan['interval']}/repeat-{plan['repeat']}/{plan['policy']}"
            path = output/trial_name
            path.mkdir(parents=True)
            write_json(path/"scheduler-config.json", configured)
            warm = workload(factory, [(128,8),(512,8),(1024,8),(1536,8)], 0, 60000+index)
            warm_result = run(warm)
            write_json(path/"warmup.json", workload_data(warm))
            write_json(path/"warmup-steps.json", warm_result.steps)
            write_json(path/"workload.json", workload_data(w))
            print("MEASURE", index+1, "/", len(plans), trial_name, flush=True)
            result = run(w)
            summary, rows = summarize(result, 2000, 100)
            summary.update(kv_released=not engine.scheduler.block_manager.used_block_ids,
                policy=plan["policy"], stage=plan["stage"], interval_ms=plan["interval"], repeat=plan["repeat"])
            assert summary["kv_released"] and summary["initial_prefix_cache_blocks"] == 0
            write_artifacts(path, result, summary, rows, dict(manifest, policy=plan["policy"]))
            trials.append(dict(name=trial_name, policy=plan["policy"], summary=summary))
            write_json(output/"trials.json", trials)
        manifest["status"] = "completed"
        manifest["gpu_after"] = command(["nvidia-smi"])
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        write_json(output/"error.json", dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        if engine:
            atexit.unregister(engine.exit)
            engine.exit()
            manifest["normal_exit"] = all(p.exitcode == 0 for p in engine.ps)
        write_json(output/"manifest.json", manifest)
    return 0 if manifest["status"] == "completed" and manifest.get("normal_exit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
