"""Predeclared frozen Scheduler V2 finalization; no production modifications."""
import argparse
import atexit
import copy
import hashlib
import json
from math import ceil
from pathlib import Path
import shutil
import sys
import traceback

from .adapter import run_workload
from .metrics import summarize
from .report import capture_manifest, command, write_artifacts, write_json
from .scheduler_v2_experiment import workload
from .workload import IsolatedWorkloads, workload_data


POLICIES = ["original", "static-256", "static-512", "frozen-v2", "v2-no-overload"]
PROFILES = {
    "interactive": ([(96,24), (192,40), (320,72), (640,112)]*2, [240,650]),
    "prefill-heavy": ([(768,4), (1152,12), (1408,20), (1856,28)]*2, [180,550]),
    "decode-heavy": ([(48,80), (160,120), (448,144), (896,160)]*2, [300,900]),
}


def policy_config(base, name):
    if name not in POLICIES:
        raise ValueError(name)
    cfg = copy.copy(base)
    cfg.scheduler_policy = "original" if name == "original" else (
        "static" if name.startswith("static-") else "slo-v2")
    cfg.scheduler_prefill_chunk = 512 if name == "static-512" else 256
    cfg.scheduler_v2_min_chunk = 256
    cfg.scheduler_v2_max_chunk = 1024
    cfg.scheduler_v2_overload_chunk = 512
    cfg.scheduler_v2_cost_model = "bucketed"
    cfg.scheduler_v2_overload = name != "v2-no-overload"
    return cfg


def build_plans(factory, repeats=5):
    if repeats < 5:
        raise ValueError("Finalization requires at least five repeats")
    plans = []
    for repeat in range(repeats):
        for profile_index, (profile, (pattern, intervals)) in enumerate(PROFILES.items()):
            for interval in intervals:
                requests = workload(factory, pattern, interval,
                                    210000+repeat*10000+profile_index*1000+interval, True)
                # All policies see paired inputs; five rotations balance position.
                shift = repeat % len(POLICIES)
                for policy in POLICIES[shift:]+POLICIES[:shift]:
                    plans.append(dict(stage=profile, interval=interval, repeat=repeat+1,
                                      policy=policy, requests=requests))
    return plans


def attach_diagnostics(scheduler):
    """Observe existing decisions; never change the returned scheduling choice."""
    if not hasattr(scheduler, "progress_decision"):
        return
    choose = scheduler._choose

    def observed(now):
        previous = scheduler.overloaded
        evaluated = bool(scheduler.waiting)
        result = choose(now)
        d = scheduler.progress_decision
        d.update(overload_evaluated=evaluated, overload_transition=int(previous != scheduler.overloaded))
        if evaluated:
            lo = min(scheduler.config.scheduler_v2_min_chunk, scheduler.max_num_batched_tokens)
            decode = scheduler._predict_decode()
            forecasts = d["completion_forecasts_ns"]
            cost_min = forecasts[lo]/ceil(d["head_remaining_tokens"]/lo)
            if scheduler.running:
                cost_min -= decode
            slack = scheduler.ttft_ns-d["waiting_age_ns"]
            d.update(overload_decode_predicate=d["predicted_decode_round_ns"]+cost_min > scheduler.tpot_ns,
                     overload_backlog_predicate=ceil(d["waiting_tokens"]/lo)*(cost_min+decode) > max(slack,1),
                     predicted_min_prefill_ns=cost_min, predicted_decode_turn_ns=decode,
                     healthy_checks=scheduler.healthy_turns)
        return result

    scheduler._choose = observed


def assert_frozen(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(__file__).resolve().parents[2]
    hashes = {p: h for p,h in manifest["source_hashes"].items() if p.startswith("nanovllm/")}
    for name, digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen production source changed: {name}")
    return hashes


def numerical_check(engine, reference_dir, output):
    import torch
    from .scheduler_v2_correctness import capture, scores
    output.mkdir()
    reference_dir = Path(reference_dir)
    for name in ["inputs.json", "probability-rules.json", "acceptance-probability.json", "hf.pt"]:
        shutil.copy2(reference_dir/name, output/("calibration.json" if name.startswith("acceptance") else name))
    cases = json.loads((output/"inputs.json").read_text())
    reference = torch.load(output/"hf.pt", map_location="cpu", weights_only=True)
    calibration = json.loads((output/"calibration.json").read_text())
    rules = {tuple(r["position"]): r for r in calibration["rows"] if r["policy"] == "slo-v2"}
    captured, rows = {}, []
    for name in [*POLICIES, "original-repeat"]:
        cfg = policy_config(engine.model_runner.config, "original" if name == "original-repeat" else name)
        overrides = {key: getattr(cfg,key) for key in cfg.__dataclass_fields__ if key.startswith("scheduler_")}
        print("CORRECTNESS", name, flush=True)
        values = capture(engine, cases, cfg.scheduler_policy, overrides)
        torch.save(values, output/f"{name}.pt")
        captured[name] = values
        for key, value in values.items():
            rule, ref = rules[key], reference[key]
            actual = scores(value, ref)
            passed = torch.isfinite(value).all().item() and rule["control_valid"]
            passed &= all(actual[m] <= min(limit,calibration["rules"]["hard_"+m]) for m,limit in rule["limits"].items())
            passed &= actual["top1"] in ref.topk(10).indices.tolist()
            passed &= (ref.max()-ref[actual["top1"]]).item() <= rule["tie_band"]
            rows.append(dict(policy=name, position=list(key), accepted=bool(passed), **actual))
    parity = all(torch.equal(v,captured["original-repeat"][key]) for key,v in captured["original"].items())
    coverage = all(set(v) == set(reference) for v in captured.values())
    passed = parity and coverage and all(r["accepted"] for r in rows)
    write_json(output/"acceptance.json", dict(passed=passed, original_exact_parity=parity,
        coverage=coverage, rows=rows, calibration=str(reference_dir),
        note="Unchanged V2 rules and control limits; no recalibration on V2.1 results."))
    if not passed:
        raise RuntimeError("Frozen-rule numerical acceptance failed; timing not started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--old-workloads", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    frozen = assert_frozen(args.frozen_manifest)
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
        factory = IsolatedWorkloads(set(tokenizer.get_vocab().values())-set(tokenizer.all_special_ids))
        old = json.loads(Path(args.old_workloads).read_text())
        factory.used_first_tokens.update(r["prompt_token_ids"][0] for p in old for r in p["requests"])
        plans = build_plans(factory, args.repeats)
        serial = [dict(p, requests=workload_data(p["requests"])) for p in plans]
        write_json(output/"planned-workloads.json", serial)
        write_json(output/"freeze.json", dict(production_hashes=frozen, policies=POLICIES,
            profiles=PROFILES, tuning=False, variant="none", repeats=args.repeats,
            targets=dict(ttft_ms=2000,tpot_ms=100), prior_workloads=args.old_workloads))
        cfg = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=2048,
                   max_num_batched_tokens=2048, max_num_seqs=4, gpu_memory_utilization=.85,
                   scheduler_ttft_ms=2000., scheduler_tpot_ms=100.)
        manifest = capture_manifest(args, cfg, serial, output, sys.argv[1:])
        manifest["command"] = manifest["command"].replace("-m benchmarks.serving ", "-m benchmarks.serving.scheduler_finalization ",1)
        manifest.update(status="running", prefix_cache="Fresh metadata every trial; isolated warmup tokens",
                        frozen_production_hashes=frozen, tuning=False, variant="none")
        write_json(output/"manifest.json", manifest)
        engine = LLM(args.model, **cfg)
        numerical_check(engine, args.reference_dir, output/"correctness")

        def reset(name):
            assert engine.is_finished() and not engine.scheduler.block_manager.used_block_ids
            config = policy_config(engine.model_runner.config, name)
            engine.scheduler = make_scheduler(config)
            attach_diagnostics(engine.scheduler)
            return {k: getattr(config,k) for k in config.__dataclass_fields__
                    if k.startswith("scheduler_") or k == "num_kvcache_blocks"}

        def run(requests):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            return run_workload(engine, requests, lambda s: SamplingParams(
                temperature=.6,max_tokens=s.output_length,ignore_eos=True),
                4,600_000_000_000,arrival_mode="open-loop",propagate_arrival=True)

        warmups = []
        for length in [192,896,1856]:
            for batch in range(1,5):
                reset("original")
                w = factory.build(batch,length,8,0,310000+length+batch)
                result = run(w)
                warmups.append(dict(requests=workload_data(w),steps=result.steps,complete=result.complete,error=result.error))
                write_json(output/"shared-warmup.json",warmups)
                if not result.complete:
                    raise RuntimeError(result.error)
        for index, plan in enumerate(plans):
            configured = reset(plan["policy"])
            name = f"{plan['stage']}/arrival-{plan['interval']}/repeat-{plan['repeat']}/{plan['policy']}"
            path = output/name
            path.mkdir(parents=True)
            write_json(path/"scheduler-config.json",configured)
            warm = workload(factory,[(128,8),(512,8),(1024,8),(1536,8)],0,410000+index)
            warm_result = run(warm)
            write_json(path/"warmup.json", workload_data(warm))
            write_json(path/"warmup-steps.json", warm_result.steps)
            if not warm_result.complete:
                raise RuntimeError(warm_result.error)
            write_json(path/"workload.json",workload_data(plan["requests"]))
            print("MEASURE",index+1,"/",len(plans),name,flush=True)
            result = run(plan["requests"])
            summary, rows = summarize(result,2000,100)
            summary.update(kv_released=not engine.scheduler.block_manager.used_block_ids,
                policy=plan["policy"],stage=plan["stage"],interval_ms=plan["interval"],repeat=plan["repeat"])
            # Preserve partial/error records before stopping, including slow samples.
            write_artifacts(path,result,summary,rows,dict(manifest,policy=plan["policy"]))
            trials.append(dict(name=name,policy=plan["policy"],summary=summary))
            write_json(output/"trials.json",trials)
            if not result.complete or not summary["kv_released"] or summary["initial_prefix_cache_blocks"]:
                raise RuntimeError(f"Trial safety check failed: {name}: {result.error}")
        assert_frozen(args.frozen_manifest)
        manifest.update(status="completed",gpu_after=command(["nvidia-smi"]),trial_count=len(trials))
    except Exception as error:
        manifest.update(status="failed",error=repr(error))
        write_json(output/"error.json",dict(error=repr(error),traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        if engine:
            atexit.unregister(engine.exit)
            engine.exit()
            manifest["normal_exit"] = all(p.exitcode == 0 for p in engine.ps)
        write_json(output/"manifest.json",manifest)
    return 0 if manifest["status"] == "completed" and manifest.get("normal_exit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
