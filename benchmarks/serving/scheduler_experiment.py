"""Counterbalanced original/static/SLO comparison on one unchanged TP runtime."""
import argparse
import atexit
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

from .adapter import run_workload
from .metrics import summarize
from .report import capture_manifest, command, write_artifacts, write_json
from .workload import IsolatedWorkloads, RequestSpec, workload_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "model-revision", "output-dir"]:
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--model-manifest")
    parser.add_argument("--tensor-parallel-size",type=int,default=2)
    parser.add_argument("--correctness-only",action="store_true")
    parser.add_argument("--correctness-corpus", choices=["repetitive", "varied"], default="repetitive")
    parser.add_argument("--exploratory-numerics", action="store_true",
        help="Retain failed raw-logit tolerance as a warning; still fail nonfinite/top-1/commit/KV checks")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = None
    trials = []
    manifest = dict(arguments=vars(args),status="initializing")
    exit_code = 1
    try:
        import torch
        from transformers import AutoTokenizer
        from nanovllm import LLM, SamplingParams
        from nanovllm.engine.policy_scheduler import make_scheduler
        from .scheduler_correctness import verify_policy_logits
        from .verification import verify_observer
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        vocab = sorted(set(tokenizer.get_vocab().values())-set(tokenizer.all_special_ids))
        factory = IsolatedWorkloads(vocab)
        engine_config = dict(tensor_parallel_size=args.tensor_parallel_size,enforce_eager=True,
            max_model_len=2048,max_num_batched_tokens=2048,max_num_seqs=4,gpu_memory_utilization=.85,
            scheduler_policy="original",scheduler_prefill_chunk=256,scheduler_min_prefill_chunk=128,
            scheduler_ttft_ms=2000.0,scheduler_tpot_ms=100.0)
        patterns = [(128,64),(1536,16)] * 6
        plans = []
        policies = ["original","static","slo-aware"]
        for repeat in range(3):
            for profile, interval in [("mixed-350ms",350),("mixed-100ms",100)]:
                # All policies receive identical measured input content for this
                # repeat. A fresh Scheduler explicitly resets CPU prefix metadata.
                workload=[]
                for i,(prompt_len,output_len) in enumerate(patterns):
                    request=factory.build(1,prompt_len,output_len,0,42+repeat*100+i)[0]
                    workload.append(RequestSpec(i,request.prompt_token_ids,output_len,i*interval*1_000_000))
                for policy in policies[repeat:]+policies[:repeat]:
                    plans.append((profile,repeat+1,policy,workload))
        write_json(output/"workloads.json",[dict(profile=p,repeat=r,policy=policy,requests=workload_data(w))
                                             for p,r,policy,w in plans])
        manifest=capture_manifest(args,engine_config,[workload_data(p[3]) for p in plans],output,sys.argv[1:])
        manifest["command"] = manifest["command"].replace("-m benchmarks.serving ","-m benchmarks.serving.scheduler_experiment ",1)
        manifest.update(slo=dict(ttft_ms=2000,tpot_ms=100),
            cache_reset="Fresh Scheduler/BlockManager metadata per trial; GPU KV storage reused, overwritten before read",
            design="One runtime, policy order rotated by repeat; batch 1..4 short/long decode warmup, not exhaustive kernel-shape coverage",
            warning="Small preliminary experiment; no statistical significance or broad performance claim")
        write_json(output/"manifest.json",manifest)
        engine=LLM(args.model,**engine_config)
        cfg=engine.model_runner.config
        def reset(policy):
            if not engine.is_finished() or engine.scheduler.block_manager.used_block_ids:
                raise RuntimeError("Cannot reset a nonempty runtime")
            config=copy.copy(cfg);config.scheduler_policy=policy
            engine.scheduler=make_scheduler(config)
        def sampling(s):
            return SamplingParams(temperature=.6,max_tokens=s.output_length,ignore_eos=True)
        def seed():
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        print("Correctness: committed teacher-prefix logits across three policies",flush=True)
        correctness=verify_policy_logits(engine,output,args.exploratory_numerics,args.correctness_corpus)
        write_json(output/"correctness.json",correctness)
        manifest["strict_logits_pass"]=correctness["strict_logits_pass"]
        if not correctness["strict_logits_pass"]:
            print("WARNING: strict raw-logit tolerance FAILED; exploratory timing only",flush=True)
        reset("original")
        probes=factory.build(2,8,4,0,9999)
        manifest["original_observer_parity"]=verify_observer(engine,probes,sampling,seed)
        if not args.correctness_only:
            print("Untimed kernel warmup for batch 1..4 and short/long contexts",flush=True)
            for length in [128,1536]:
                for batch in range(1,5):
                    warm=factory.build(batch,length,8,0,9000+length+batch)
                    seed()
                    result=run_workload(engine,warm,sampling,4,300_000_000_000,arrival_mode="open-loop",propagate_arrival=True)
                    if not result.complete:raise RuntimeError(result.error)
            for profile,repeat,policy,workload in plans:
                reset(policy)
                name=f"{profile}/repeat-{repeat:02d}/{policy}"
                trial_dir=output/name;trial_dir.mkdir(parents=True)
                # Calibrate policy EWMA with isolated, untimed native operations.
                warm=factory.build(4,128,8,0,8000+len(trials))
                seed()
                warm_result=run_workload(engine,warm,sampling,4,300_000_000_000,arrival_mode="open-loop",propagate_arrival=True)
                if not warm_result.complete:raise RuntimeError(warm_result.error)
                write_json(trial_dir/"warmup.json",workload_data(warm))
                write_json(trial_dir/"warmup-steps.json",warm_result.steps)
                write_json(trial_dir/"workload.json",workload_data(workload))
                seed()
                print(f"Measured {name}",flush=True)
                result=run_workload(engine,workload,sampling,4,300_000_000_000,
                                    arrival_mode="open-loop",propagate_arrival=True)
                summary,rows=summarize(result,2000,100)
                released=not engine.scheduler.block_manager.used_block_ids
                summary.update(policy=policy,profile=profile,repeat=repeat,kv_released=released,
                               prefix_metadata_entries=len(engine.scheduler.block_manager.hash_to_block_id))
                write_artifacts(trial_dir,result,summary,rows,dict(manifest,policy=policy,profile=profile,repeat=repeat))
                trials.append(dict(name=name,summary=summary))
                write_json(output/"summary.json",trials)
                if not result.complete or not released or summary["initial_prefix_cache_blocks"]:
                    raise RuntimeError(f"Trial failed or prefix cache contaminated: {name}")
        manifest["gpu_after"]=command(["nvidia-smi"])
        exit_code=0
    except Exception as error:
        manifest["error"]=repr(error)
        write_json(output/"error.json",dict(error=repr(error),traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()
            manifest["normal_exit"]=all(p.exitcode==0 for p in engine.ps)
            if not manifest["normal_exit"]:exit_code=1
        manifest["status"]="completed" if exit_code==0 else "failed"
        write_json(output/"manifest.json",manifest)
    return exit_code


if __name__=="__main__":
    raise SystemExit(main())
