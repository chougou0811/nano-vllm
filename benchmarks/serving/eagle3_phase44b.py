"""Phase 4.4B correctness gate and frozen-contract closed-loop trials."""
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

from nanovllm import LLM
from nanovllm.speculative.concurrent_engine import ConcurrentLLMEngine
from nanovllm.speculative.graph_policy import GraphCacheConfig
from benchmarks.serving import eagle3_phase42 as frozen


ROOT = Path("/root/autodl-tmp/eagle3-phase4.4b-20260924")
CONFIG = dict(tensor_parallel_size=2,enforce_eager=True,max_model_len=1024,
              max_num_batched_tokens=2048,max_num_seqs=4,gpu_memory_utilization=.70,
              scheduler_policy="original")


def hashes():
    paths = list(Path("nanovllm").rglob("*.py"))+list(Path("tests").glob("test_*.py"))
    paths += [p for p in Path("benchmarks/serving").glob("*.py") if "44b_summary" not in p.name]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def policy(system):
    return asdict(GraphCacheConfig(max_graph_entries=int(system[-1]) if system.startswith("second") else 4,
                                   capture_policy="second" if system.startswith("second") else "never"))


def reset(engine, system):
    engine.model_runner.call("graph_control","reset",dict(config=policy(system),enabled=system!="eager"))


def clean(engine):
    state = frozen.cleanup_state(engine,"speculative")
    ranks = engine.model_runner.call("graph_control","snapshot")
    if any(state.values()) or any(r["target_states"] for r in ranks):
        raise RuntimeError("Request/transaction state leak")
    return state,ranks


def audit_run(engine,requests,concurrency,system):
    reset(engine,system)
    runner = engine.model_runner
    runner.call("graph_control","audit",dict(enabled=True))
    trace,drafts = [],[]
    call,step = runner.call,engine.step

    def observed_call(method,*args):
        value = call(method,*args)
        if method=="eagle3_batch" and args[0] in ("verify","commit"):
            trace.append(dict(operation=args[0],payload=args[1],result=value))
        return value

    def observed_step():
        value = step()
        last = engine.speculative_coordinator.last_step
        if last.get("kind")=="decode":
            trace.append(dict(commits=[asdict(x) for x in last["entries"]]))
        drafts.append({sid:dict(C=s.target_cursor,D=s.draft_cursor,
                               actual_D=engine.speculative_coordinator.drafts[sid].cursor)
                       for sid,s in engine.speculative_coordinator.requests.items()})
        return value

    runner.call,engine.step = observed_call,observed_step
    try:
        run = frozen.run_closed_loop(engine,"speculative",requests,concurrency)
    finally:
        runner.call,engine.step = call,step
    audits = runner.call("graph_control","audit_snapshot")
    _,cache = clean(engine)
    mapping = {r["sequence_id"]:r["request_id"] for r in run["requests"]}
    physical_blocks = {}
    # Normalize only ownership identifiers, not token IDs/cursors or outputs.
    for item in trace:
        for entry in item.get("payload",{}).get("entries",[]):
            entry["seq_id"] = mapping[entry["seq_id"]]
            if "blocks" in entry:
                for block in entry["blocks"]:
                    physical_blocks.setdefault(block,len(physical_blocks))
                entry["blocks"] = [physical_blocks[b] for b in entry["blocks"]]
        for entry in item.get("commits",[]):
            entry["seq_id"] = mapping[entry["seq_id"]]
        if "result" in item:
            result = item["result"]
            ranks = result["ranks"] if isinstance(result,dict) else result
            for rank in ranks:
                for row in rank:
                    row[0] = mapping[row[0]]
    for audit in audits:
        for item in audit["states"]:
            for row in item["rows"]:
                row["seq_id"] = mapping[row["seq_id"]]
    drafts = [{mapping[sid]:value for sid,value in row.items()} for row in drafts]
    runner.call("graph_control","clear")
    released = clean(engine)[1]
    return dict(outputs=[r["output_token_ids"] for r in run["requests"]],trace=trace,drafts=drafts,
                audits=audits,cache=cache,released=released,physical_block_map=physical_blocks,
                cleanup=run["summary"]["cleanup"])


def correctness(engine,args,result,save):
    runner = engine.model_runner
    cases = [(1,[255]*4,[12]*4),(1,[256]*4,[12]*4),(1,[257]*4,[12]*4),
             (1,[768]*4,[12]*4),(2,[255]*8,[12]*8),
             (2,[255,256,257,255],[12,20,8,16]),(4,[255]*16,[12]*16),
             (4,[255,256,257,768]*2,[8,16,12,20]*2)]
    if args.pilot:
        cases = cases[:1]
    for concurrency,lengths,limits in cases:
        def requests():
            stream = engine.tokenizer.encode("A careful experiment preserves every committed state transition. "*200,
                                              add_special_tokens=False)
            return [frozen.Request(i,stream[:length],limit) for i,(length,limit) in enumerate(zip(lengths,limits))]
        reference = audit_run(engine,requests(),concurrency,"eager")
        actual = audit_run(engine,requests(),concurrency,"second8")
        flags = {name:reference[name]==actual[name] for name in ("outputs","trace","drafts","audits")}
        result["correctness"].append(dict(concurrency=concurrency,lengths=lengths,limits=limits,
                                          checks=flags,reference=reference,actual=actual))
        save()
        print(json.dumps(dict(concurrency=concurrency,lengths=lengths,checks=flags)),flush=True)
        if not all(flags.values()):
            raise RuntimeError("Concurrent graph versus frozen path disagreement")
    # Explicit same-state transition, asymmetric availability/key veto and eviction.
    from benchmarks.serving.eagle3_phase44a import live_verification,verify_rollback
    reset(engine,"second2")
    runner.call("graph_control","audit",dict(enabled=True))
    transitions = []
    for context in (255,256,257):
        entries = live_verification(engine,[context])
        for _ in range(3):
            transitions.append(verify_rollback(runner,entries))
        engine.scheduler.close_all()
    entries = live_verification(engine,[257])
    same = []
    for veto in (None,"missing",None,"key",None):
        runner.call("graph_control","audit",dict(enabled=True,veto=veto))
        same.append(verify_rollback(runner,entries))
    if any(x!=same[0] for x in same):
        raise RuntimeError("Synchronized fallback changed target verification")
    engine.scheduler.close_all()
    result["transition_cache"] = clean(engine)[1]
    result["fallback_passed"] = True
    # Reuse the live key, then throw in host acceptance after verification.
    runner.call("graph_control","audit",dict(enabled=True))
    from benchmarks.serving.eagle3_phase44a import admit
    admit(engine,[257])
    while engine.scheduler.waiting:
        engine.step()
    old = frozen.coordinator_module.accept_greedy

    def intentional(*args,**kwargs):
        raise RuntimeError("phase44b intentional acceptance exception")

    frozen.coordinator_module.accept_greedy = intentional
    try:
        engine.step()
        raise AssertionError("Expected injected acceptance failure")
    except RuntimeError as exc:
        if "phase44b intentional" not in str(exc):
            raise
        result["exception_cleanup"] = clean(engine)[0]
    finally:
        frozen.coordinator_module.accept_greedy = old
    runner.call("graph_control","clear")
    result["final_release"] = clean(engine)
    result["correctness_passed"] = True


def serving(engine,args,result,save):
    ordinary = args.systems==["ordinary"]
    for workload in frozen.WORKLOADS:
        if not ordinary:
            reset(engine,"never")
        warm = frozen.build_requests(engine.tokenizer,workload,args.concurrency,0,warmup=True)
        run = frozen.run_closed_loop(engine,"ordinary" if ordinary else "speculative",warm,args.concurrency)
        result["warmups"].append(dict(workload=workload,summary=run["summary"]))
    for repeat in range(args.repeats):
        order = list(frozen.WORKLOADS[repeat:]+frozen.WORKLOADS[:repeat])
        if repeat%2:
            order.reverse()
        for workload in order:
            systems = list(args.systems)
            shift = (repeat+frozen.WORKLOADS.index(workload))%len(systems)
            systems = systems[shift:]+systems[:shift]
            if repeat%2:
                systems.reverse()
            for system in systems:
                if system in ("second2","second8") and workload not in ("long-long","mixed-output"):
                    continue
                if not ordinary:
                    reset(engine,system)
                torch.manual_seed(4200+repeat)
                requests = frozen.build_requests(engine.tokenizer,workload,args.concurrency,repeat)
                inputs = [dict(request_id=r.request_id,prompt=r.prompt,output_limit=r.output_limit) for r in requests]
                run = frozen.run_closed_loop(engine,"ordinary" if ordinary else "speculative",requests,args.concurrency)
                run.update(system=system,workload=workload,repeat=repeat,concurrency=args.concurrency,inputs=inputs)
                if ordinary:
                    if any(run["summary"]["cleanup"].values()):
                        raise RuntimeError("Ordinary request cleanup failed")
                else:
                    _,ranks = clean(engine)
                    run["graphs"] = ranks
                    if any(r["dropped_events"] for r in ranks):
                        raise RuntimeError("Graph event log overflow")
                    for key in ("allocated","reserved"):
                        name = f"rank0_peak_{key}"
                        run["summary"]["memory"][name] = max(run["summary"]["memory"][name],ranks[0]["memory"][f"peak_{key}"])
                    engine.model_runner.call("graph_control","clear")
                    run["released"] = clean(engine)[1]
                if run["summary"]["prefix_cache_hit_requests"]:
                    raise RuntimeError("Prefix cache contamination")
                name = f"{system}-c{args.concurrency}-{workload}-r{repeat}.json"
                (args.output/name).write_text(json.dumps(run,indent=2,default=str))
                result["trials"].append(dict(system=system,workload=workload,repeat=repeat,path=name,
                                              summary=run["summary"]))
                save()
                print(json.dumps(dict(system=system,c=args.concurrency,workload=workload,repeat=repeat,
                    tps=run["summary"]["output_tokens_per_s"])),flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage",choices=["correctness","serving"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gate",type=Path)
    parser.add_argument("--systems",nargs="+",default=["eager","never","second4","second2","second8"])
    parser.add_argument("--concurrency",type=int,choices=[1,2,4],default=1)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--pilot",action="store_true")
    args = parser.parse_args()
    before = hashes()
    if args.stage=="serving":
        gate = json.loads(args.gate.read_text())
        if not gate.get("correctness_passed") or not gate["normal_exit"] or gate["pilot"]:
            raise RuntimeError("Full correctness gate required")
        if gate["source_hashes"]!=before:
            raise RuntimeError("Sources changed after correctness gate")
    args.output.mkdir(parents=True,exist_ok=False)
    result = dict(command=[sys.executable,*sys.argv],stage=args.stage,pilot=args.pilot,
        config=CONFIG,systems=args.systems,concurrency=args.concurrency,source_hashes=before,
        git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        versions={n:importlib.metadata.version(n) for n in ("torch","flash-attn","triton","transformers")},
        cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
        environment={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","NCCL_DEBUG","TORCH_DISABLE_ADDR2LINE","OMP_NUM_THREADS")},
        gpu=subprocess.check_output(["nvidia-smi"],text=True),
        topology=subprocess.check_output(["nvidia-smi","topo","-m"],text=True),
        target=frozen.TARGET,draft=frozen.DRAFT,reference=frozen.REFERENCE,
        target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        draft_revision="3d13517724e81cb409ddf1d4650772ec52f1e18e",
        correctness=[],warmups=[],trials=[],normal_exit=False)

    def save():
        (args.output/"manifest.json").write_text(json.dumps(result,indent=2,default=str))

    engine = None
    try:
        if args.stage=="correctness":
            from benchmarks.serving.eagle3_phase44b_audit import AuditRunner
            import nanovllm.speculative.graph_runner as module
            module.GraphConcurrentModelRunner = AuditRunner
        if args.systems==["ordinary"]:
            engine = LLM(frozen.TARGET,**CONFIG)
        else:
            engine = ConcurrentLLMEngine(frozen.TARGET,draft_path=frozen.DRAFT,reference_path=frozen.REFERENCE,
                speculative_length=3,audit=False,target_graph_config=policy("second4"),**CONFIG)
        if args.stage=="correctness":
            correctness(engine,args,result,save)
        else:
            serving(engine,args,result,save)
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
            result["source_regression"] = before!=hashes()
            save()


if __name__=="__main__":
    main()
