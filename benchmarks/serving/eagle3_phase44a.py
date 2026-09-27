"""Phase 4.4A bounded correctness and target-only timing driver."""
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
from benchmarks.serving.eagle3_graph_target import GraphRunner
from benchmarks.serving.eagle3_phase42 import TARGET, DRAFT, REFERENCE, cleanup_state


def frozen_hashes():
    paths = list(Path("nanovllm").rglob("*.py")) + list(Path("tests").glob("test_*.py"))
    for directory in ("benchmarks/eagle3-phase4", "benchmarks/eagle3-phase4_2", "benchmarks/eagle3-phase4_3"):
        paths += [p for p in Path(directory).rglob("*") if p.is_file()]
    paths += [p for p in Path("benchmarks/serving").glob("*phase4*.py") if "44" not in p.name]
    paths += [p for p in Path("docs").glob("EAGLE3_PHASE4*.md") if "4_4" not in p.name]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def prompts(tokenizer, contexts):
    seed = tokenizer.encode("A careful engineer checks state transitions, numerical results, and reproducible experiments. "*180,
                            add_special_tokens=False)
    return [[5100+i*17]+(seed[i*7:]+seed)*4 for i in range(len(contexts))]


def admit(engine, contexts, output=24):
    seqs = []
    for tokens,length in zip(prompts(engine.tokenizer,contexts),contexts):
        engine.add_request(tokens[:length],SamplingParams(max_tokens=output,ignore_eos=True,temperature=1e-9))
        seqs.append(engine.scheduler.waiting[-1])
    return seqs


def clean(engine):
    state = cleanup_state(engine,"speculative")
    ranks = engine.model_runner.call("graph_control","cleanup")
    if any(state.values()) or any(r["targets"] for r in ranks):
        raise RuntimeError(f"State leak: {state}, {ranks}")
    return dict(host=state,ranks=ranks)


def generation(engine, contexts, mode):
    runner = engine.model_runner
    runner.call("graph_control","mode",dict(mode=mode,reset_index=True))
    seqs = admit(engine,contexts)
    mapping = {seq.seq_id:i for i,seq in enumerate(seqs)}
    commits, statuses = [], []
    original_call = runner.call

    def call(method,*args):
        value = original_call(method,*args)
        if method == "eagle3_batch" and args[0] in ("verify","commit"):
            ranks = value["ranks"] if args[0]=="verify" else value
            statuses.append(dict(operation=args[0],ranks=[[
                [mapping[row[0]],*row[1:]] for row in rank] for rank in ranks]))
        return value

    runner.call = call
    try:
        while not engine.is_finished():
            engine.step()
            step = engine.speculative_coordinator.last_step
            if step.get("kind")=="decode":
                rows = []
                for entry in step["entries"]:
                    row = asdict(entry)
                    row["seq_id"] = mapping[row["seq_id"]]
                    rows.append(row)
                commits.append(rows)
    finally:
        runner.call = original_call
    state = clean(engine)
    result = dict(contexts=contexts,mode=mode,commits=commits,statuses=statuses,
                  outputs=[list(s.completion_token_ids) for s in seqs],cleanup=state)
    runner.call("graph_control","clear")
    result["released"] = clean(engine)
    return result


def live_verification(engine, contexts):
    seqs = admit(engine,contexts,output=24)
    while engine.scheduler.waiting:
        engine.step()
    coordinator = engine.speculative_coordinator
    manager = engine.scheduler.block_manager
    entries = []
    for seq in seqs:
        state = coordinator.requests[seq.seq_id]
        target = engine.model_runner._eagle_batch_states[seq.seq_id]
        proposal = coordinator.drafts[seq.seq_id].propose(target["features"],target["tokens"],3,owner=seq.seq_id)
        tx = manager.reserve_rows(seq,state.target_cursor,4,generation=state.generation)
        state.transaction = tx
        entries.append(dict(seq_id=seq.seq_id,generation=state.generation,
                            proposals=proposal,blocks=list(seq.block_table)))
    return entries


def verify_rollback(runner,entries):
    value = runner.call("eagle3_batch","verify",dict(entries=entries))
    runner.call("eagle3_batch","rollback",dict(entries=[
        dict(seq_id=e["seq_id"],generation=e["generation"]) for e in entries]))
    return value


def correctness(engine,args,result,save):
    runner = engine.model_runner
    for batch in args.batches:
        cases = [[length]*batch for length in (255,256,257,768,1024)]
        if batch > 1:
            cases.append([255,256,257,768][:batch])
        if args.pilot:
            cases = cases[:1]
        for contexts in cases:
            eager = generation(engine,contexts,"eager")
            graph = generation(engine,contexts,"audit")
            checks = {name:eager[name]==graph[name] for name in ("commits","statuses","outputs")}
            result["correctness"].append(dict(contexts=contexts,eager=eager,graph=graph,checks=checks))
            save()
            print(json.dumps(dict(contexts=contexts,checks=checks)),flush=True)
            if not all(checks.values()):
                raise RuntimeError("End-to-end graph disagreement")
    # A one-rank cache veto forces symmetric eager, then the same graph replays.
    runner.call("graph_control","mode",dict(mode="graph",reset_index=True))
    entries = live_verification(engine,[255])
    runner.call("graph_control","prepare",dict(entries=entries))
    transitions = []
    for veto in (False,True,False):
        runner.call("graph_control","mode",dict(mode="graph",mask_rank1=veto))
        value = verify_rollback(runner,entries)
        transitions.append(value)
    if not all(value==transitions[0] for value in transitions):
        raise RuntimeError("Graph/eager/graph fallback changed verification")
    result["symmetric_fallback"] = dict(passed=True,verifications=transitions)
    engine.scheduler.close_all()
    runner.call("graph_control","clear")
    result["fallback_cleanup"] = clean(engine)
    # Raise only after both ranks completed graph verification, inside acceptance.
    runner.call("graph_control","mode",dict(mode="audit",reset_index=True))
    admit(engine,[255])
    while engine.scheduler.waiting:
        engine.step()
    original_accept = coordinator_module.accept_greedy

    def intentional(*args,**kwargs):
        raise RuntimeError("phase44a intentional acceptance exception")

    coordinator_module.accept_greedy = intentional
    try:
        engine.step()
        raise AssertionError("Expected intentional exception")
    except RuntimeError as error:
        if "phase44a intentional acceptance exception" not in str(error):
            raise
        result["exception_cleanup"] = clean(engine)
    finally:
        coordinator_module.accept_greedy = original_accept
    runner.call("graph_control","clear")
    runner.call("graph_control","flush")
    result["released_cleanup"] = clean(engine)
    result["correctness_passed"] = True


def performance(engine,args,result,save):
    runner = engine.model_runner
    for batch in args.batches:
        for context in (256,768):
            runner.call("graph_control","mode",dict(mode="graph",reset_index=True))
            entries = live_verification(engine,[context]*batch)
            runner.call("graph_control","prepare",dict(entries=entries))
            cell = dict(rows=4*batch,contexts=[context]*batch,samples=[],trace_samples=[])
            reference = None
            for mode in ("eager","coordinated-eager","graph"):
                runner.call("graph_control","mode",dict(mode=mode))
                for _ in range(4):
                    value = verify_rollback(runner,entries)
                    if reference is None:
                        reference = value
                    if value != reference:
                        raise RuntimeError("Performance warmup changed target IDs/status")
            for repeat in range(args.repeats):
                modes = ["eager","coordinated-eager","graph"]
                order = modes[repeat%3:]+modes[:repeat%3]
                if repeat%2:
                    order.reverse()
                for mode in order:
                    runner.call("graph_control","mode",dict(mode=mode,measure=True))
                    for sample in range(args.samples):
                        index = len(runner.timing_records)
                        start = perf_counter_ns()
                        value = runner.call("eagle3_batch","verify",dict(entries=entries))
                        host_ns = perf_counter_ns()-start
                        runner.call("eagle3_batch","rollback",dict(entries=[
                            dict(seq_id=e["seq_id"],generation=e["generation"]) for e in entries]))
                        if value != reference:
                            raise RuntimeError("Timed target IDs/status changed")
                        cell["samples"].append(dict(repeat=repeat,sample=sample,mode=mode,
                                                     timing_index=index,host_rpc_ns=host_ns))
                    runner.call("graph_control","flush")
            for mode in ("eager","graph"):
                runner.call("graph_control","mode",dict(mode=mode,measure=True))
                path = str(args.output/f"M{4*batch}-ctx{context}-{mode}-trace")
                runner.call("graph_control","trace_start",dict(path=path))
                for _ in range(2):
                    index = len(runner.timing_records)
                    verify_rollback(runner,entries)
                    cell["trace_samples"].append(dict(mode=mode,timing_index=index))
                runner.call("graph_control","trace_stop")
                runner.call("graph_control","flush")
            engine.scheduler.close_all()
            cell["cleanup_before_release"] = clean(engine)
            runner.call("graph_control","clear")
            cell["cleanup_after_release"] = clean(engine)
            result["performance"].append(cell)
            save()
            print(json.dumps(dict(rows=4*batch,context=context,completed=True)),flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage",choices=["correctness","performance"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gate",type=Path)
    parser.add_argument("--batches",type=int,nargs="+",default=[1,2,4])
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--samples",type=int,default=10)
    args = parser.parse_args()
    if args.stage=="performance":
        if args.gate is None:
            raise ValueError("Performance requires completed correctness manifest")
        gate = json.loads(args.gate.read_text())
        if not gate.get("correctness_passed") or not gate.get("normal_exit") or gate["pilot"]:
            raise RuntimeError("Correctness gate is not closed for all M")
        coverage = {tuple(c["contexts"]) for c in gate["correctness"]}
        required = {(length,)*batch for batch in (1,2,4) for length in (255,256,257,768,1024)}
        if not required.issubset(coverage):
            raise RuntimeError("Incomplete shape/context correctness coverage")
    args.output.mkdir(parents=True,exist_ok=False)
    config = dict(tensor_parallel_size=2,enforce_eager=True,max_model_len=2048,
                  max_num_batched_tokens=2048,max_num_seqs=4,gpu_memory_utilization=.70,
                  scheduler_policy="original")
    before = frozen_hashes()
    result = dict(stage=args.stage,pilot=args.pilot,prototype_only=True,command=[sys.executable,*sys.argv],
        config=config,target=TARGET,draft=DRAFT,reference=REFERENCE,
        target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        draft_revision="3d13517724e81cb409ddf1d4650772ec52f1e18e",
        git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        versions={name:importlib.metadata.version(name) for name in ("torch","flash-attn","triton","transformers")},
        cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
        gpu=subprocess.check_output(["nvidia-smi"],text=True),
        topology=subprocess.check_output(["nvidia-smi","topo","-m"],text=True),
        environment={k:os.environ.get(k) for k in ("NCCL_DEBUG","NCCL_DEBUG_SUBSYS","CUDA_VISIBLE_DEVICES","HF_HUB_OFFLINE")},
        frozen_hashes_before=before,correctness=[],performance=[],normal_exit=False)
    sources = {str(p):p.read_text() for p in [Path(__file__),
        Path("benchmarks/serving/eagle3_graph_target.py"),Path("benchmarks/serving/eagle3_graph_key.py")]}
    (args.output/"sources.json").write_text(json.dumps(sources,indent=2))
    result["prototype_hashes"] = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
    if args.stage=="performance":
        if gate["prototype_hashes"] != result["prototype_hashes"] or gate["frozen_hashes_after"] != before:
            raise RuntimeError("Source changed since correctness gate")

    def save():
        (args.output/"manifest.json").write_text(json.dumps(result,indent=2,default=str))

    engine = None
    try:
        engine_module.ConcurrentModelRunner = GraphRunner
        engine = engine_module.ConcurrentLLMEngine(TARGET,draft_path=DRAFT,reference_path=REFERENCE,
                                                    speculative_length=3,audit=False,**config)
        engine.model_runner.call("graph_control","init",dict(raw=str(args.output)))
        if args.stage=="correctness":
            correctness(engine,args,result,save)
        else:
            performance(engine,args,result,save)
        result["final_cleanup"] = clean(engine)
        engine.model_runner.call("graph_control","flush")
    except Exception:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        try:
            if engine is not None:
                engine.model_runner.call("graph_control","flush")
                atexit.unregister(engine.exit)
                engine.exit()
            result["normal_exit"] = True
        except Exception:
            result["exit_failure"] = traceback.format_exc()
            raise
        finally:
            result["frozen_hashes_after"] = frozen_hashes()
            result["source_regression"] = before != result["frozen_hashes_after"]
            save()


if __name__ == "__main__":
    main()
