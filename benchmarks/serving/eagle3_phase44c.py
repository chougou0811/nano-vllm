"""Additive shadow-key instrumentation; frozen production execution is untouched."""
import argparse
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


ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=1024,
              max_num_batched_tokens=2048, max_num_seqs=4,
              gpu_memory_utilization=.70, scheduler_policy="original")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def inventory():
    result = {}
    for directory in ("nanovllm", "tests", "benchmarks", "docs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                if path.suffix in (".py", ".md", ".json", ".csv", ".diff"):
                    result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def requests_for(tokenizer, workload, count):
    from benchmarks.serving.eagle3_phase42 import PHRASES, SHAPES, WORKLOADS, Request
    prompts, outputs = SHAPES[workload]
    phrase = tokenizer.encode(PHRASES[WORKLOADS.index(workload)] * 160, add_special_tokens=False)
    requests = []
    for index in range(count):
        length = prompts[index % len(prompts)]
        offset = index * 53 % len(phrase)
        stream = phrase[offset:] + phrase[:offset]
        prompt = (stream * (length // len(stream) + 1))[:length]
        prompt[0] = 500 + ((WORKLOADS.index(workload) * 1000 + index * 17) %
                           (tokenizer.vocab_size - 1000))
        requests.append(Request(index, prompt, outputs[index % len(outputs)]))
    return requests


class ShadowObserver:
    def __init__(self, engine, requests, concurrency, enabled=True):
        import torch
        self.engine, self.requests, self.concurrency = engine, requests, concurrency
        self.enabled = enabled
        self.events, self.closes, self.milestones = [], [], {}
        self.completed, self.step_index, self.observer_ns = 0, 0, 0
        runner = engine.model_runner
        cfg = runner.config.hf_config
        self.key_kwargs = dict(block_size=runner.block_size,
            model_identity=digest(cfg.to_dict()), dtype=str(cfg.dtype),
            device_class=str(torch.cuda.get_device_capability()),
            tp_size=runner.world_size, training=runner.model.training,
            feature_layers=(2, cfg.num_hidden_layers//2, cfg.num_hidden_layers-3))
        self.original_call, self.original_step = runner.call, engine.step
        self.call_local, self.step_local = "call" in vars(runner), "step" in vars(engine)
        runner.call, engine.step = self.call, self.step

    def call(self, method, *args):
        if method == "eagle3_batch" and args[0] == "verify" and self.enabled:
            from nanovllm.speculative.batched_runtime import verification_layout
            from nanovllm.speculative.graph_policy import make_key
            start = perf_counter_ns()
            entries = args[1]["entries"]
            layout = verification_layout(entries, self.engine.model_runner._eagle_batch_states,
                                         self.engine.model_runner.block_size)
            key = make_key(layout, **self.key_kwargs)
            mapping = {r.sequence_id: r.request_id for r in self.requests if r.sequence_id is not None}
            request_ids = [mapping[e["seq_id"]] for e in entries]
            q = [b-a for a,b in zip(layout["cu_seqlens_q"], layout["cu_seqlens_q"][1:])]
            self.events.append(dict(index=len(self.events), step=self.step_index,
                completed_before=self.completed, key=asdict(key) if key else None,
                key_id=digest(asdict(key)) if key else None, M=sum(q), q_lengths=q,
                exact_max_k=max(c+n for c,n in zip(layout["context_lens"], q)),
                table_width=len(layout["block_tables"][0]),
                context_tuple=layout["context_lens"], request_ids=request_ids,
                waves=[i//self.concurrency for i in request_ids],
                output_limits=[self.requests[i].output_limit for i in request_ids],
                prompt_lengths=[len(self.requests[i].prompt) for i in request_ids]))
            self.observer_ns += perf_counter_ns() - start
        result = self.original_call(method, *args)
        if method == "eagle3_batch" and args[0] == "close":
            if len(result) != 2 or result[0] != result[1] or not all(row[-1] == 1 for row in result[0]):
                raise AssertionError("Rank close/zero acknowledgment mismatch")
            self.closes.extend(row[0] for row in result[0])
        return result

    def step(self):
        result = self.original_step()
        self.completed += len(result[0])
        for horizon in (32,128,512):
            if self.completed >= horizon and str(horizon) not in self.milestones:
                self.milestones[str(horizon)] = dict(completed=self.completed,
                    verifications=len(self.events), through_step=self.step_index)
        self.step_index += 1
        return result

    def close(self):
        if self.call_local:
            self.engine.model_runner.call = self.original_call
        else:
            del self.engine.model_runner.call
        if self.step_local:
            self.engine.step = self.original_step
        else:
            del self.engine.step


def trial(engine, workload, concurrency, count, observe=True):
    from benchmarks.serving.eagle3_phase42 import run_closed_loop, cleanup_state
    requests = requests_for(engine.tokenizer, workload, count)
    observer = ShadowObserver(engine, requests, concurrency, observe)
    try:
        run = run_closed_loop(engine, "speculative", requests, concurrency)
    finally:
        observer.close()
    assert not any(cleanup_state(engine, "speculative").values())
    assert run["summary"]["prefix_cache_hit_requests"] == 0
    assert len(observer.closes) == count and len(set(observer.closes)) == count
    run.update(workload=workload, concurrency=concurrency, request_count=count,
               inputs=[dict(prompt=r.prompt, output_limit=r.output_limit) for r in requests],
               events=observer.events, milestones=observer.milestones,
               observer_ns=observer.observer_ns, rank_close_count=len(observer.closes),
               all_rank_close_zero=True)
    return run


def signature(run):
    return dict(outputs=[r["output_token_ids"] for r in run["requests"]],
        steps=[{key:s.get(key) for key in ("is_prefill", "batch_size", "scheduled_tokens",
            "context_lengths", "target_query_tokens", "proposed_tokens", "accepted_tokens",
            "committed_tokens")} for s in run["steps"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "shadow", "bounded"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1,2,4), default=1)
    parser.add_argument("--workload")
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        dump(args.output / "before.json", inventory())
        return
    import torch
    from benchmarks.serving import eagle3_phase42 as frozen
    from nanovllm.speculative.concurrent_engine import ConcurrentLLMEngine
    from nanovllm.speculative.graph_policy import GraphCacheConfig
    manifest = dict(stage=args.stage, command=sys.argv, config=CONFIG,
        source_hashes=inventory(), head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        versions={n:importlib.metadata.version(n) for n in ("torch","flash-attn","triton","transformers")},
        gpu=subprocess.check_output(["nvidia-smi"],text=True),
        topology=subprocess.check_output(["nvidia-smi","topo","-m"],text=True),
        environment={k:os.environ.get(k) for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","NCCL_DEBUG")},
        target=frozen.TARGET, draft=frozen.DRAFT, reference=frozen.REFERENCE,
        target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        draft_revision="3d13517724e81cb409ddf1d4650772ec52f1e18e", trials=[], normal_exit=False)
    engine = None
    try:
        graph_config = GraphCacheConfig(max_event_records=200000) if args.stage=="bounded" else None
        engine = ConcurrentLLMEngine(frozen.TARGET, draft_path=frozen.DRAFT,
            reference_path=frozen.REFERENCE, speculative_length=3, audit=False,
            target_graph_config=graph_config, **CONFIG)
        if args.stage == "shadow":
            # Output/step parity is checked before any long trial in this fresh process.
            off = trial(engine,"mixed-output",args.concurrency,max(4,2*args.concurrency),False)
            on = trial(engine,"mixed-output",args.concurrency,max(4,2*args.concurrency),True)
            dump(args.output/"observer-off.json",off)
            dump(args.output/"observer-on.json",on)
            assert signature(off)==signature(on), "Observer changed deterministic execution"
            manifest["observer_parity"] = True
            order = list(frozen.WORKLOADS)
            shift = (1,2,4).index(args.concurrency)*2
            order = order[shift:]+order[:shift]
            if args.workload:
                order = [args.workload]
            for workload in order:
                warm = frozen.build_requests(engine.tokenizer,workload,args.concurrency,0,True)
                warm_result = frozen.run_closed_loop(engine,"speculative",warm,args.concurrency)
                dump(args.output/f"warmup-{workload}.json",warm_result)
                run = trial(engine,workload,args.concurrency,args.count)
                path = args.output/f"{workload}.json"
                dump(path,run)
                manifest["trials"].append(str(path))
                print(json.dumps(dict(completed=workload,c=args.concurrency,n=args.count,
                      eligible=sum(e["key"] is not None for e in run["events"]),
                      seconds=run["summary"]["duration_s"])),flush=True)
                dump(args.output/"manifest.json",manifest)
        else:
            assert args.workload
            for repeat in range(3):
                pair = {}
                for mode in (("eager","graph") if repeat%2==0 else ("graph","eager")):
                    engine.model_runner.call("graph_control","clear")
                    engine.model_runner.call("graph_control","reset",
                        dict(config=graph_config,enabled=mode=="graph"))
                    warm = frozen.build_requests(engine.tokenizer,args.workload,args.concurrency,0,True)
                    frozen.run_closed_loop(engine,"speculative",warm,args.concurrency)
                    engine.model_runner.call("graph_control","clear")
                    engine.model_runner.call("graph_control","reset",
                        dict(config=graph_config,enabled=mode=="graph"))
                    run = trial(engine,args.workload,args.concurrency,args.count)
                    run.update(mode=mode,repeat=repeat)
                    run["cache"] = engine.model_runner.call("graph_control","snapshot")
                    engine.model_runner.call("graph_control","clear")
                    run["released"] = engine.model_runner.call("graph_control","snapshot")
                    for rank in run["released"]:
                        assert not rank["target_states"] and not rank["live_entries"]
                        assert not rank["dropped_events"]
                    path = args.output/f"{mode}-{repeat}.json"
                    dump(path,run)
                    pair[mode] = signature(run)
                    manifest["trials"].append(str(path))
                    print(json.dumps(dict(mode=mode,repeat=repeat,c=args.concurrency,
                        seconds=run["summary"]["duration_s"])),flush=True)
                    dump(args.output/"manifest.json",manifest)
                assert pair["eager"] == pair["graph"], "Bounded graph parity failure"
        engine.exit()
        engine = None
        manifest["normal_exit"] = True
    except BaseException:
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        dump(args.output/"manifest.json",manifest)
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    main()
