"""Additional correctness-only test: reuse a key across unrelated requests."""
import hashlib
from pathlib import Path
from time import perf_counter_ns

import torch

import benchmarks.serving.eagle3_phase44a as driver


def reuse(engine,args,result,save):
    runner = engine.model_runner
    original_prompts = driver.prompts
    result["metadata_reuse"] = []
    source = Path(__file__).read_bytes()
    (args.output/"reuse-source.py").write_bytes(source)
    result["reuse_source_sha256"] = hashlib.sha256(source).hexdigest()
    result["unprofiled_graph_replay_api"] = []
    original_replay = torch.cuda.CUDAGraph.replay

    def timed_replay(graph):
        start = perf_counter_ns()
        value = original_replay(graph)
        result["unprofiled_graph_replay_api"].append(dict(rows=4*batch,ns=perf_counter_ns()-start))
        return value

    torch.cuda.CUDAGraph.replay = timed_replay
    try:
        for batch in (1,2,4):
            identity = None
            layouts = []
            for variant in range(3):
                def changed(tokenizer,contexts):
                    tokens = original_prompts(tokenizer,contexts)
                    return [[(token+17*variant)%runner.config.hf_config.vocab_size for token in row]
                            for row in tokens]

                driver.prompts = changed
                contexts = [255]*batch if variant==0 else list(range(256-batch,256))
                runner.call("graph_control","mode",dict(mode="audit",reset_index=True))
                entries = driver.live_verification(engine,contexts)
                value = driver.verify_rollback(runner,entries)
                assert len(runner.fixed_graphs)==1
                obj = next(iter(runner.fixed_graphs.values()))
                if identity is None:
                    identity = id(obj)
                assert id(obj)==identity and obj.capture_count==1
                layouts.append(dict(variant=variant,contexts=contexts,entries=entries,
                                    verification=value,key_reused=variant>0))
                engine.scheduler.close_all()
                driver.clean(engine)
            assert layouts[0]["entries"]!=layouts[1]["entries"]
            runner.call("graph_control","clear")
            result["metadata_reuse"].append(dict(rows=4*batch,variants=layouts,
                                                 cleanup=driver.clean(engine)))
            save()
        runner.call("graph_control","flush")
        result["metadata_reuse_passed"] = True
    finally:
        driver.prompts = original_prompts
        torch.cuda.CUDAGraph.replay = original_replay


if __name__ == "__main__":
    driver.correctness = reuse
    driver.main()
