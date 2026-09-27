"""Small Phase 4.1 concurrent EAGLE correctness validation."""
import argparse
import atexit
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns
import traceback

import torch

from nanovllm import SamplingParams
from nanovllm.speculative.concurrent_engine import ConcurrentLLMEngine
from nanovllm.speculative.session import generate


TARGET = "/root/autodl-tmp/models/Qwen3-14B"
DRAFT = "/root/autodl-tmp/models/Qwen3-14B_eagle3"
REFERENCE = "/root/autodl-tmp/references/eagle-pinned"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, default=str))

    config = dict(
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.70,
        scheduler_policy="original",
    )
    save("manifest.json", dict(
        config=config,
        target=TARGET,
        draft=DRAFT,
        reference=REFERENCE,
        target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        draft_revision="3d13517724e81cb409ddf1d4650772ec52f1e18e",
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=subprocess.check_output(["nvidia-smi"], text=True),
        environment={key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "NCCL_DEBUG", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "TMPDIR",
        )},
    ))
    sources = {}
    for path in sorted(Path("nanovllm/speculative").glob("*.py")):
        if path.name in {
            "batch.py", "batched_runtime.py", "block_transactions.py",
            "concurrent_engine.py", "coordinator.py", "scheduler_adapter.py",
        }:
            sources[str(path)] = path.read_text()
    save("sources.json", sources)

    engine = ConcurrentLLMEngine(
        TARGET,
        draft_path=DRAFT,
        reference_path=REFERENCE,
        speculative_length=3,
        audit=True,
        **config,
    )
    checks = {}
    runs = []

    def state_clean():
        block_manager = engine.scheduler.block_manager
        return dict(
            waiting=len(engine.scheduler.waiting),
            running=len(engine.scheduler.running),
            used_blocks=len(block_manager.used_block_ids),
            transactions=len(block_manager.transactions),
            host_requests=len(engine.speculative_coordinator.requests),
            host_drafts=len(engine.speculative_coordinator.drafts),
            rank0_targets=len(getattr(engine.model_runner, "_eagle_batch_states", {})),
        )

    def run_concurrent(name, cases, inject=None, accept_pattern=None):
        ids = []
        for prompt, maximum, ignore_eos in cases:
            before = set(engine.speculative_coordinator.requests)
            engine.add_request(
                prompt,
                SamplingParams(max_tokens=maximum, ignore_eos=ignore_eos),
            )
            ids.append((set(engine.speculative_coordinator.requests) - before).pop())
        started = perf_counter_ns()
        outputs, steps = {}, []
        injected = False
        proposal_controls_installed = False
        while not engine.is_finished():
            emitted, token_work = engine.step()
            last = engine.speculative_coordinator.last_step
            entry = dict(kind=last.get("kind"), token_work=token_work,
                         batch_size=last.get("batch_size"),
                         target_query_tokens=last.get("target_query_tokens"))
            if last.get("kind") == "decode":
                entry.update(
                    proposal_ns=last["proposal_ns"],
                    verification_ns=last["verification_ns"],
                    commits=[asdict(item) for item in last["entries"]],
                )
            steps.append(entry)
            outputs.update({seq_id: list(tokens) for seq_id, tokens in emitted})
            if (accept_pattern is not None and not proposal_controls_installed and
                    all(seq_id in engine.speculative_coordinator.drafts for seq_id in ids)):
                vocabulary = engine.model_runner.config.hf_config.vocab_size
                for seq_id, accepted in zip(ids, accept_pattern):
                    draft_state = engine.speculative_coordinator.drafts[seq_id]
                    original = draft_state.propose

                    def controlled(*call_args, _original=original,
                                   _accepted=accepted, **call_kwargs):
                        proposed = _original(*call_args, **call_kwargs)
                        if _accepted < len(proposed):
                            proposed[_accepted] = (proposed[_accepted] + 1) % vocabulary
                        return proposed

                    draft_state.propose = controlled
                proposal_controls_installed = True
            if inject is not None and not injected and emitted:
                prompt, maximum = inject
                before = set(engine.speculative_coordinator.requests)
                engine.add_request(prompt, SamplingParams(max_tokens=maximum, ignore_eos=True))
                ids.append((set(engine.speculative_coordinator.requests) - before).pop())
                injected = True
        record = dict(
            name=name,
            seq_ids=ids,
            outputs=[outputs[seq_id] for seq_id in ids],
            steps=steps,
            elapsed_ns=perf_counter_ns() - started,
            cleanup=state_clean(),
            gpu_allocated=torch.cuda.memory_allocated(),
            gpu_reserved=torch.cuda.memory_reserved(),
            gpu_peak=torch.cuda.max_memory_allocated(),
        )
        save(f"{name}.json", record)
        runs.append(record)
        return record

    try:
        seed = engine.tokenizer.encode(
            "A careful engineer checks every state transition before publishing a result. " * 240
        )

        # Concurrency=1 uses the frozen Phase 2 path as a detailed oracle.
        reference = generate(
            engine,
            seed[:63],
            draft_path=DRAFT,
            reference_path=REFERENCE,
            max_tokens=16,
            k=3,
            ignore_eos=True,
            audit=True,
            draft_state_mode="persistent",
        )
        one = run_concurrent("concurrency-1", [(seed[:63], 16, True)])
        reference_steps = [dict(
            proposed_token_ids=step["proposed_tokens"],
            target_token_ids=step["target_ids"],
            accepted_length=step["accepted"],
            committed_token_ids=step["committed_tokens"],
        ) for step in reference["steps"]]
        concurrent_steps = [dict(
            proposed_token_ids=list(commit["proposed_token_ids"]),
            target_token_ids=list(commit["target_token_ids"]),
            accepted_length=commit["accepted_length"],
            committed_token_ids=list(commit["committed_token_ids"]),
        ) for step in one["steps"] if step["kind"] == "decode"
          for commit in step["commits"]]
        checks["concurrency_1_outputs"] = one["outputs"][0] == reference["token_ids"]
        checks["concurrency_1_steps"] = concurrent_steps == reference_steps

        cases2 = [(seed[:47], 9, True), (seed[7:86], 13, True)]
        two_a = run_concurrent("concurrency-2-a", cases2)
        two_b = run_concurrent("concurrency-2-b", cases2)
        checks["concurrency_2_deterministic"] = two_a["outputs"] == two_b["outputs"]

        cases4 = [
            (seed[3:34], 3, True),
            (seed[11:66], 6, True),
            (seed[19:98], 9, True),
            (seed[27:130], 12, True),
        ]
        four_a = run_concurrent("concurrency-4-a", cases4)
        four_b = run_concurrent("concurrency-4-b", cases4)
        checks["concurrency_4_deterministic"] = four_a["outputs"] == four_b["outputs"]
        checks["independent_finish"] = [len(value) for value in four_a["outputs"]] == [3, 6, 9, 12]

        boundary_cases = [(seed[:length], 8, True) for length in (255, 256, 257)]
        boundary_a = run_concurrent("boundary-a", boundary_cases)
        boundary_b = run_concurrent("boundary-b", boundary_cases)
        checks["boundary_deterministic"] = boundary_a["outputs"] == boundary_b["outputs"]

        heterogeneous = run_concurrent(
            "heterogeneous-acceptance",
            [(seed[:length], 8, True) for length in (255, 256, 257, 511)],
            accept_pattern=(3, 0, 2, 1),
        )
        first_decode = next(step for step in heterogeneous["steps"]
                            if step["kind"] == "decode")
        checks["heterogeneous_acceptance"] = [
            commit["accepted_length"] for commit in first_decode["commits"]
        ] == [3, 0, 2, 1]

        replacement = run_concurrent(
            "continuous-replacement",
            [(seed[5:46], 3, True), (seed[13:82], 11, True)],
            inject=(seed[31:92], 5),
        )
        checks["continuous_replacement"] = [len(value) for value in replacement["outputs"]] == [3, 11, 5]

        eos_reference = generate(
            engine,
            seed[41:88],
            draft_path=DRAFT,
            reference_path=REFERENCE,
            max_tokens=4,
            k=3,
            ignore_eos=True,
            audit=True,
            draft_state_mode="persistent",
        )
        original_eos = engine.model_runner.config.eos
        engine.model_runner.config.eos = eos_reference["token_ids"][1]
        engine.scheduler.original.eos = eos_reference["token_ids"][1]
        eos = run_concurrent("eos", [(seed[41:88], 4, False)])
        engine.model_runner.config.eos = original_eos
        engine.scheduler.original.eos = original_eos
        checks["eos"] = eos["outputs"] == [eos_reference["token_ids"][:2]]
        checks["eos_decode_transaction"] = any(
            step["kind"] == "decode" for step in eos["steps"]
        )

        # Fail a proposal after prefill; the engine must clear every owner/page.
        engine.add_request(seed[:33], SamplingParams(max_tokens=8, ignore_eos=True))
        engine.step()
        draft_state = next(iter(engine.speculative_coordinator.drafts.values()))
        draft_state.propose = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("intentional phase4 exception")
        )
        try:
            engine.step()
        except RuntimeError as error:
            checks["intentional_exception"] = str(error) == "intentional phase4 exception"
        else:
            checks["intentional_exception"] = False
        checks["exception_cleanup"] = all(value == 0 for value in state_clean().values())
        checks["all_run_cleanup"] = all(
            all(value == 0 for value in run["cleanup"].values()) for run in runs
        )
        checks["prefix_cache_unpublished"] = not engine.scheduler.block_manager.hash_to_block_id
        save("summary.json", dict(checks=checks, runs=[run["name"] for run in runs],
                                  final_cleanup=state_clean()))
        if not all(checks.values()):
            raise RuntimeError("Phase 4.1 correctness check failed; raw records retained")
    except Exception:
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


if __name__ == "__main__":
    main()
