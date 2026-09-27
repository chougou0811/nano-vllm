"""Phase 4.2 fresh-process serving benchmark for ordinary vs concurrent EAGLE."""
import argparse
import atexit
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns
import traceback

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.speculative.concurrent_engine import ConcurrentLLMEngine
from nanovllm.speculative.draft_state import DraftState
import nanovllm.speculative.coordinator as coordinator_module


TARGET = "/root/autodl-tmp/models/Qwen3-14B"
DRAFT = "/root/autodl-tmp/models/Qwen3-14B_eagle3"
REFERENCE = "/root/autodl-tmp/references/eagle-pinned"
WORKLOADS = (
    "short-short", "short-long", "long-short", "long-long",
    "mixed-prompt", "mixed-output",
)
SHAPES = {
    "short-short": ((64,), (32,)),
    "short-long": ((64,), (128,)),
    "long-short": ((768,), (32,)),
    "long-long": ((768,), (128,)),
    "mixed-prompt": ((64, 192, 384, 768), (64,)),
    "mixed-output": ((256,), (16, 32, 64, 128)),
}
PHRASES = (
    "A database engineer traces a transaction through logs and verifies each invariant. ",
    "The compiler lowers a numerical program while preserving shapes and dependencies. ",
    "An astronomer compares repeated observations before reporting a physical conclusion. ",
    "A distributed system records ownership epochs, queue state, and committed progress. ",
    "The experiment uses fixed inputs and retains every measurement including outliers. ",
    "A kernel launch is only one component of end to end serving latency. ",
)


def distribution(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {key: None for key in ("mean", "min", "max", "p50", "p95", "p99")} | {"count": 0}

    def percentile(q):
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return dict(count=len(values), mean=sum(values) / len(values),
                min=values[0], max=values[-1], p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99))


@dataclass
class Request:
    request_id: int
    prompt: list[int]
    output_limit: int
    sequence: object | None = None
    sequence_id: int | None = None
    arrival_ns: int | None = None
    first_scheduled_ns: int | None = None
    first_token_ns: int | None = None
    finish_ns: int | None = None
    token_times_ns: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    schedule_steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    initial_cached_tokens: int | None = None


class TimingCollector:
    """Benchmark-only host spans; no new CUDA synchronization."""

    def __init__(self, engine, system):
        self.engine = engine
        self.system = system
        self.scheduler = engine.scheduler
        self.runner = engine.model_runner
        self.block_manager = self.scheduler.block_manager
        self.current = None
        self.steps = []
        self.by_sequence = {}
        self.originals = []
        self._install()

    def _replace(self, obj, name, value):
        had_local = name in vars(obj)
        original = getattr(obj, name)
        self.originals.append((obj, name, original, had_local))
        setattr(obj, name, value)
        return original

    def _add(self, key, value):
        if self.current is not None:
            self.current[key] = self.current.get(key, 0) + value

    def _install(self):
        original_schedule = self._replace(self.scheduler, "schedule", None)

        def schedule():
            before = len(self.block_manager.used_block_ids)
            start = perf_counter_ns()
            seqs, is_prefill = original_schedule()
            end = perf_counter_ns()
            self.current.update(
                schedule_ns=end - start,
                is_prefill=is_prefill,
                selected_sequence_ids=[seq.seq_id for seq in seqs],
                batch_size=len(seqs),
                scheduled_tokens=[seq.num_scheduled_tokens for seq in seqs],
                context_lengths=[seq.num_tokens for seq in seqs],
                used_blocks_before=before,
                used_blocks_after_schedule=len(self.block_manager.used_block_ids),
            )
            for seq in seqs:
                request = self.by_sequence[seq.seq_id]
                if request.first_scheduled_ns is None:
                    request.first_scheduled_ns = end - self.current["origin_ns"]
                    request.initial_cached_tokens = seq.num_cached_tokens
                request.schedule_steps += 1
                request.prefill_steps += int(is_prefill)
                request.decode_steps += int(not is_prefill)
            return seqs, is_prefill

        self.scheduler.schedule = schedule

        original_call = self._replace(self.runner, "call", None)

        def call(method, *args):
            start = perf_counter_ns()
            try:
                return original_call(method, *args)
            finally:
                elapsed = perf_counter_ns() - start
                if self.current is None:
                    pass
                elif self.system == "ordinary" and method == "run":
                    self._add("target_run_ns", elapsed)
                    self._add("target_forward_count", 1)
                elif method == "eagle3_batch":
                    operation = args[0]
                    key = {
                        "prefill": "target_prefill_ns",
                        "verify": "target_verify_ns",
                        "commit": "target_commit_rpc_ns",
                        "close": "target_close_rpc_ns",
                        "begin": "target_control_rpc_ns",
                        "seed": "target_control_rpc_ns",
                        "rollback": "target_rollback_rpc_ns",
                    }.get(operation, "target_other_rpc_ns")
                    self._add(key, elapsed)
                    if operation == "verify":
                        self._add("target_forward_count", 1)

        self.runner.call = call

        postprocess_names = (["postprocess"] if self.system == "ordinary"
                             else ["postprocess_prefill", "postprocess_decode"])
        for name in postprocess_names:
            original = self._replace(self.scheduler, name, None)

            def measured(*args, _original=original, **kwargs):
                start = perf_counter_ns()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self._add("postprocess_ns", perf_counter_ns() - start)

            setattr(self.scheduler, name, measured)

        if self.system == "speculative":
            for name, key in (("reserve_rows", "reserve_ns"),
                              ("commit_rows", "block_commit_ns"),
                              ("rollback_rows", "block_rollback_ns")):
                original = self._replace(self.block_manager, name, None)

                def measured(*args, _original=original, _key=key, **kwargs):
                    start = perf_counter_ns()
                    try:
                        value = _original(*args, **kwargs)
                        if _key == "reserve_ns":
                            active = sum(len(tx.new_block_ids)
                                         for tx in self.block_manager.transactions.values())
                            self.current["peak_tentative_blocks"] = max(
                                self.current.get("peak_tentative_blocks", 0), active
                            )
                        return value
                    finally:
                        self._add(_key, perf_counter_ns() - start)

                setattr(self.block_manager, name, measured)

            original_propose = DraftState.propose
            self.original_draft_propose = original_propose

            def propose(state, *args, **kwargs):
                start = perf_counter_ns()
                value = original_propose(state, *args, **kwargs)
                self._add("draft_ns", perf_counter_ns() - start)
                self._add("draft_catchup_ns", state.last_metrics["conditioning_ns"])
                self._add("draft_forward_count", state.last_metrics["draft_forwards"])
                self._add("draft_tokens_processed", state.last_metrics["draft_tokens_processed"])
                return value

            DraftState.propose = propose

            original_accept = coordinator_module.accept_greedy
            self.original_accept = original_accept

            def accept(*args, **kwargs):
                start = perf_counter_ns()
                try:
                    return original_accept(*args, **kwargs)
                finally:
                    self._add("accept_ns", perf_counter_ns() - start)

            coordinator_module.accept_greedy = accept

    def bind(self, request):
        self.by_sequence[request.sequence_id] = request

    def begin_step(self, origin):
        if self.current is not None:
            raise RuntimeError("Nested engine step")
        self.current = dict(step_id=len(self.steps), origin_ns=origin)

    def finish_step(self, start, end):
        step = self.current
        step["step_ns"] = end - start
        step["used_blocks_after"] = len(self.block_manager.used_block_ids)
        step["gpu_allocated"] = torch.cuda.memory_allocated()
        step["gpu_reserved"] = torch.cuda.memory_reserved()
        if self.system == "speculative":
            last = self.engine.speculative_coordinator.last_step
            step["target_query_tokens"] = last.get("target_query_tokens", 0)
            if last.get("kind") == "decode":
                commits = last["entries"]
                step["proposed_tokens"] = sum(item.actual_k for item in commits)
                step["accepted_tokens"] = sum(item.accepted_length for item in commits)
                step["committed_decode_tokens"] = sum(
                    len(item.committed_token_ids) for item in commits
                )
                step["verification_count"] = 1
                step["clipping_reasons"] = [item.clipping_reason for item in commits]
            else:
                step.update(proposed_tokens=0, accepted_tokens=0,
                            committed_decode_tokens=0, verification_count=0,
                            clipping_reasons=[])
        if self.system == "ordinary":
            known_keys = ("schedule_ns", "target_run_ns", "postprocess_ns")
            step["accept_commit_ns"] = 0
        elif step["is_prefill"]:
            known_keys = ("schedule_ns", "target_prefill_ns", "target_control_rpc_ns",
                          "postprocess_ns")
            step["accept_commit_ns"] = 0
        else:
            known_keys = (
                "schedule_ns", "draft_ns", "reserve_ns", "target_verify_ns",
                "accept_ns", "target_commit_rpc_ns", "block_commit_ns", "postprocess_ns",
            )
            step["accept_commit_ns"] = sum(step.get(key, 0) for key in (
                "accept_ns", "target_commit_rpc_ns", "block_commit_ns", "postprocess_ns",
            ))
        known = sum(step.get(key, 0) for key in known_keys)
        step["sync_other_ns"] = max(0, step["step_ns"] - known)
        del step["origin_ns"]
        self.steps.append(step)
        self.current = None
        return step

    def close(self):
        if self.system == "speculative":
            DraftState.propose = self.original_draft_propose
            coordinator_module.accept_greedy = self.original_accept
        for obj, name, original, had_local in reversed(self.originals):
            if had_local:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def build_requests(tokenizer, workload, concurrency, repeat, warmup=False):
    count = concurrency if warmup else max(4, 2 * concurrency)
    prompt_shapes, output_shapes = SHAPES[workload]
    phrase = tokenizer.encode(
        PHRASES[WORKLOADS.index(workload)] * 160,
        add_special_tokens=False,
    )
    requests = []
    phase = 100 if warmup else repeat
    for index in range(count):
        prompt_length = prompt_shapes[index % len(prompt_shapes)]
        output_length = output_shapes[index % len(output_shapes)]
        if warmup:
            output_length = min(output_length, 2 + 2 * (index % max(1, concurrency)))
        offset = (phase * 37 + index * 53) % len(phrase)
        stream = phrase[offset:] + phrase[:offset]
        prompt = (stream * ((prompt_length // len(stream)) + 1))[:prompt_length]
        unique = 500 + ((WORKLOADS.index(workload) * 1000 + phase * 41 + index * 17) %
                        (tokenizer.vocab_size - 1000))
        prompt[0] = unique
        requests.append(Request(index, prompt, output_length))
    return requests


def cleanup_state(engine, system):
    manager = engine.scheduler.block_manager
    state = dict(
        waiting=len(engine.scheduler.waiting),
        running=len(engine.scheduler.running),
        used_blocks=len(manager.used_block_ids),
        transactions=len(getattr(manager, "transactions", {})),
    )
    if system == "speculative":
        coordinator = engine.speculative_coordinator
        state.update(host_requests=len(coordinator.requests), host_drafts=len(coordinator.drafts),
                     rank0_targets=len(getattr(engine.model_runner, "_eagle_batch_states", {})))
    return state


def run_closed_loop(engine, system, requests, concurrency):
    if not engine.is_finished():
        raise RuntimeError("Closed-loop trial requires idle engine")
    collector = TimingCollector(engine, system)
    torch.cuda.reset_peak_memory_stats()
    origin = perf_counter_ns()
    active = {}
    next_request = 0

    def admit(request):
        nonlocal next_request
        request.arrival_ns = perf_counter_ns() - origin
        engine.add_request(
            request.prompt,
            SamplingParams(temperature=1e-9, max_tokens=request.output_limit,
                           ignore_eos=True),
        )
        request.sequence = engine.scheduler.waiting[-1]
        request.sequence_id = request.sequence.seq_id
        collector.bind(request)
        active[request.sequence_id] = request
        next_request += 1

    try:
        while next_request < len(requests) and len(active) < concurrency:
            admit(requests[next_request])
        while active:
            collector.begin_step(origin)
            start = perf_counter_ns()
            outputs, _ = engine.step()
            end = perf_counter_ns()
            collector.finish_step(start, end)
            for request in list(active.values()):
                tokens = request.sequence.completion_token_ids
                old = len(request.output_token_ids)
                committed = tokens[old:]
                request.output_token_ids.extend(committed)
                request.token_times_ns.extend([end - origin] * len(committed))
                if committed and request.first_token_ns is None:
                    request.first_token_ns = end - origin
            for sequence_id, output in outputs:
                request = active.pop(sequence_id)
                if list(output) != request.output_token_ids:
                    raise RuntimeError("Engine output disagrees with committed Sequence")
                request.finish_ns = end - origin
                if len(output) != request.output_limit:
                    raise RuntimeError("Fixed-length request ended early")
            while next_request < len(requests) and len(active) < concurrency:
                admit(requests[next_request])
        finish = perf_counter_ns() - origin
    finally:
        collector.close()

    rows = []
    for request in requests:
        times = request.token_times_ns
        itls = [(right - left) / 1e6 for left, right in zip(times, times[1:])]
        rows.append(dict(
            request_id=request.request_id,
            sequence_id=request.sequence_id,
            prompt_length=len(request.prompt),
            output_limit=request.output_limit,
            arrival_ns=request.arrival_ns,
            first_scheduled_ns=request.first_scheduled_ns,
            first_token_ns=request.first_token_ns,
            finish_ns=request.finish_ns,
            output_token_ids=request.output_token_ids,
            token_times_ns=times,
            ttft_ms=(request.first_token_ns - request.arrival_ns) / 1e6,
            e2e_ms=(request.finish_ns - request.arrival_ns) / 1e6,
            tpot_ms=((times[-1] - times[0]) / (len(times) - 1) / 1e6
                     if len(times) > 1 else None),
            itl_ms=itls,
            schedule_steps=request.schedule_steps,
            prefill_steps=request.prefill_steps,
            decode_steps=request.decode_steps,
            initial_cached_tokens=request.initial_cached_tokens,
        ))
    total_tokens = sum(len(row["output_token_ids"]) for row in rows)
    step_ns = sum(step["step_ns"] for step in collector.steps)
    decode = [step for step in collector.steps if not step["is_prefill"]]
    summary = dict(
        duration_s=finish / 1e9,
        requests=len(rows),
        output_tokens=total_tokens,
        output_tokens_per_s=total_tokens * 1e9 / finish,
        requests_per_s=len(rows) * 1e9 / finish,
        latency=dict(
            ttft_ms=distribution(row["ttft_ms"] for row in rows),
            e2e_ms=distribution(row["e2e_ms"] for row in rows),
            tpot_ms=distribution(row["tpot_ms"] for row in rows),
            itl_ms=distribution(value for row in rows for value in row["itl_ms"]),
        ),
        wall_clock=dict(
            step_ns=step_ns,
            schedule_ns=sum(step.get("schedule_ns", 0) for step in collector.steps),
            draft_ns=sum(step.get("draft_ns", 0) for step in collector.steps),
            draft_catchup_ns=sum(step.get("draft_catchup_ns", 0) for step in collector.steps),
            reserve_ns=sum(step.get("reserve_ns", 0) for step in collector.steps),
            target_verify_ns=sum(step.get("target_verify_ns", 0) for step in collector.steps),
            target_prefill_ns=sum(step.get("target_prefill_ns", 0) for step in collector.steps),
            target_run_ns=sum(step.get("target_run_ns", 0) for step in collector.steps),
            accept_commit_ns=sum(step.get("accept_commit_ns", 0) for step in collector.steps),
            sync_other_ns=sum(step.get("sync_other_ns", 0) for step in collector.steps),
            driver_refill_ns=max(0, finish - step_ns),
        ),
        scheduler=dict(
            decode_batch_size=distribution(step["batch_size"] for step in decode),
            target_query_tokens=distribution(step.get("target_query_tokens", 0) for step in decode),
            max_used_blocks=max((step["used_blocks_after"] for step in collector.steps), default=0),
            peak_tentative_blocks=max((step.get("peak_tentative_blocks", 0)
                                       for step in collector.steps), default=0),
        ),
        speculation=dict(
            proposed_tokens=sum(step.get("proposed_tokens", 0) for step in decode),
            accepted_tokens=sum(step.get("accepted_tokens", 0) for step in decode),
            committed_decode_tokens=sum(step.get("committed_decode_tokens", 0) for step in decode),
            verifications=sum(step.get("verification_count", 0) for step in decode),
            target_forwards=sum(step.get("target_forward_count", 0) for step in collector.steps),
            draft_forwards=sum(step.get("draft_forward_count", 0) for step in collector.steps),
            draft_tokens_processed=sum(step.get("draft_tokens_processed", 0)
                                       for step in collector.steps),
        ),
        memory=dict(
            rank0_peak_allocated=torch.cuda.max_memory_allocated(),
            rank0_peak_reserved=torch.cuda.max_memory_reserved(),
            rank0_max_step_allocated=max(step["gpu_allocated"] for step in collector.steps),
            rank0_max_step_reserved=max(step["gpu_reserved"] for step in collector.steps),
            kv_capacity_blocks=len(engine.scheduler.block_manager.blocks),
        ),
        cleanup=cleanup_state(engine, system),
        prefix_cache_hit_requests=sum((row["initial_cached_tokens"] or 0) > 0 for row in rows),
    )
    spec = summary["speculation"]
    spec["acceptance_rate"] = (spec["accepted_tokens"] / spec["proposed_tokens"]
                               if spec["proposed_tokens"] else None)
    spec["accepted_per_verification"] = (spec["accepted_tokens"] / spec["verifications"]
                                         if spec["verifications"] else None)
    spec["effective_outputs_per_verification"] = (
        spec["committed_decode_tokens"] / spec["verifications"]
        if spec["verifications"] else None
    )
    wall = summary["wall_clock"]
    wall["draft_fraction"] = wall["draft_ns"] / finish
    wall["verify_fraction"] = wall["target_verify_ns"] / finish
    return dict(summary=summary, requests=rows, steps=collector.steps)


def nvidia_memory():
    value = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    return [line.strip() for line in value.splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["ordinary", "speculative"], required=True)
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    config = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=1024,
                  max_num_batched_tokens=2048, max_num_seqs=4,
                  gpu_memory_utilization=.70, scheduler_policy="original")
    manifest = dict(
        system=args.system,
        concurrency=args.concurrency,
        repeats=5,
        workloads=list(WORKLOADS),
        config=config,
        target=TARGET,
        draft=DRAFT if args.system == "speculative" else None,
        reference=REFERENCE if args.system == "speculative" else None,
        target_revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        draft_revision=("3d13517724e81cb409ddf1d4650772ec52f1e18e"
                        if args.system == "speculative" else None),
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        torch=torch.__version__, cuda=torch.version.cuda,
        environment={key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "NCCL_DEBUG", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "PYTHONPATH",
        )},
        workload_order="rotate by repeat; reverse odd repeats",
        outlier_policy="retain all",
        timing_scope="in-process pretokenized closed-loop arrival to committed Sequence output",
    )
    engine = None
    result = dict(manifest=manifest, warmups=[], trials=[], normal_exit=False)
    try:
        if args.system == "ordinary":
            engine = LLM(TARGET, **config)
        else:
            engine = ConcurrentLLMEngine(
                TARGET, draft_path=DRAFT, reference_path=REFERENCE,
                speculative_length=3, audit=False, **config,
            )
        manifest["gpu_before_warmup"] = nvidia_memory()
        for workload in WORKLOADS:
            warm = build_requests(engine.tokenizer, workload, args.concurrency, 0, warmup=True)
            run = run_closed_loop(engine, args.system, warm, args.concurrency)
            result["warmups"].append(dict(
                workload=workload,
                output_tokens=run["summary"]["output_tokens"],
                decode_batches=sorted({step["batch_size"] for step in run["steps"]
                                       if not step["is_prefill"]}),
                cleanup=run["summary"]["cleanup"],
            ))
        manifest["gpu_after_warmup"] = nvidia_memory()
        for repeat in range(5):
            order = list(WORKLOADS[repeat:] + WORKLOADS[:repeat])
            if repeat % 2:
                order.reverse()
            for workload in order:
                torch.manual_seed(4200 + repeat)
                requests = build_requests(
                    engine.tokenizer, workload, args.concurrency, repeat, warmup=False
                )
                run = run_closed_loop(engine, args.system, requests, args.concurrency)
                run.update(workload=workload, repeat=repeat,
                           execution_index=len(result["trials"]))
                result["trials"].append(run)
                print(json.dumps(dict(system=args.system, concurrency=args.concurrency,
                                      workload=workload, repeat=repeat,
                                      output_tps=run["summary"]["output_tokens_per_s"])),
                      flush=True)
        manifest["gpu_after_measurement"] = nvidia_memory()
        result["final_cleanup"] = cleanup_state(engine, args.system)
        if any(result["final_cleanup"].values()):
            raise RuntimeError(f"Final cleanup failed: {result['final_cleanup']}")
        if any(trial["summary"]["prefix_cache_hit_requests"] for trial in result["trials"]):
            raise RuntimeError("Prefix cache contaminated measured workload")
        result["normal_exit"] = True
    except Exception:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(result, indent=2, default=str))
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


if __name__ == "__main__":
    main()
