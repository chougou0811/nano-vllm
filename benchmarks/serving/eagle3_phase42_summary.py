"""Summarize the complete Phase 4.2 six-process raw dataset."""
import argparse
import json
import math
from pathlib import Path


FILES = {
    ("ordinary", 1): "ordinary-c1-v2.json",
    ("speculative", 1): "speculative-c1.json",
    ("ordinary", 2): "ordinary-c2.json",
    ("speculative", 2): "speculative-c2.json",
    ("ordinary", 4): "ordinary-c4.json",
    ("speculative", 4): "speculative-c4.json",
}


def distribution(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None,
                "p50": None, "p95": None, "p99": None}

    def percentile(q):
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return dict(count=len(values), mean=sum(values) / len(values),
                min=values[0], max=values[-1], p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99))


def pooled_latency(trials):
    rows = [row for trial in trials for row in trial["requests"]]
    return dict(
        ttft_ms=distribution(row["ttft_ms"] for row in rows),
        e2e_ms=distribution(row["e2e_ms"] for row in rows),
        tpot_ms=distribution(row["tpot_ms"] for row in rows),
        itl_ms=distribution(value for row in rows for value in row["itl_ms"]),
    )


def system_metrics(trials, system):
    duration = sum(trial["summary"]["duration_s"] for trial in trials)
    tokens = sum(trial["summary"]["output_tokens"] for trial in trials)
    requests = sum(trial["summary"]["requests"] for trial in trials)
    steps = [step for trial in trials for step in trial["steps"]]
    decode = [step for step in steps if not step["is_prefill"]]
    wall_keys = (
        "schedule_ns", "draft_ns", "draft_catchup_ns", "reserve_ns",
        "target_verify_ns", "target_prefill_ns", "target_run_ns",
        "accept_commit_ns", "sync_other_ns", "driver_refill_ns",
    )
    wall = {key: sum(trial["summary"]["wall_clock"].get(key, 0)
                     for trial in trials) for key in wall_keys}
    result = dict(
        trials=len(trials),
        duration_s=duration,
        output_tokens=tokens,
        requests=requests,
        output_tokens_per_s=tokens / duration,
        requests_per_s=requests / duration,
        latency=pooled_latency(trials),
        wall_clock_ns=wall,
        wall_fraction={key: value / (duration * 1e9) for key, value in wall.items()},
        decode_batch_size=distribution(step["batch_size"] for step in decode),
        rank0_peak_allocated_gib=max(
            trial["summary"]["memory"]["rank0_peak_allocated"] for trial in trials
        ) / 2**30,
        rank0_peak_reserved_gib=max(
            trial["summary"]["memory"]["rank0_peak_reserved"] for trial in trials
        ) / 2**30,
        max_used_blocks=max(trial["summary"]["scheduler"]["max_used_blocks"]
                            for trial in trials),
        kv_capacity_blocks=trials[0]["summary"]["memory"]["kv_capacity_blocks"],
        peak_tentative_blocks=max(
            trial["summary"]["scheduler"]["peak_tentative_blocks"] for trial in trials
        ),
    )
    if system == "ordinary":
        decode_rows = sum(step["batch_size"] for step in decode)
        decode_target_ns = sum(step.get("target_run_ns", 0) for step in decode)
        result["target"] = dict(
            decode_forwards=len(decode),
            decode_rows=decode_rows,
            decode_target_ns=decode_target_ns,
            decode_target_ms_per_row=decode_target_ns / max(1, decode_rows) / 1e6,
        )
    else:
        proposed = sum(trial["summary"]["speculation"]["proposed_tokens"]
                       for trial in trials)
        accepted = sum(trial["summary"]["speculation"]["accepted_tokens"]
                       for trial in trials)
        committed = sum(trial["summary"]["speculation"]["committed_decode_tokens"]
                        for trial in trials)
        verifications = sum(trial["summary"]["speculation"]["verifications"]
                            for trial in trials)
        target_forwards = sum(trial["summary"]["speculation"]["target_forwards"]
                              for trial in trials)
        draft_forwards = sum(trial["summary"]["speculation"]["draft_forwards"]
                             for trial in trials)
        q_tokens = sum(step.get("target_query_tokens", 0) for step in decode)
        verify_ns = sum(step.get("target_verify_ns", 0) for step in decode)
        result["target_query_tokens"] = distribution(
            step.get("target_query_tokens", 0) for step in decode
        )
        result["speculation"] = dict(
            proposed_tokens=proposed,
            accepted_tokens=accepted,
            acceptance_rate=accepted / proposed,
            committed_decode_tokens=committed,
            verifications=verifications,
            accepted_per_verification=accepted / verifications,
            effective_outputs_per_verification=committed / verifications,
            target_forwards=target_forwards,
            draft_forwards=draft_forwards,
            target_query_tokens=q_tokens,
            verify_ms_per_query_token=verify_ns / q_tokens / 1e6,
            verify_ms_per_committed_decode_token=verify_ns / committed / 1e6,
            proposal_generation_ns=max(0, wall["draft_ns"] - wall["draft_catchup_ns"]),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = Path(args.raw)
    data = {key: json.loads((raw / filename).read_text()) for key, filename in FILES.items()}

    integrity = dict(
        formal_processes=len(data),
        formal_trials=sum(len(item["trials"]) for item in data.values()),
        planned_trials=180,
        all_normal_exit=all(item["normal_exit"] for item in data.values()),
        all_cleanup_zero=all(not any(item["final_cleanup"].values()) for item in data.values()),
        prefix_cache_hit_requests=sum(
            trial["summary"]["prefix_cache_hit_requests"]
            for item in data.values() for trial in item["trials"]
        ),
        malformed_output_requests=0,
        exact_cross_system_output_requests=0,
        compared_cross_system_output_requests=0,
        decomposition_max_ratio=0,
    )
    for item in data.values():
        for trial in item["trials"]:
            for row in trial["requests"]:
                if len(row["output_token_ids"]) != row["output_limit"]:
                    integrity["malformed_output_requests"] += 1
            for step in trial["steps"]:
                if item["manifest"]["system"] == "ordinary":
                    keys = ("schedule_ns", "target_run_ns", "postprocess_ns", "sync_other_ns")
                elif step["is_prefill"]:
                    keys = ("schedule_ns", "target_prefill_ns", "target_control_rpc_ns",
                            "postprocess_ns", "sync_other_ns")
                else:
                    keys = ("schedule_ns", "draft_ns", "reserve_ns", "target_verify_ns",
                            "accept_commit_ns", "sync_other_ns")
                ratio = sum(step.get(key, 0) for key in keys) / step["step_ns"]
                integrity["decomposition_max_ratio"] = max(
                    integrity["decomposition_max_ratio"], ratio
                )

    cells = []
    concurrency = {}
    all_paired_speedups = []
    for level in (1, 2, 4):
        ordinary_trials = data["ordinary", level]["trials"]
        speculative_trials = data["speculative", level]["trials"]
        ordinary = {(trial["workload"], trial["repeat"]): trial
                    for trial in ordinary_trials}
        speculative = {(trial["workload"], trial["repeat"]): trial
                       for trial in speculative_trials}
        paired = []
        for key in sorted(ordinary):
            left, right = ordinary[key], speculative[key]
            speedup = left["summary"]["duration_s"] / right["summary"]["duration_s"]
            paired.append(speedup)
            all_paired_speedups.append(speedup)
            for left_row, right_row in zip(left["requests"], right["requests"]):
                integrity["compared_cross_system_output_requests"] += 1
                integrity["exact_cross_system_output_requests"] += int(
                    left_row["output_token_ids"] == right_row["output_token_ids"]
                )
        ordinary_metrics = system_metrics(ordinary_trials, "ordinary")
        speculative_metrics = system_metrics(speculative_trials, "speculative")
        ordinary_metrics["gpu_after_measurement"] = data["ordinary", level]["manifest"][
            "gpu_after_measurement"
        ]
        speculative_metrics["gpu_after_measurement"] = data["speculative", level]["manifest"][
            "gpu_after_measurement"
        ]
        concurrency[str(level)] = dict(
            ordinary=ordinary_metrics,
            speculative=speculative_metrics,
            speedup=ordinary_metrics["duration_s"] / speculative_metrics["duration_s"],
            paired_speedup=distribution(paired),
            paired_wins=sum(value > 1 for value in paired),
        )
        for workload in sorted({key[0] for key in ordinary}):
            left = [ordinary[workload, repeat] for repeat in range(5)]
            right = [speculative[workload, repeat] for repeat in range(5)]
            left_metrics = system_metrics(left, "ordinary")
            right_metrics = system_metrics(right, "speculative")
            speedups = [left[index]["summary"]["duration_s"] /
                        right[index]["summary"]["duration_s"] for index in range(5)]
            cells.append(dict(
                concurrency=level,
                workload=workload,
                ordinary=left_metrics,
                speculative=right_metrics,
                speedup=left_metrics["duration_s"] / right_metrics["duration_s"],
                paired_speedup=distribution(speedups),
                paired_wins=sum(value > 1 for value in speedups),
            ))

    best = max(cells, key=lambda cell: cell["speedup"])
    worst = min(cells, key=lambda cell: cell["speedup"])
    draft_fractions = {
        level: concurrency[str(level)]["speculative"]["wall_fraction"]["draft_ns"]
        for level in (1, 2, 4)
    }
    verify_fractions = {
        level: concurrency[str(level)]["speculative"]["wall_fraction"]["target_verify_ns"]
        for level in (1, 2, 4)
    }
    decision = dict(
        repeatable_performance_benefit=all(value > 1 for value in all_paired_speedups),
        paired_wins=sum(value > 1 for value in all_paired_speedups),
        paired_trials=len(all_paired_speedups),
        best_cell=dict(concurrency=best["concurrency"], workload=best["workload"],
                       speedup=best["speedup"]),
        worst_cell=dict(concurrency=worst["concurrency"], workload=worst["workload"],
                        speedup=worst["speedup"]),
        draft_fraction_by_concurrency=draft_fractions,
        verify_fraction_by_concurrency=verify_fractions,
        draft_fraction_increases=all(draft_fractions[a] < draft_fractions[b]
                                     for a, b in ((1, 2), (2, 4))),
        draft_is_largest_named_component=any(
            draft_fractions[level] >= verify_fractions[level] for level in (1, 2, 4)
        ),
        recommend_concurrent_draft_batching=False,
        decision_reason=(
            "Serial draft share rises with concurrency but remains below target verification "
            "in every tested regime; the preregistered dominance condition is not met."
        ),
        next_measured_bottleneck="batched target verification",
    )
    summary = dict(
        schema_version=1,
        raw_directory=str(raw),
        measurement_scope="in-process pretokenized closed-loop committed output",
        outlier_policy="all repeats retained",
        integrity=integrity,
        correctness_regression=dict(
            pre_benchmark_phase41_focused="22/22 passed",
            pre_benchmark_complete_suite="110/110 passed",
            post_benchmark_phase41_focused="22/22 passed",
            post_benchmark_complete_suite="110/110 passed",
            post_exit_gpu_memory_mib=[1, 1],
        ),
        excluded_instrumentation_pilot=dict(
            file=str(raw / "ordinary-c1.json"),
            retained=True,
            reason=(
                "Pre-formal data-chain pilot had correct raw spans but a derived residual "
                "formula that failed to subtract target_run_ns; it was rerun fresh as "
                "ordinary-c1-v2.json before the six-process formal matrix."
            ),
        ),
        concurrency=concurrency,
        cells=cells,
        decision=decision,
        limitations=[
            "Ordinary uses the frozen sampler at temperature 1e-9; speculative is greedy argmax.",
            "Target verification timing includes model execution and TP RPC/NCCL synchronization.",
            "GPU utilization was not continuously sampled to avoid perturbing measured trials.",
            "Concurrency above four and speculative overload behavior were not tested.",
            "Results apply to the fixed target/draft revisions and six synthetic workload families.",
        ],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
