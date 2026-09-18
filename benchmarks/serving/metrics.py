from dataclasses import asdict
import math


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(count=0, mean=None, min=None, max=None, p50=None, p95=None, p99=None)

    def percentile(q):
        position = (len(values) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return dict(count=len(values), mean=sum(values)/len(values), min=values[0],
                max=values[-1], p50=percentile(.5), p95=percentile(.95), p99=percentile(.99))


def request_metrics(record, ttft_slo_ms=None, tpot_slo_ms=None):
    def elapsed(end, start):
        return None if end is None or start is None else (end - start) / 1e6

    times = record.token_times_ns
    itls = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    tpot = (times[-1] - times[0]) / (len(times)-1) / 1e6 if len(times) > 1 else None
    values = dict(
        actual_output_length=len(record.output_token_ids),
        queue_delay_ms=elapsed(record.first_scheduled_ns, record.arrival_ns),
        ingress_queue_delay_ms=elapsed(record.admitted_ns, record.arrival_ns),
        scheduler_queue_delay_ms=elapsed(record.first_scheduled_ns, record.admitted_ns),
        submission_ms=elapsed(record.enqueued_ns, record.admitted_ns),
        engine_queue_delay_ms=elapsed(record.first_scheduled_ns, record.enqueued_ns),
        ttft_ms=elapsed(record.first_token_ns, record.arrival_ns),
        tpot_ms=tpot, itl_ms=itls,
        e2e_ms=elapsed(record.finish_ns, record.arrival_ns) if record.status == "completed" else None,
        slo_met=None,
    )
    if ttft_slo_ms is not None or tpot_slo_ms is not None:
        values["slo_met"] = (
            record.status == "completed" and bool(times) and
            (ttft_slo_ms is None or values["ttft_ms"] <= ttft_slo_ms) and
            (tpot_slo_ms is None or tpot is None or tpot <= tpot_slo_ms)
        )
    return values


def summarize(result, ttft_slo_ms=None, tpot_slo_ms=None):
    rows = [dict(**asdict(r), **request_metrics(r, ttft_slo_ms, tpot_slo_ms))
            for r in result.records]
    completed = [r for r in rows if r["status"] == "completed"]
    offered = [r for r in rows if r["arrival_ns"] <= result.window_end_ns]
    first_arrival = min(r["arrival_ns"] for r in rows)
    duration_s = max(0, result.window_end_ns - first_arrival) / 1e9
    output_tokens = sum(r["actual_output_length"] for r in offered)
    metrics = {name: distribution(r[name] for r in completed) for name in [
        "queue_delay_ms", "ingress_queue_delay_ms", "scheduler_queue_delay_ms",
        "submission_ms", "engine_queue_delay_ms",
        "ttft_ms", "tpot_ms", "e2e_ms",
    ]}
    metrics["itl_ms"] = distribution(t for r in completed for t in r["itl_ms"])
    slo_enabled = ttft_slo_ms is not None or tpot_slo_ms is not None
    good = sum(r["slo_met"] is True for r in offered) if slo_enabled else None
    summary = dict(
        schema_version=2, complete=result.complete, error=result.error,
        arrival_mode=result.arrival_mode, origin_perf_counter_ns=result.origin_ns,
        scope="in_process_pretokenized_token_commit", arrival_model="deterministic_logical",
        percentile_method="linear", latency_population="completed_requests",
        queue_delay_definition="first_scheduled - logical_arrival",
        planned_requests=len(rows), arrived_requests=len(offered), completed_requests=len(completed),
        failed_requests=sum(r["status"] == "failed" for r in rows),
        timed_out_requests=sum(r["status"] == "timeout" for r in rows),
        not_arrived_requests=sum(r["status"] == "not_arrived" for r in rows),
        window_start_ns=first_arrival, window_end_ns=result.window_end_ns, duration_s=duration_s,
        output_tokens=output_tokens, output_tokens_include_eos=True,
        request_throughput_rps=len(completed)/duration_s if duration_s else None,
        output_token_throughput_tps=output_tokens/duration_s if duration_s else None,
        latency=metrics, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
        slo_good_requests=good,
        slo_violation_rate=(len(offered)-good)/len(offered) if slo_enabled and offered else None,
        slo_goodput_rps=good/duration_s if slo_enabled and duration_s else None,
        num_engine_steps=len(result.steps),
        max_decode_batch=max((len(s["request_ids"]) for s in result.steps if not s["is_prefill"]), default=0),
        observer_bookkeeping_ns=result.observer_bookkeeping_ns, observer_calls=result.observer_calls,
        observer_bookkeeping_fraction_of_window=result.observer_bookkeeping_ns/(duration_s*1e9) if duration_s else None,
        observer_cost_scope="proxy bookkeeping only; excludes delegated scheduler work and driver",
        prefix_cache_blocks=sum(r.prefix_cache_blocks for r in result.records),
        initial_prefix_cache_blocks=sum(r.initial_prefix_cache_blocks for r in result.records),
        prefix_cache_hit_requests=sum(r.initial_prefix_cache_blocks > 0 for r in result.records),
        kv_allocation_events=sum(r.num_kv_allocations for r in result.records),
        max_waiting=max((st.get("queue_before", {}).get("waiting", 0) for st in result.steps), default=0),
        max_waiting_after_schedule=max((st.get("queue_after_schedule", {}).get("waiting", 0) for st in result.steps), default=0),
        max_running=max((st.get("queue_before", {}).get("running", 0) for st in result.steps), default=0),
        step_latency_by_batch={str(b): distribution(st["step_latency_ms"] for st in result.steps
                              if not st["is_prefill"] and len(st["request_ids"]) == b and "step_latency_ms" in st)
                              for b in sorted({len(st["request_ids"]) for st in result.steps if not st["is_prefill"]})},
    )
    return summary, rows
