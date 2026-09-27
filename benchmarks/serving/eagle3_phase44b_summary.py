"""Offline full-matrix analysis; retains every Phase 4.4B trial."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics

from benchmarks.serving.eagle3_phase42 import distribution,WORKLOADS


def load(path):
    return json.loads(path.read_text())


def graph_metrics(run):
    if "graphs" not in run:
        return None
    rank = run["graphs"][0]
    events = [e for e in rank["events"] if e["kind"]=="verify"]
    # The post-release snapshot supplies clear/eviction metadata for every
    # lifetime; timed events still come from the pre-clear serving snapshot.
    captures = run["released"][0]["lifetimes"]
    if run["system"]=="eager":
        assert not events and not captures
        return None
    steps = [s for s in run["steps"] if not s["is_prefill"]]
    assert len(events)==len(steps)
    keys = defaultdict(list)
    request_keys = defaultdict(set)
    reasons = Counter()
    clipped = Counter()
    fallback_causes = Counter()
    observed = set()
    captured = set()
    non_context_shapes = set()
    table_shapes = set()
    new_context_only = new_table_width = 0
    capture_count = 0
    for event,step in zip(events,steps):
        event["endpoint_ns"] = step["target_verify_ns"]
        reasons[event["reason"]] += 1
        for reason in set(step["clipping_reasons"]):
            if reason:
                clipped[reason] += 1
        if event["key"] is not None:
            key = event["key"]
            without_context = json.dumps({k:v for k,v in key.items() if k!="max_k"},sort_keys=True)
            without_table = json.dumps({k:v for k,v in key.items() if k not in ("max_k","table_width")},sort_keys=True)
            if event["key_id"] not in observed:
                new_context_only += without_context in non_context_shapes
                new_table_width += (without_context not in non_context_shapes and without_table in table_shapes)
            non_context_shapes.add(without_context)
            table_shapes.add(without_table)
            if event["action"]!="replay":
                if event["key_id"] in captured:
                    fallback_causes["previously_captured_now_absent"] += 1
                elif event["key_id"] not in observed:
                    fallback_causes["first_exact_key_occurrence"] += 1
                elif capture_count>=rank["config"]["max_captures"]:
                    fallback_causes["capture_budget_exhausted"] += 1
                elif event["action"]=="capture":
                    fallback_causes["capture_call_returns_eager"] += 1
                else:
                    fallback_causes["policy_or_consensus_miss"] += 1
            observed.add(event["key_id"])
            keys[event["key_id"]].append(event)
            for seq_id in event["seq_ids"]:
                request_keys[seq_id].add(event["key_id"])
        elif any(step["clipping_reasons"]):
            fallback_causes["clipped_q"] += 1
        else:
            fallback_causes["unsupported_batch_or_shape"] += 1
        if event["action"]=="capture":
            capture_count += 1
            captured.add(event["key_id"])
    lifetimes = []
    for entry in captures:
        seen = keys[entry["key_id"]]
        eager = [e["endpoint_ns"] for e in seen if e["action"]=="eager"]
        hits = [e["endpoint_ns"] for e in seen if e["action"]=="replay"]
        saved = statistics.mean(eager)-statistics.mean(hits) if eager and hits else None
        z = entry["replay_count"]
        cost = entry["capture_ns"]
        if z==0:
            verdict = "never_reused_unprofitable"
        elif saved is None:
            verdict = "unidentified"
        else:
            verdict = "profitable" if saved>0 and z*saved>=cost else "unprofitable"
        lifetimes.append(dict(key_id=entry["key_id"],key=entry["key"],entry_id=entry["entry_id"],
            capture_ms=cost/1e6,replays=z,per_hit_saved_ms=None if saved is None else saved/1e6,
            break_even_replays=math.ceil(cost/saved) if saved is not None and saved>0 else None,
            verdict=verdict,allocated_delta=entry["allocated_delta"],reserved_delta=entry["reserved_delta"],
            capture_peak=entry["capture_peak"],created_step=entry["created_step"],last_used=entry["last_used"],
            release_reason=entry.get("release_reason"),released_step=entry.get("released_step"),
            release_ms=entry.get("release_ns",0)/1e6,
            eager_endpoint_ms=statistics.mean(eager)/1e6 if eager else None,
            hit_endpoint_ms=statistics.mean(hits)/1e6 if hits else None,
            matched_hit_speedup=statistics.mean(eager)/statistics.mean(hits) if eager and hits else None))
    eligible = sum(e["key"] is not None for e in events)
    hits = sum(e["action"]=="replay" for e in events)
    captured_key_verdicts = Counter()
    for key_id in captured:
        entries = [e for e in lifetimes if e["key_id"]==key_id]
        z = sum(e["replays"] for e in entries)
        cost = sum(e["capture_ms"] for e in entries)
        saving = entries[0]["per_hit_saved_ms"]
        verdict = ("never_reused_unprofitable" if not z else "unidentified" if saving is None
                   else "profitable" if saving>0 and saving*z>=cost else "unprofitable")
        captured_key_verdicts[verdict] += 1
    release = [e for e in rank["events"] if e["kind"]=="release"]
    graph_wall = sum(e["capture_ns"] for e in events)+sum(e["duration_ns"] for e in release)
    hit_steps = [s for e,s in zip(events,steps) if e["action"]=="replay"]
    return dict(verifications=len(events),eligible=eligible,ineligible=len(events)-eligible,
        hits=hits,misses=eligible-hits,eager_fallback=len(events)-hits,captures=len(captures),
        evictions=sum(e["reason"]=="eviction" for e in release),
        eligible_rate=eligible/len(events) if events else None,
        hit_rate=hits/len(events) if events else None,eligible_hit_rate=hits/eligible if eligible else None,
        eligible_miss_rate=(eligible-hits)/eligible if eligible else None,
        eager_fallback_rate=(len(events)-hits)/len(events) if events else None,
        unique_keys=len(keys),unique_keys_per_request=distribution(
            len(request_keys[r["sequence_id"]]) for r in run["requests"]),
        occurrences_per_key=distribution(len(rows) for rows in keys.values()),
        unique_max_k=distribution(rows[0]["key"]["max_k"] for rows in keys.values()),
        verification_rows=dict(Counter(e["key"]["rows"] for e in events if e["key"])),
        block_table_widths=dict(Counter(e["key"]["table_width"] for e in events if e["key"])),
        reasons=dict(reasons),clipping_reasons=dict(clipped),fallback_causes=dict(fallback_causes),
        new_key_context_only=new_context_only,new_key_table_width=new_table_width,
        captured_unique_keys=len(captured),
        ragged_q=sum(len(set(e["q_lengths"]))>1 for e in events),
        unseen_context_key=sum(e["key"] is not None and e["occurrence"]==1 for e in events),
        unsupported_uniform_batch=sum(e["key"] is None and all(q==4 for q in e["q_lengths"]) for e in events),
        capture_ms=sum(e["capture_ns"] for e in events)/1e6,
        eviction_ms=sum(e["duration_ns"] for e in release)/1e6,
        agreement_ms=sum(e["agreement_ns"] for e in events)/1e6,
        replay_ms=sum(e["replay_ns"] for e in events)/1e6,
        eager_forward_ms=sum(e["eager_ns"] for e in events)/1e6,
        graph_management_fraction=graph_wall/(run["summary"]["duration_s"]*1e9),
        replays_per_capture=hits/len(captures) if captures else None,
        replays_per_captured_unique_key=hits/len(captured) if captured else None,
        capture_to_replay=len(captures)/hits if hits else None,
        replay_distribution=distribution(e["replay_count"] for e in captures),
        lifetime_verdicts=dict(Counter(e["verdict"] for e in lifetimes)),lifetimes=lifetimes,
        captured_key_verdicts=dict(captured_key_verdicts),
        hit_endpoint_ms=distribution(e["endpoint_ns"]/1e6 for e in events if e["action"]=="replay"),
        hit_step_ns=sum(s["step_ns"] for s in hit_steps),
        hit_step_draft_ns=sum(s["draft_ns"] for s in hit_steps),
        hit_step_verify_ns=sum(s["target_verify_ns"] for s in hit_steps),
        peak_live_bytes=rank["peak_live_bytes"],peak_live_reserved=rank["peak_live_reserved"],
        memory_before_release=[r["memory"] for r in run["graphs"]],
        memory_after_release=[r["memory"] for r in run["released"]])


def aggregate(runs):
    duration = sum(r["summary"]["duration_s"] for r in runs)
    requests = [q for r in runs for q in r["requests"]]
    wall = {k:sum(r["summary"]["wall_clock"].get(k,0) for r in runs)
            for k in runs[0]["summary"]["wall_clock"] if k.endswith("_ns")}
    graph = [r["graph_metrics"] for r in runs if r["graph_metrics"] is not None]
    result = dict(trials=len(runs),duration_s=duration,requests=len(requests),
        tokens=sum(r["summary"]["output_tokens"] for r in runs),
        output_tokens_per_s=sum(r["summary"]["output_tokens"] for r in runs)/duration,
        requests_per_s=len(requests)/duration,
        latency={k:distribution(q[k] for q in requests) for k in ("ttft_ms","e2e_ms","tpot_ms")},
        wall_clock_ns=wall,wall_clock_fraction={k:v/1e9/duration for k,v in wall.items()},
        memory_peak_allocated=max(r["summary"]["memory"]["rank0_peak_allocated"] for r in runs),
        memory_peak_reserved=max(r["summary"]["memory"]["rank0_peak_reserved"] for r in runs))
    steps = [s for r in runs for s in r["steps"]]
    spec = {k:sum(r["summary"]["speculation"][k] for r in runs) for k in (
        "proposed_tokens","accepted_tokens","committed_decode_tokens","verifications",
        "target_forwards","draft_forwards","draft_tokens_processed")}
    for name,denominator in (("acceptance_rate","proposed_tokens"),("accepted_per_verification","verifications")):
        spec[name] = spec["accepted_tokens"]/spec[denominator] if spec[denominator] else None
    spec["effective_outputs_per_verification"] = (
        spec["committed_decode_tokens"]/spec["verifications"] if spec["verifications"] else None)
    captures = sum(g["captures"] for g in graph)
    spec.update(target_serving_forward_calls=len(steps),
                target_prefill_calls=sum(s["is_prefill"] for s in steps),
                target_extra_warmup_forwards=3*captures,
                target_capture_recordings=captures,
                target_executed_forward_calls=len(steps)+3*captures)
    result["speculation"] = spec
    result["scheduler"] = dict(
        batch_size=distribution(s["batch_size"] for s in steps if not s["is_prefill"]),
        target_query_tokens=distribution(s.get("target_query_tokens",0) for s in steps if not s["is_prefill"]),
        max_used_blocks=max(s["used_blocks_after"] for s in steps),
        peak_tentative_blocks=max(s.get("peak_tentative_blocks",0) for s in steps),
        capacity_blocks=sorted({r["summary"]["memory"]["kv_capacity_blocks"] for r in runs}))
    result["latency"]["itl_ms"] = distribution(t for q in requests for t in q["itl_ms"])
    if graph:
        totals = {k:sum(g[k] for g in graph) for k in ("verifications","eligible","ineligible","hits","misses",
            "eager_fallback","captures","evictions","unique_keys","capture_ms","eviction_ms","agreement_ms","replay_ms",
            "ragged_q","unseen_context_key","unsupported_uniform_batch","new_key_context_only",
            "new_key_table_width","captured_unique_keys","eager_forward_ms",
            "hit_step_ns","hit_step_draft_ns","hit_step_verify_ns")}
        totals.update(hit_rate=totals["hits"]/totals["verifications"],
            eligible_rate=totals["eligible"]/totals["verifications"],
            eligible_hit_rate=totals["hits"]/totals["eligible"] if totals["eligible"] else None,
            eligible_miss_rate=totals["misses"]/totals["eligible"] if totals["eligible"] else None,
            eager_fallback_rate=totals["eager_fallback"]/totals["verifications"],
            replays_per_capture=totals["hits"]/totals["captures"] if totals["captures"] else None,
            graph_management_fraction=(totals["capture_ms"]+totals["eviction_ms"])/1e3/duration,
            replay_distribution=distribution(e["replays"] for g in graph for e in g["lifetimes"]),
            lifetime_verdicts=dict(sum((Counter(g["lifetime_verdicts"]) for g in graph),Counter())),
            captured_key_verdicts=dict(sum((Counter(g["captured_key_verdicts"]) for g in graph),Counter())),
            peak_live_bytes=max(g["peak_live_bytes"] for g in graph),
            peak_live_reserved_proxy=max(g["peak_live_reserved"] for g in graph),
            unique_keys_per_trial=distribution(g["unique_keys"] for g in graph),
            capture_to_replay=totals["captures"]/totals["hits"] if totals["hits"] else None,
            cache_churn=totals["evictions"]/totals["captures"] if totals["captures"] else None,
            capture_distribution_ms=distribution(e["capture_ms"] for g in graph for e in g["lifetimes"]),
            matched_hit_speedup=distribution(e["matched_hit_speedup"] for g in graph for e in g["lifetimes"] if e["replays"]),
            fallback_causes=dict(sum((Counter(g["fallback_causes"]) for g in graph),Counter())),
            verification_rows=dict(sum((Counter(g["verification_rows"]) for g in graph),Counter())),
            block_table_widths=dict(sum((Counter(g["block_table_widths"]) for g in graph),Counter())),
            clipping_reasons=dict(sum((Counter(g["clipping_reasons"]) for g in graph),Counter())))
        totals["replays_per_captured_unique_key"] = totals["hits"]/totals["captured_unique_keys"] if totals["captured_unique_keys"] else None
        totals["hit_step_draft_fraction"] = totals["hit_step_draft_ns"]/totals["hit_step_ns"] if totals["hit_step_ns"] else None
        totals["hit_step_verify_fraction"] = totals["hit_step_verify_ns"]/totals["hit_step_ns"] if totals["hit_step_ns"] else None
        matched = [e for g in graph for e in g["lifetimes"] if e["replays"] and e["matched_hit_speedup"] is not None]
        eager_ms = sum(e["eager_endpoint_ms"]*e["replays"] for e in matched)
        hit_ms = sum(e["hit_endpoint_ms"]*e["replays"] for e in matched)
        hit_count = sum(e["replays"] for e in matched)
        totals["matched_hit_target_speedup"] = eager_ms/hit_ms if hit_ms else None
        totals["matched_hit_eager_endpoint_ms"] = eager_ms/hit_count if hit_count else None
        totals["matched_hit_graph_endpoint_ms"] = hit_ms/hit_count if hit_count else None
        result["graph"] = totals
    # Nested timers are not additive: catch-up is inside draft; cache work is
    # inside target verification. Form an explicit non-overlapping partition.
    partition = {k:wall[k] for k in ("schedule_ns","draft_ns","reserve_ns","target_prefill_ns",
                 "target_run_ns","accept_commit_ns","sync_other_ns","driver_refill_ns")}
    partition["prefill_control_and_postprocess_ns"] = sum(
        s.get("target_control_rpc_ns",0)+s.get("postprocess_ns",0) for s in steps if s["is_prefill"])
    if runs[0]["system"]=="ordinary":
        partition["ordinary_decode_postprocess_ns"] = sum(s.get("postprocess_ns",0) for s in steps if not s["is_prefill"])
    if graph:
        for label,field in (("capture_ns","capture_ms"),("eviction_ns","eviction_ms"),
                            ("graph_agreement_ns","agreement_ms"),("target_replay_forward_ns","replay_ms"),
                            ("target_eager_forward_ns","eager_forward_ms")):
            partition[label] = sum(g[field]*1e6 for g in graph)
        partition["verify_endpoint_other_ns"] = wall["target_verify_ns"]-sum(partition[k] for k in (
            "capture_ns","eviction_ns","graph_agreement_ns","target_replay_forward_ns","target_eager_forward_ns"))
        assert partition["verify_endpoint_other_ns"]>=0
    else:
        partition["target_verify_ns"] = wall["target_verify_ns"]
    assert abs(sum(partition.values())-duration*1e9)<100, "Wall breakdown must reconcile"
    result["wall_partition_ns"] = partition
    result["wall_partition_fraction"] = {k:v/1e9/duration for k,v in partition.items()}
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    runs = []
    manifests = []
    for path in sorted(args.raw.glob("run-*/manifest.json")):
        manifest = load(path)
        assert manifest["normal_exit"] and not manifest["source_regression"]
        manifests.append(dict(path=str(path),command=manifest["command"],versions=manifest["versions"],
                              config=manifest["config"],source_hashes=manifest["source_hashes"]))
        for trial in manifest["trials"]:
            run = load(path.parent/trial["path"])
            run["raw"] = str(path.parent/trial["path"])
            assert not any(run["summary"]["cleanup"].values())
            assert not run["summary"]["prefix_cache_hit_requests"]
            if "released" in run:
                assert all(r["live_entries"]==r["target_states"]==r["dropped_events"]==0 for r in run["released"])
            run["graph_metrics"] = graph_metrics(run)
            runs.append(run)
    assert len(runs)==420, f"Need complete preregistered matrix: {len(runs)}/420"
    result = dict(trials=len(runs),outliers_removed=0,manifests=manifests,cells=[],rollups=[],comparisons=[],
                  correctness=load(args.raw/"correctness-final/manifest.json"))
    gate = result["correctness"]
    result["correctness"] = {k:gate[k] for k in ("correctness_passed","normal_exit","source_regression","fallback_passed","exception_cleanup")}
    cases = []
    for case in gate["correctness"]:
        current = case["actual"]
        rank_signature = lambda audit: [(s["operation"],[(r["seq_id"],r["cursor"],r["token_count"],
            r["feature_sha"],r["phase"]) for r in s["rows"]]) for s in audit["states"]]
        assert rank_signature(current["audits"][0])==rank_signature(current["audits"][1])
        cases.append(dict(concurrency=case["concurrency"],lengths=case["lengths"],limits=case["limits"],
            checks=case["checks"],request_pairs=len(current["outputs"]),
            rank_comparisons=sum(len(a["checks"]) for a in current["audits"]),
            actions=dict(Counter(e["action"] for e in current["cache"][0]["events"] if e["kind"]=="verify"))))
    result["correctness"].update(cases=cases,committed_feature_rank_replication=True,
        request_pairs=sum(c["request_pairs"] for c in cases),rank_comparisons=sum(c["rank_comparisons"] for c in cases))
    flat,lifetimes = [],[]
    for run in runs:
        flat.append(dict(system=run["system"],concurrency=run["concurrency"],workload=run["workload"],repeat=run["repeat"],
            raw=run["raw"],duration_s=run["summary"]["duration_s"],output_tps=run["summary"]["output_tokens_per_s"],
            graph=run["graph_metrics"]))
        if run["graph_metrics"]:
            lifetimes.extend(dict(system=run["system"],concurrency=run["concurrency"],workload=run["workload"],
                repeat=run["repeat"],**entry) for entry in run["graph_metrics"]["lifetimes"])
    for system in ("ordinary","eager","never","second4","second2","second8"):
        for concurrency in (1,2,4):
            selected = [r for r in runs if r["system"]==system and r["concurrency"]==concurrency]
            result["rollups"].append(dict(system=system,concurrency=concurrency,**aggregate(selected)))
            for workload in WORKLOADS:
                cell = [r for r in selected if r["workload"]==workload]
                if cell:
                    assert len(cell)==5
                    result["cells"].append(dict(system=system,concurrency=concurrency,workload=workload,**aggregate(cell)))
    equality = Counter()
    for concurrency in (1,2,4):
        for workload in WORKLOADS:
            peers = {r["system"]:[x for x in runs if x["system"]==r["system"] and x["concurrency"]==concurrency
                     and x["workload"]==workload] for r in runs if r["concurrency"]==concurrency and r["workload"]==workload}
            for system in ("never","second4","second2","second8"):
                if system not in peers:
                    continue
                speeds = {base:[] for base in ("ordinary","eager")}
                for current in peers[system]:
                    for base in speeds:
                        reference = next(r for r in peers[base] if r["repeat"]==current["repeat"])
                        assert current["inputs"]==reference["inputs"]
                        equal = [a["output_token_ids"]==b["output_token_ids"] for a,b in zip(current["requests"],reference["requests"])]
                        equality[f"{system}_vs_{base}_pairs"] += len(equal)
                        equality[f"{system}_vs_{base}_equal"] += sum(equal)
                        if base=="eager" and not all(equal):
                            raise RuntimeError("Graph serving output differs from eager")
                        if base=="eager":
                            fields = ("is_prefill","batch_size","context_lengths","scheduled_tokens",
                                      "target_query_tokens","proposed_tokens","accepted_tokens",
                                      "committed_decode_tokens","clipping_reasons")
                            signature = lambda run: [tuple(s.get(k) for k in fields) for s in run["steps"]]
                            if signature(current)!=signature(reference):
                                raise RuntimeError("Graph serving schedule/acceptance structure differs from eager")
                            equality[f"{system}_vs_eager_step_structure_equal_trials"] += 1
                        speeds[base].append(reference["summary"]["duration_s"]/current["summary"]["duration_s"])
                result["comparisons"].append(dict(system=system,concurrency=concurrency,workload=workload,
                    paired_speedups=speeds,paired_distribution={k:distribution(v) for k,v in speeds.items()},
                    aggregate_speedup={base:sum(r["summary"]["duration_s"] for r in peers[base])/
                        sum(r["summary"]["duration_s"] for r in peers[system]) for base in speeds}))
    result["output_parity"] = dict(equality)
    controls = []
    for c in (1,2,4):
        path = args.raw/f"frozen-eager-c{c}.json"
        data = load(path)
        assert data["normal_exit"] and len(data["trials"])==30
        assert not any(data["final_cleanup"].values())
        for trial in data["trials"]:
            assert not any(trial["summary"]["cleanup"].values())
            assert not trial["summary"]["prefix_cache_hit_requests"]
            trial.update(system="default_eager",concurrency=c,graph_metrics=None)
            peer = next(r for r in runs if r["system"]=="eager" and r["concurrency"]==c
                        and r["workload"]==trial["workload"] and r["repeat"]==trial["repeat"])
            assert [r["output_token_ids"] for r in trial["requests"]]==[r["output_token_ids"] for r in peer["requests"]]
            controls.append(trial)
    result["default_runner_control"] = dict(trials=len(controls),output_parity=True,cells=[],rollups=[])
    for c in (1,2,4):
        selected = [r for r in controls if r["concurrency"]==c]
        result["default_runner_control"]["rollups"].append(dict(concurrency=c,**aggregate(selected)))
        for workload in WORKLOADS:
            current = [r for r in selected if r["workload"]==workload]
            default_time = sum(r["summary"]["duration_s"] for r in current)
            peers = {s:sum(r["summary"]["duration_s"] for r in runs if r["concurrency"]==c
                          and r["workload"]==workload and r["system"]==s) for s in ("eager","second4")}
            result["default_runner_control"]["cells"].append(dict(concurrency=c,workload=workload,
                graph_over_default_speedup=default_time/peers["second4"],
                disabled_over_default_speedup=default_time/peers["eager"],**aggregate(current)))
    result["total_measured_trials_including_controls"] = len(runs)+len(controls)
    result["cpu_regression"] = {}
    for name in ("post-focused","post-cpu"):
        log = args.raw/f"{name}.log"
        value = log.read_text()
        count = int(re.search(r"Ran (\d+) tests",value).group(1))
        passed = value.rstrip().endswith("\nOK")
        assert passed, f"CPU regression failed: {log}"
        result["cpu_regression"][name] = dict(tests=count,passed=passed,log=str(log))
    primary = [c for c in result["comparisons"] if c["system"]=="second4"]
    result["status"] = "complete; opt-in integration correct, cold exact-key capture not beneficial in tested cells"
    result["decision"] = dict(keep_eager_default=True,graph_remains_opt_in=True,
        primary_cells_with_aggregate_graph_over_eager_gain=sum(c["aggregate_speedup"]["eager"]>1 for c in primary),
        primary_paired_trials_with_graph_over_eager_gain=sum(v>1 for c in primary for v in c["paired_speedups"]["eager"]),
        draft_batching_supported_by_this_evidence=False,
        next_phase_started=False,git_commit_made=False,
        followup="Approval required: longer-horizon exact-key reuse, then an independent padded/bucketed-graph safety design if warranted")
    result["lifetime_amortization"] = lifetimes
    result["trial_table"] = flat
    result["metric_contract"] = dict(
        throughput="committed tokens / serving wall time, never sum(request E2E)",
        percentiles="pooled requests for TTFT/E2E/TPOT; pooled committed-token intervals for ITL",
        cache_hit_rate="replays / all verification batches; eligible_hit_rate has eligible denominator",
        key_counts="per-trial unique keys summed across trials, not global de-duplicated identities",
        capture_amortization="same-key noncapture eager versus replay endpoint means; observational estimate",
        memory="allocated deltas include first-use engine workspace; reserved deltas are allocator proxies, not ownership",
        partition="disjoint rank-0 host wall endpoints; not isolated GPU kernel time; draft catch-up is nested",
        graph_cold_start="cache, occurrence history and capture allowance reset each measured trial",
        eager_baseline="same-process opt-in runner with GraphCache disabled; frozen eager arithmetic and transactions")
    result["metric_contract"]["target_forwards"] = (
        "legacy Phase4.2 target_forwards excludes speculative prefill; explicit serving/executed counts add prefill "
        "and three capture warmups. Capture recordings are separate, not additional executed GPU forwards.")
    result["metric_contract"]["rank1_peak"] = (
        "rank0 peak is trial-local; the frozen collector resets only rank0 allocator counters. "
        "Rank1 peak snapshots can include earlier trials; report process maxima and instantaneous/release values, "
        "not purported isolated per-trial rank1 peaks.")
    (args.output/"summary.json").write_text(json.dumps(result,indent=2))
    with (args.output/"trials.csv").open("w") as file:
        rows = [{k:v for k,v in r.items() if k!="graph"} for r in flat]
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output/"capture-lifetimes.csv").open("w") as file:
        rows = [{k:v for k,v in entry.items() if k not in ("key","capture_peak")} for entry in lifetimes]
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cells = {(r["system"],r["concurrency"],r["workload"]):r for r in result["cells"]}
    lines = ["# Serving Throughput", "", "Five measured repeats per cell; all capture/eviction costs included.", "",
             "| c | Workload | Ordinary tok/s | Eager EAGLE | Never | Graph-4 | Graph/eager | Graph/ordinary |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in (1,2,4):
        for workload in WORKLOADS:
            t = {s:cells[s,c,workload]["output_tokens_per_s"] for s in ("ordinary","eager","never","second4")}
            lines.append(f"| {c} | {workload} | {t['ordinary']:.2f} | {t['eager']:.2f} | {t['never']:.2f} | "
                         f"{t['second4']:.2f} | {t['second4']/t['eager']:.3f}x | {t['second4']/t['ordinary']:.3f}x |")
    (args.output/"throughput-table.md").write_text("\n".join(lines)+"\n")
    lines = ["# Request and Token Latencies", "", "All values in milliseconds: P50 / P95 / P99. No outlier removal.", "",
             "| System | c | Workload | TTFT | TPOT | ITL | E2E |", "|---|---|---|---:|---:|---:|---:|"]
    for row in result["cells"]:
        values = [" / ".join(f"{row['latency'][k][p]:.2f}" for p in ("p50","p95","p99"))
                  for k in ("ttft_ms","tpot_ms","itl_ms","e2e_ms")]
        lines.append(f"| {row['system']} | {row['concurrency']} | {row['workload']} | "+" | ".join(values)+" |")
    (args.output/"latency-table.md").write_text("\n".join(lines)+"\n")
    lines = ["# Exact-Key Cache", "", "Hit rate denominator is all verification batches, not just eligible batches.", "",
             "| System | c | Workload | Eligible | Hit | Unique keys/trial mean | Captures | Replays | Evictions | Capture ms |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["cells"]:
        if "graph" not in row:
            continue
        g = row["graph"]
        lines.append(f"| {row['system']} | {row['concurrency']} | {row['workload']} | {g['eligible_rate']:.1%} | "
                     f"{g['hit_rate']:.1%} | {g['unique_keys_per_trial']['mean']:.1f} | {g['captures']} | "
                     f"{g['hits']} | {g['evictions']} | {g['capture_ms']:.1f} |")
    (args.output/"cache-table.md").write_text("\n".join(lines)+"\n")
    for row in result["rollups"]:
        print(row["system"],row["concurrency"],round(row["output_tokens_per_s"],3),row.get("graph",{}).get("hit_rate"))


if __name__=="__main__":
    main()
