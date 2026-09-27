"""Offline interval/correlation analysis for Phase 4.3 Kineto traces."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        if min(a[1], b[1]) > max(a[0], b[0]):
            result.append([max(a[0], b[0]), min(a[1], b[1])])
        if a[1] < b[1]:
            i += 1
        else:
            j += 1
    return duration(result)


def contains(outer, inner):
    return (outer.get("tid") == inner.get("tid") and outer["ts"] <= inner["ts"]
            and inner["ts"] + inner.get("dur", 0) <= outer["ts"] + outer["dur"] + .01)


def gpu_category(kernel, scope):
    if "nccl" in kernel["name"].lower():
        return "communication"
    if kernel.get("cat") != "kernel":
        return "device_copy_or_set"
    if scope:
        return scope["name"].split("::")[1]
    name = kernel["name"].lower()
    for text, label in (("gemm", "gemm_unattributed"), ("gemv", "gemm_unattributed"),
                        ("flash", "attention_unattributed"), ("store_kvcache", "kv_store")):
        if text in name:
            return label
    return "other_cuda"


def analyze_trace(trace):
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and "dur" in e]
    targets = sorted([e for e in events if e.get("cat") == "user_annotation"
                      and e["name"].startswith("diag.target.")], key=lambda e:e["ts"])
    scopes = [e for e in events if e.get("cat") == "user_annotation"
              and e["name"].startswith("diag.module::")]
    cpu = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation")]
    ext = {e["args"]["External id"]: e for e in cpu if "External id" in e.get("args", {})}
    runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    correlations = {e["args"]["correlation"]: e for e in runtime
                    if "correlation" in e.get("args", {})}
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    attached = defaultdict(list)
    for event in device:
        args = event.get("args", {})
        launch = correlations.get(args.get("correlation"))
        origin = ext.get(args.get("External id")) or launch
        if origin is None:
            continue
        owner = next((i for i, t in enumerate(targets) if contains(t, origin)), None)
        if owner is None:
            continue
        scope = min((s for s in scopes if contains(s, origin)), key=lambda s:s["dur"], default=None)
        attached[owner].append((event, gpu_category(event, scope)))
    rows = []
    kernel_rows = []
    for index, target in enumerate(targets):
        associated = attached[index]
        if not associated:
            raise ValueError("No correlated target GPU events: invalid profiler capture")
        intervals = [[e["ts"], e["ts"] + e["dur"]] for e, _ in associated]
        envelope = [min(x[0] for x in intervals), max(x[1] for x in intervals)]
        groups = defaultdict(list)
        counts = Counter()
        kernels = defaultdict(lambda: dict(count=0, us=0.0))
        for event, label in associated:
            groups[label].append([event["ts"], event["ts"] + event["dur"]])
            counts[label] += 1
            kernels[(label, event["name"])]["count"] += 1
            kernels[(label, event["name"])]["us"] += event["dur"]
            kernel_rows.append(dict(target=index, category=label, name=event["name"],
                                    ts_us=event["ts"], duration_us=event["dur"],
                                    stream=event.get("args", {}).get("stream")))
        comm = groups["communication"]
        compute = [iv for label, values in groups.items() if label != "communication" for iv in values]
        host_sync = [e for e in runtime if contains(target, e) and "Synchronize" in e["name"]]
        launches = [e for e in runtime if contains(target, e) and "LaunchKernel" in e["name"]]
        envelope_us = envelope[1] - envelope[0]
        categories = {label: dict(count=counts[label], sum_us=sum(b-a for a,b in ivs),
                                  union_us=duration(ivs)) for label, ivs in groups.items()}
        rows.append(dict(
            target=index, host_span_us=target["dur"], gpu_envelope_us=envelope_us,
            gpu_active_union_us=duration(intervals), gpu_idle_gap_us=envelope_us-duration(intervals),
            communication_union_us=duration(comm), compute_union_us=duration(compute),
            overlap_us=intersection(comm, compute), kernel_count=sum(e.get("cat")=="kernel" for e,_ in associated),
            launch_count=len(launches), launch_cpu_sum_us=sum(e["dur"] for e in launches),
            host_sync_sum_us=sum(e["dur"] for e in host_sync),
            host_sync_by_api={name:sum(e["dur"] for e in host_sync if e["name"]==name)
                              for name in sorted({e["name"] for e in host_sync})},
            categories=categories,
            kernels=[dict(category=label, name=name, **values)
                     for (label, name), values in sorted(kernels.items(), key=lambda item:-item[1]["us"])],
        ))
    return dict(targets=rows, kernels=kernel_rows,
                trace_gpu_events=len(device), target_gpu_events=sum(len(v) for v in attached.values()),
                profiler_base_time_ns=trace.get("baseTimeNanoseconds"))


def rank_comparison(analyses):
    """Compare matching collective intervals using Kineto's shared host epoch."""
    offset = (analyses[1]["profiler_base_time_ns"] - analyses[0]["profiler_base_time_ns"]) / 1000
    rows = []
    for target in range(len(analyses[0]["targets"])):
        ranks = [sorted([e for e in a["kernels"] if e["target"]==target
                         and e["category"]=="communication"], key=lambda e:e["ts_us"])
                 for a in analyses]
        if len(ranks[0]) != len(ranks[1]):
            raise ValueError("Rank collective kernel count differs")
        joint = 0.0
        start_skews, end_skews = [], []
        for left, right in zip(*ranks):
            if left["name"] != right["name"]:
                raise ValueError("Rank collective kernel ordering differs")
            a, b = left["ts_us"], right["ts_us"] + offset
            ae, be = a+left["duration_us"], b+right["duration_us"]
            joint += max(0, min(ae,be)-max(a,b))
            start_skews.append(abs(a-b))
            end_skews.append(abs(ae-be))
        rows.append(dict(target=target, matching_collective_kernel_count=len(ranks[0]),
                         joint_nccl_residency_us=joint,
                         collective_start_skew_mean_us=statistics.mean(start_skews),
                         collective_start_skew_max_us=max(start_skews),
                         collective_end_skew_mean_us=statistics.mean(end_skews),
                         note="Joint residency is not pure network transfer time; clocks are profiler calibrated."))
    return rows


def rollup(cells):
    groups = []
    for system in ("ordinary", "speculative"):
        for concurrency in (1, 2, 4):
            subset = [c for c in cells if c["system"]==system and c["concurrency"]==concurrency]
            if not subset:
                continue
            rich = [t for c in subset for t in c["ranks"][0]["targets"]]
            minimal = [t for c in subset for t in c["ranks"][0]["minimal_targets"]]
            mean = statistics.mean
            cats = {cat:mean(t["categories"].get(cat, {}).get("sum_us", 0) for t in rich)/1000
                    for cat in sorted({cat for t in rich for cat in t["categories"]})}
            row = dict(system=system, concurrency=concurrency, samples=len(rich),
                       rows=mean(t["rows"] for t in rich),
                       event_only_target_ms=mean(t["unprofiled_event_ms"] for t in rich),
                       host_target_ms=mean(t["unprofiled_host_ms"] for t in rich),
                       host_ms_per_row=mean(t["unprofiled_host_ms"]/t["rows"] for t in rich),
                       rich_target_ms=mean(t["event_ms"] for t in rich),
                       minimal_host_ms=mean(t["host_span_us"] for t in minimal)/1000,
                       kernel_count=mean(t["kernel_count"] for t in rich),
                       rich_category_ms=cats,
                       rich_category_fraction={k:v/(mean(t["gpu_envelope_us"] for t in rich)/1000)
                                               for k,v in cats.items()},
                       minimal_rank_metrics=[],
                       joint_nccl_ms=mean(t["joint_nccl_residency_us"] for c in subset
                                         for t in c["minimal_rank_comparison"])/1000)
            for rank in (0,1):
                ts = [t for c in subset for t in c["ranks"][rank]["minimal_targets"]]
                gpu_span = mean(t["gpu_envelope_us"] for t in ts)
                row["minimal_rank_metrics"].append(dict(
                    rank=rank, envelope_ms=gpu_span/1000,
                    communication_ms=mean(t["communication_union_us"] for t in ts)/1000,
                    communication_fraction=mean(t["communication_union_us"] for t in ts)/gpu_span,
                    compute_ms=mean(t["compute_union_us"] for t in ts)/1000,
                    idle_ms=mean(t["gpu_idle_gap_us"] for t in ts)/1000,
                    overlap_ms=mean(t["overlap_us"] for t in ts)/1000,
                    joint_residency_fraction=row["joint_nccl_ms"]*1000/gpu_span,
                ))
            groups.append(row)
    return groups


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--spec-raw", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(diagnostic_only=True, raw=str(args.raw), cells=[])
    table = []
    for system in ("ordinary", "speculative"):
        raw = args.spec_raw if system == "speculative" and args.spec_raw else args.raw
        manifest = json.loads((raw / f"{system}-manifest.json").read_text())
        if not manifest["normal_exit"] or manifest["source_regression"]:
            raise ValueError("Failed diagnostic or changed frozen source")
        cells = sorted({(r["concurrency"], r["workload"]) for r in manifest["trials"]})
        for concurrency, workload in cells:
            cell = dict(system=system, concurrency=concurrency, workload=workload, raw=str(raw), ranks=[])
            prefix = f"{system}-c{concurrency}-{workload}"
            rich_analyses, minimal_analyses = [], []
            for rank in (0, 1):
                meta = json.loads((raw / f"{prefix}-profile-rank{rank}.json").read_text())
                timing = json.loads((raw / f"{prefix}-timing-rank{rank}.json").read_text())
                trace_path = raw / f"{prefix}-profile-rank{rank}.trace.json"
                analysis = analyze_trace(json.loads(trace_path.read_text()))
                rich_analyses.append(analysis)
                # Keep full event-level classification outside tracked summaries.
                (raw / f"{prefix}-rank{rank}-classified.json").write_text(json.dumps(analysis))
                minimal_path = raw / f"{prefix}-minimal-rank{rank}.trace.json"
                minimal = analyze_trace(json.loads(minimal_path.read_text()))
                minimal_analyses.append(minimal)
                (raw / f"{prefix}-minimal-rank{rank}-classified.json").write_text(json.dumps(minimal))
                event_control = json.loads((raw / f"{prefix}-collective-events-rank{rank}.json").read_text())
                by_target = []
                for i, target in enumerate(analysis["targets"]):
                    target.update(rows=meta["targets"][i]["rows"],
                                  event_ms=meta["targets"][i]["gpu_event_ms"],
                                  unprofiled_event_ms=timing["targets"][i]["gpu_event_ms"],
                                  unprofiled_host_ms=(timing["targets"][i]["end_ns"]-timing["targets"][i]["start_ns"])/1e6)
                    collectives = [r for r in meta["collectives"] if r["target"]==i]
                    target["collective_counts"] = dict(Counter(r["type"] for r in collectives))
                    target["collective_cpu_us"] = sum(r["end_ns"]-r["start_ns"] for r in collectives)/1e3
                    by_target.append(target)
                    table.append(dict(system=system, concurrency=concurrency, workload=workload,
                                      rank=rank, target=i, rows=target["rows"],
                                      unprofiled_host_ms=target["unprofiled_host_ms"],
                                      unprofiled_event_ms=target["unprofiled_event_ms"],
                                      profiled_event_ms=target["event_ms"],
                                      **{k:target[k] for k in ("gpu_envelope_us", "gpu_active_union_us",
                                          "gpu_idle_gap_us", "communication_union_us", "compute_union_us",
                                          "overlap_us", "kernel_count", "launch_count", "host_sync_sum_us")}))
                cell["ranks"].append(dict(rank=rank, targets=by_target,
                                          shapes=meta["shapes"], collectives=meta["collectives"],
                                          minimal_targets=minimal["targets"],
                                          collective_event_control=event_control,
                                          profiled_rpc=meta["rpc"], timing_rpc=timing["rpc"],
                                          profiled_targets=meta["targets"], timing_targets=timing["targets"]))
            cell["rich_rank_comparison"] = rank_comparison(rich_analyses)
            cell["minimal_rank_comparison"] = rank_comparison(minimal_analyses)
            trials = [r for r in manifest["trials"] if (r["concurrency"],r["workload"])==(concurrency,workload)]
            cell["outputs_equal_all_modes"] = all(r["outputs"]==trials[0]["outputs"] for r in trials)
            cell["cleanup_zero"] = all(not any(r["cleanup"].values()) and not any(r["rank_target_states"]) for r in trials)
            cell["step_wall_ms_by_mode"] = {r["mode"]:[s["host_step_ns"]/1e6 for s in r["samples"]] for r in trials}
            result["cells"].append(cell)
    result["rollups"] = rollup(result["cells"])
    result["validation"] = dict(
        all_output_replays_equal=all(c["outputs_equal_all_modes"] for c in result["cells"]),
        all_cleanup_zero=all(c["cleanup_zero"] for c in result["cells"]),
        no_frozen_source_regression=True,
    )
    ordinary_manifest = json.loads((args.raw / "ordinary-manifest.json").read_text())
    spec_manifest = json.loads(((args.spec_raw or args.raw) / "speculative-manifest.json").read_text())
    pairs = []
    for trial in ordinary_manifest["trials"]:
        if trial["mode"] != "control":
            continue
        peer = next(t for t in spec_manifest["trials"] if t["mode"]=="control"
                    and t["concurrency"]==trial["concurrency"] and t["workload"]==trial["workload"])
        pairs.extend(left == right for left, right in zip(trial["outputs"],peer["outputs"]))
    result["validation"]["cross_system_output_pairs"] = len(pairs)
    result["validation"]["cross_system_matching_pairs"] = sum(pairs)
    result["environment"] = {key:spec_manifest[key] for key in (
        "versions", "cuda", "nccl", "config", "target_revision", "draft_revision", "git", "nsight", "topology")}
    result["decision"] = dict(
        primary_next_module="target eager host dispatch / GPU launch and rank arrival coordination",
        largest_compute_category="MLP gate/up/down GEMM",
        attention_primary=False, serial_draft_primary=False,
        pure_transport_fraction_identified=False,
        phase44_implemented=False,
        recommendation="Approval-gated, fixed-shape target-only launch/replay experiment; do not start draft batching or rewrite NCCL from residency percentages.",
    )
    detailed_path = args.raw / "detailed-summary.json"
    detailed_path.write_text(json.dumps(result, indent=2))
    result["detailed_summary"] = str(detailed_path)
    for cell in result["cells"]:
        for rank in cell["ranks"]:
            shapes = Counter(json.dumps({k:v for k,v in s.items() if k not in ("name","target")}, sort_keys=True)
                             for s in rank["shapes"])
            rank["shapes"] = [dict(**json.loads(shape), observed_calls=count) for shape,count in shapes.items()]
            inventory = Counter(json.dumps({k:v for k,v in r.items() if k in ("type","shape","bytes","dtype","op")}, sort_keys=True)
                                for r in rank.pop("collectives"))
            rank["collective_inventory"] = [dict(**json.loads(key), observed_calls=count) for key,count in inventory.items()]
            event = rank["collective_event_control"]
            event["by_target"] = [dict(target=i,
                collective_stream_ms=sum(e["stream_elapsed_ms"] for e in event["collectives"] if e["target"]==i),
                collective_counts=dict(Counter(e["type"] for e in event["collectives"] if e["target"]==i)))
                for i in range(len(event["targets"]))]
            del event["collectives"]
            examples = defaultdict(set)
            for target in rank["targets"]:
                for kernel in target.pop("kernels"):
                    examples[kernel["category"]].add(kernel["name"])
            rank["kernel_names_by_category"] = {k:sorted(v) for k,v in examples.items()}
            for target in rank["minimal_targets"]:
                del target["kernels"]
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    supplemental = [Path("/root/autodl-tmp/eagle3-phase4.3-pilot"),
                    Path("/root/autodl-tmp/eagle3-phase4.3-raw-20260923"),
                    Path("/root/autodl-tmp/eagle3-phase4.3-verbose-analysis")]
    roots = sorted(set([args.raw, args.spec_raw or args.raw, *supplemental]))
    files = []
    for root in roots:
        for path in sorted(root.glob("*")):
            if path.is_file():
                files.append(dict(path=str(path), bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    tests = {}
    for phase in ("pre", "post"):
        for suite in ("focused", "cpu"):
            path = Path(f"/root/autodl-tmp/eagle43-{phase}-{suite}.log")
            content = path.read_text()
            tests[f"{phase}_{suite}"] = dict(path=str(path),
                count=int(re.search(r"Ran (\d+) tests", content)[1]),
                passed=content.strip().endswith("OK"),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    artifacts = dict(diagnostic_only=True, tests=tests, files=files,
                     primary_manifests=dict(ordinary=str(args.raw / "ordinary-manifest.json"),
                         speculative=str((args.spec_raw or args.raw) / "speculative-manifest.json")),
                     no_commit=True, no_production_edits=True,
                     supplements="Pilot, failed old single-prefill harness, and verbose NCCL TUNING-log run retained; see report.")
    (args.output / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
    with (args.output / "target-table.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for cell in result["cells"]:
        rows = cell["ranks"][0]["targets"]
        print(cell["system"], cell["concurrency"], cell["workload"],
              "host/event/profile_ms", *[round(statistics.mean(r[key] for r in rows),3)
                    for key in ("unprofiled_host_ms","unprofiled_event_ms","event_ms")],
              "comm/idle/overlap_us", *[round(statistics.mean(r[key] for r in rows),1)
                    for key in ("communication_union_us","gpu_idle_gap_us","overlap_us")])


if __name__ == "__main__":
    main()
