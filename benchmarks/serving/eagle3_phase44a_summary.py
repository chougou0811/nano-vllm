"""Offline, non-filtering analysis of fixed-shape graph probe artifacts."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

from benchmarks.serving.eagle3_phase43_summary import contains, duration, intersection, rank_comparison


def stats(values):
    values = sorted(values)
    if not values:
        raise ValueError("Empty measurement group")

    def percentile(q):
        index = (len(values)-1)*q
        lower = int(index)
        return values[lower]+(values[min(lower+1,len(values)-1)]-values[lower])*(index-lower)

    return dict(n=len(values),mean=statistics.mean(values),min=values[0],max=values[-1],
                p50=percentile(.5),p95=percentile(.95),p99=percentile(.99))


def analyze(trace):
    events = [e for e in trace["traceEvents"] if e.get("ph")=="X" and "dur" in e]
    targets = sorted([e for e in events if e.get("cat")=="user_annotation"
                      and e["name"].startswith("diag.target.")],key=lambda e:e["ts"])
    runtimes = [e for e in events if e.get("cat") in ("cuda_runtime","cuda_driver")]
    correlations = {e["args"]["correlation"]:e for e in runtimes if "correlation" in e.get("args",{})}
    cpu = [e for e in events if e.get("cat") in ("cpu_op","user_annotation")]
    external = {e["args"]["External id"]:e for e in cpu if "External id" in e.get("args",{})}
    device = [e for e in events if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
    attached = defaultdict(list)
    methods = Counter()
    for event in device:
        args = event.get("args",{})
        origin = external.get(args.get("External id")) or correlations.get(args.get("correlation"))
        owner = next((i for i,t in enumerate(targets) if origin and contains(t,origin)),None)
        method = "external_or_launch_correlation"
        if owner is None:
            # Frozen finite/status CPU reads drain verification before return.
            # This fallback is explicitly counted, never silently imputed.
            owners = [i for i,t in enumerate(targets) if t["ts"]<=event["ts"]
                      and event["ts"]+event["dur"]<=t["ts"]+t["dur"]+.01]
            if len(owners)==1:
                owner = owners[0]
                method = "synchronized_endpoint_time_window"
        if owner is not None:
            methods[method] += 1
            attached[owner].append(event)
    rows,kernels = [],[]
    for index,target in enumerate(targets):
        associated = attached[index]
        if not associated:
            raise ValueError("No target device events")
        intervals = [[e["ts"],e["ts"]+e["dur"]] for e in associated]
        envelope = max(b for _,b in intervals)-min(a for a,_ in intervals)
        apis = [e for e in runtimes if contains(target,e)]
        launches = [e for e in apis if "launchkernel" in e["name"].lower()]
        graph_launches = [e for e in apis if "graphlaunch" in e["name"].lower()]
        nccl = [e for e in associated if "nccl" in e["name"].lower()]
        rows.append(dict(target=index,timing_index=int(target["name"].split(".")[-1]),
            host_span_us=target["dur"],cpu_kernel_launch_count=len(launches),
            cpu_kernel_launch_us=sum(e["dur"] for e in launches),
            cpu_graph_launch_count=len(graph_launches),
            cpu_graph_launch_us=sum(e["dur"] for e in graph_launches),
            graph_launch_device_active_overlap_us=intersection(intervals,
                [[e["ts"],e["ts"]+e["dur"]] for e in graph_launches]),
            cpu_launch_total_us=sum(e["dur"] for e in launches+graph_launches),
            gpu_kernel_count=sum(e.get("cat")=="kernel" for e in associated),
            gpu_envelope_us=envelope,gpu_active_union_us=duration(intervals),
            gpu_idle_gap_us=envelope-duration(intervals),collective_kernel_count=len(nccl),
            launch_api_names=dict(Counter(e["name"] for e in launches+graph_launches))))
        for event in associated:
            kernels.append(dict(target=index,name=event["name"],ts_us=event["ts"],duration_us=event["dur"],
                category="communication" if "nccl" in event["name"].lower() else "compute_or_copy"))
    return dict(targets=rows,kernels=kernels,attachment_methods=dict(methods),
                total_device_events=len(device),target_device_events=sum(map(len,attached.values())),
                profiler_base_time_ns=trace["baseTimeNanoseconds"])


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--correctness",type=Path,required=True)
    parser.add_argument("--raw",type=Path,nargs="+",required=True)
    parser.add_argument("--reuse",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    gate = load(args.correctness/"manifest.json")
    assert gate["correctness_passed"] and gate["normal_exit"] and not gate["source_regression"]
    audit_ranks = [load(args.correctness/f"rank{rank}.json") for rank in (0,1)]
    validation = dict(cases=len(gate["correctness"]),
        paired_requests=sum(len(c["contexts"]) for c in gate["correctness"]),
        all_generation_traces_equal=all(all(c["checks"].values()) for c in gate["correctness"]),
        rank_audit_records=[len(r["audits"]) for r in audit_ranks],
        exact_replay_checks=[sum(len(a["checks"]) for a in r["audits"]) for r in audit_ranks],
        all_byte_checks=all(all(v for k,v in check.items() if k!="replay")
            for r in audit_ranks for a in r["audits"] for check in a["checks"]),
        symmetric_fallback=gate["symmetric_fallback"]["passed"],
        exception_cleanup=gate["exception_cleanup"],released_cleanup=gate["released_cleanup"],
        normal_exit=gate["normal_exit"],source_regression=gate["source_regression"])
    released = [[r["allocated"] for r in c["graph"]["released"]["ranks"]] for c in gate["correctness"]]
    validation["post_release_allocated_bytes"] = released
    validation["post_release_allocated_constant"] = all(row==released[0] for row in released)
    validation["rank_errors"] = [r["error"] for r in audit_ranks]
    result = dict(prototype_only=True,correctness=validation,processes=[],cells=[],
                  timing_scope="Full verification RPC, including graph consensus, copies, finite/status validation; rollback excluded",
                  profiler_scope="Mechanism only; never the speedup denominator",outliers_filtered=0)
    flat = []
    for raw in args.raw:
        manifest = load(raw/"manifest.json")
        assert manifest["normal_exit"] and not manifest["source_regression"]
        assert manifest["prototype_hashes"]==gate["prototype_hashes"]
        ranks = [load(raw/f"rank{r}.json") for r in (0,1)]
        result["processes"].append(dict(raw=str(raw),environment={k:manifest[k] for k in
            ("versions","cuda","nccl","config","target_revision","draft_revision","git","topology")},
            final_cleanup=manifest["final_cleanup"],normal_exit=True))
        for cell in manifest["performance"]:
            row = dict(raw=str(raw),rows=cell["rows"],context=cell["contexts"][0],modes={},
                cleanup_before_release=cell["cleanup_before_release"],
                cleanup_after_release=cell["cleanup_after_release"])
            samples = []
            for sample in cell["samples"]:
                index = sample["timing_index"]
                pair = [r["timings"][index] for r in ranks]
                assert all(p["index"]==index and p["mode"]==sample["mode"] for p in pair)
                value = dict(raw=str(raw),rows=row["rows"],context=row["context"],**sample,
                    host_rpc_ms=sample["host_rpc_ns"]/1e6,
                    rank0_host_ms=(pair[0]["end_ns"]-pair[0]["start_ns"])/1e6,
                    rank1_host_ms=(pair[1]["end_ns"]-pair[1]["start_ns"])/1e6,
                    rank0_event_ms=pair[0]["gpu_event_ms"],rank1_event_ms=pair[1]["gpu_event_ms"],
                    rank0_arrival_ns=pair[0]["start_ns"],rank1_arrival_ns=pair[1]["start_ns"],
                    entry_skew_us=abs(pair[0]["start_ns"]-pair[1]["start_ns"])/1e3,
                    ms_per_row=sample["host_rpc_ns"]/1e6/row["rows"])
                samples.append(value)
                flat.append(value)
            for mode in ("eager","coordinated-eager","graph"):
                selected = [s for s in samples if s["mode"]==mode]
                row["modes"][mode] = {metric:stats([s[metric] for s in selected]) for metric in
                    ("host_rpc_ms","rank0_host_ms","rank1_host_ms","rank0_event_ms",
                     "rank1_event_ms","entry_skew_us","ms_per_row")}
                row["modes"][mode]["repeat_host_mean_ms"] = [statistics.mean(s["host_rpc_ms"]
                    for s in selected if s["repeat"]==repeat) for repeat in sorted({s["repeat"] for s in selected})]
            eager = row["modes"]["eager"]["host_rpc_ms"]["mean"]
            graph = row["modes"]["graph"]["host_rpc_ms"]["mean"]
            row["target_speedup"] = eager/graph
            row["ms_saved"] = eager-graph
            row["reduction_pct"] = 100*(1-graph/eager)
            row["repeat_speedups"] = [e/g for e,g in zip(row["modes"]["eager"]["repeat_host_mean_ms"],
                                                       row["modes"]["graph"]["repeat_host_mean_ms"])]
            row["traces"] = {}
            for mode in ("eager","graph"):
                prefix = f"M{row['rows']}-ctx{row['context']}-{mode}-trace"
                analyses = [analyze(load(raw/f"{prefix}-rank{rank}.json")) for rank in (0,1)]
                comparison = rank_comparison(analyses)
                for rank,analysis in enumerate(analyses):
                    (raw/f"{prefix}-rank{rank}-classified.json").write_text(json.dumps(analysis))
                    del analysis["kernels"]
                row["traces"][mode] = dict(ranks=analyses,matched_collectives=comparison)
            result["cells"].append(row)
    result["combined"] = []
    for rows in (4,8,16):
        for context in (256,768):
            cells = [c for c in result["cells"] if c["rows"]==rows and c["context"]==context]
            combined = dict(rows=rows,context=context,modes={},mechanism={})
            for mode in ("eager","coordinated-eager","graph"):
                samples = [s for s in flat if s["rows"]==rows and s["context"]==context and s["mode"]==mode]
                combined["modes"][mode] = {metric:stats([s[metric] for s in samples]) for metric in
                    ("host_rpc_ms","rank0_event_ms","rank1_event_ms","entry_skew_us","ms_per_row")}
            for mode in ("eager","graph"):
                traced = [t for c in cells for t in c["traces"][mode]["ranks"][0]["targets"]]
                mechanism = {metric:stats([t[metric] for t in traced]) for metric in
                    ("cpu_kernel_launch_count","cpu_kernel_launch_us","cpu_graph_launch_count",
                     "cpu_graph_launch_us","cpu_launch_total_us","gpu_kernel_count","gpu_idle_gap_us")}
                mechanism["graph_launch_device_active_overlap_us"] = stats([
                    t["graph_launch_device_active_overlap_us"] for t in traced])
                mechanism["matched_collective_start_skew_us"] = stats([s["collective_start_skew_mean_us"]
                    for c in cells for s in c["traces"][mode]["matched_collectives"]])
                mechanism["rank1_gpu_idle_gap_us"] = stats([t["gpu_idle_gap_us"]
                    for c in cells for t in c["traces"][mode]["ranks"][1]["targets"]])
                combined["mechanism"][mode] = mechanism
            eager = combined["modes"]["eager"]["host_rpc_ms"]["mean"]
            graph = combined["modes"]["graph"]["host_rpc_ms"]["mean"]
            combined.update(target_speedup=eager/graph,ms_saved=eager-graph,reduction_pct=100*(1-graph/eager),
                repeat_speedups=[value for c in cells for value in c["repeat_speedups"]])
            result["combined"].append(combined)
    if args.reuse:
        reuse = load(args.reuse/"manifest.json")
        assert reuse["metadata_reuse_passed"] and reuse["normal_exit"] and not reuse["source_regression"]
        reuse_ranks = [load(args.reuse/f"rank{rank}.json") for rank in (0,1)]
        result["metadata_reuse"] = dict(raw=str(args.reuse),cases=reuse["metadata_reuse"],
            passed=True,normal_exit=True,graph_captures=[len(r["keys"]) for r in reuse_ranks],
            exact_replay_checks=[sum(len(a["checks"]) for a in r["audits"]) for r in reuse_ranks],
            all_byte_checks=all(all(v for k,v in check.items() if k!="replay")
                for r in reuse_ranks for a in r["audits"] for check in a["checks"]),
            rank0_unprofiled_replay_api_ms={str(m):stats([s["ns"]/1e6
                for s in reuse["unprofiled_graph_replay_api"] if s["rows"]==m]) for m in (4,8,16)})
    result["tests"] = {}
    for phase in ("pre","post"):
        for suite in ("focused","cpu"):
            path = Path(f"/root/autodl-tmp/eagle44a-{phase}-{suite}.log")
            if path.exists():
                content = path.read_text()
                match = re.search(r"Ran (\d+) tests",content)
                result["tests"][f"{phase}_{suite}"] = dict(path=str(path),count=int(match[1]) if match else None,
                    passed=content.strip().endswith("OK"),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result["decision"] = dict(
        same_shape_correctness_passed=validation["all_byte_checks"] and validation["all_generation_traces_equal"],
        all_repeat_target_speedups_above_one=all(s>1 for c in result["combined"] for s in c["repeat_speedups"]),
        all_cell_reductions_above_five_pct=all(c["reduction_pct"]>5 for c in result["combined"]),
        internal_collective_skew_reduced=all(c["mechanism"]["graph"]["matched_collective_start_skew_us"]["mean"]
            <c["mechanism"]["eager"]["matched_collective_start_skew_us"]["mean"] for c in result["combined"]),
        rpc_entry_skew_consistently_reduced=all(c["modes"]["graph"]["entry_skew_us"]["mean"]
            <c["modes"]["eager"]["entry_skew_us"]["mean"] for c in result["combined"]),
        serving_speedup_measured=False,phase44b_implemented=False,
        recommendation="Bounded Phase 4.4B is supported by target-only evidence, subject to user approval and a safe key-coverage/hit-rate design. RPC entry skew did not consistently improve.")
    (args.output/"summary.json").write_text(json.dumps(result,indent=2))
    with (args.output/"target-timings.csv").open("w") as file:
        writer = csv.DictWriter(file,fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    artifacts = []
    for root in [args.correctness,*args.raw,*([args.reuse] if args.reuse else [])]:
        for path in sorted(root.glob("*")):
            if path.is_file():
                artifacts.append(dict(path=str(path),bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output/"artifacts.json").write_text(json.dumps(artifacts,indent=2))
    for cell in result["cells"]:
        print(cell["rows"],cell["context"],round(cell["target_speedup"],3),
              round(cell["ms_saved"],3),cell["repeat_speedups"])


if __name__ == "__main__":
    main()
