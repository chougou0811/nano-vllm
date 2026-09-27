"""Offline summary of unfiltered Phase5.1 measurements and profiler traces."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from benchmarks.serving.eagle3_phase42 import distribution


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def union_length(intervals):
    end = None
    total = 0.0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop-start
        elif stop > end:
            total += stop-end
        end = stop if end is None else max(end, stop)
    return total


def trace_summary(path):
    events = json.loads(path.read_text())["traceEvents"]
    scopes = [e for e in events if e.get("name") == "phase51.verify" and e.get("ph") == "X"
              and e.get("cat") == "user_annotation"]
    windows = []
    # The frozen prototype tracer names both verify and its following rollback
    # "phase51.verify". The driver emits exactly two verify/rollback pairs.
    # Retain all spans and label them; do not average rollback into verification.
    for index, scope in enumerate(scopes):
        start, end = scope["ts"], scope["ts"]+scope["dur"]
        selected = [e for e in events if e.get("ph") == "X" and start <= e.get("ts", -1) < end]
        gpu = [e for e in selected if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
        kernels = [e for e in gpu if e.get("cat") == "kernel"]
        collectives = [e for e in kernels if "nccl" in e.get("name", "").lower()]
        apis = [e for e in selected if e.get("cat") in ("cuda_runtime", "cuda_driver")]
        counts = Counter(e["name"] for e in apis)
        lo = min((e["ts"] for e in gpu), default=start)
        hi = max((e["ts"]+e["dur"] for e in gpu), default=start)
        busy = union_length((e["ts"], e["ts"]+e["dur"]) for e in gpu)
        copies = [e for e in gpu if e.get("cat") == "gpu_memcpy"]
        operation = "verify" if index % 2 == 0 else "rollback"
        if len(scopes) != 4:
            raise RuntimeError(f"Unexpected trace scope count: {path}: {len(scopes)}")
        windows.append(dict(operation=operation, cpu_scope_us=scope["dur"], gpu_envelope_us=hi-lo,
            gpu_busy_union_us=busy, gpu_gap_us=max(0, hi-lo-busy),
            kernel_count=len(kernels), nccl_count=len(collectives),
            nccl_residency_us=sum(e["dur"] for e in collectives),
            non_nccl_kernel_us=sum(e["dur"] for e in kernels if e not in collectives),
            copy_count=len(copies), copy_us=sum(e["dur"] for e in copies),
            launch_calls=sum(n for k,n in counts.items() if "LaunchKernel" in k or "GraphLaunch" in k),
            graph_launch_calls=sum(n for k,n in counts.items() if "GraphLaunch" in k),
            graph_launch_cpu_us=sum(e["dur"] for e in apis if "GraphLaunch" in e["name"]),
            api_counts=dict(counts),
            nccl_start_us=[e["ts"] for e in sorted(collectives, key=lambda e:e["ts"])]))
    return dict(file=str(path), windows=windows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = {p.parent.name:json.loads(p.read_text()) for p in args.root.glob("*/manifest.json")}
    fixed = []
    for name, run in manifests.items():
        for cell in run.get("fixed", []):
            modes = {}
            for mode in sorted({s["mode"] for s in cell["samples"]}):
                rows = [s for s in cell["samples"] if s["mode"] == mode]
                modes[mode] = dict(host_ms=distribution(s["host_ns"]/1e6 for s in rows),
                    repeat_median_ms=[median(s["host_ns"]/1e6 for s in rows if s["repeat"] == repeat)
                                      for repeat in range(5)])
            fixed.append(dict(run=name, batch=cell["batch"], context=cell["context"],
                              layers=cell["layers"], label=cell.get("label"), modes=modes))
    traces = {}
    for path in args.root.glob("*/*-rank[01].json"):
        traces[str(path)] = trace_summary(path)
    for path, row in traces.items():
        if not path.endswith("rank0.json"):
            continue
        other = traces.get(path.replace("rank0.json", "rank1.json"))
        if other is None:
            continue
        row["rank_skew"] = []
        for a,b in zip(row["windows"], other["windows"]):
            aa,bb = a["nccl_start_us"], b["nccl_start_us"]
            row["rank_skew"].append(dict(matching_counts=len(aa)==len(bb),
                absolute_start_skew_us=distribution(abs(x-y) for x,y in zip(aa,bb))))
    serving = []
    for c in (1,2,4):
        a = manifests.get(f"serving-c{c}-literal")
        b = manifests.get(f"serving-c{c}-optimized")
        if not a or not b or not a.get("normal_exit") or not b.get("normal_exit"):
            continue
        by_key = {(t["workload"],t["repeat"]):t for t in a["trials"]}
        for right in b["trials"]:
            left = by_key[right["workload"],right["repeat"]]
            ls,rs = left["summary"],right["summary"]
            serving.append(dict(concurrency=c, workload=right["workload"], repeat=right["repeat"],
                speedup=ls["duration_s"]/rs["duration_s"], eager=ls, optimized=rs,
                outputs_equal=[r["output_token_ids"] for r in left["requests"]] ==
                              [r["output_token_ids"] for r in right["requests"]],
                speculation_equal=ls["speculation"]==rs["speculation"],
                eager_has_no_graphs=all(r["entries"]==0 for r in left["rank_memory"]["ranks"])))
    aggregate = []
    for c in (1,2,4):
        cells = [r for r in serving if r["concurrency"] == c]
        if not cells:
            continue
        repeats = []
        for repeat in range(5):
            group = [r for r in cells if r["repeat"] == repeat]
            repeats.append(sum(r["eager"]["duration_s"] for r in group)/
                           sum(r["optimized"]["duration_s"] for r in group))
        start = manifests[f"serving-c{c}-optimized"]["startup"]
        startup_s = max(r["duration_ns"] for r in start)/1e9
        request_saving = sum(r["eager"]["duration_s"]-r["optimized"]["duration_s"] for r in cells)/sum(r["eager"]["requests"] for r in cells)
        aggregate.append(dict(concurrency=c, pairs=len(cells),
            geometric_mean_speedup=math.exp(statistics.mean(math.log(r["speedup"]) for r in cells)),
            repeat_aggregate_speedups=repeats,
            heldout_median_speedup=median(r["speedup"] for r in cells if r["workload"]=="heldout-mixed"),
            startup_s=startup_s, projected_payback_requests=math.ceil(startup_s/request_saving) if request_saving>0 else None))
    output = dict(fixed=fixed, traces=traces, serving_pairs=serving, serving_aggregate=aggregate,
        runs={name:dict(normal_exit=d.get("normal_exit"),failure=d.get("failure"),
            protected_changed=d.get("protected_changed"),command=d["command"],
            source_hashes=d["source_hashes"],correctness_passed=d.get("correctness_passed"))
              for name,d in manifests.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(dict(fixed_cells=len(fixed),traces=len(traces),serving_pairs=len(serving),
                         serving_aggregate=aggregate),indent=2))


if __name__ == "__main__":
    main()
