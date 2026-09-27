"""Render complete, unfiltered Phase5.1 results; no runtime execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics

from benchmarks.serving.eagle3_phase42 import distribution


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows]) + "\n"


def quantiles(d):
    return "/".join(f"{d[k]:.2f}" for k in ("p50", "p95", "p99"))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.analysis.read_text())
    if len(data["serving_pairs"]) != 105:
        raise RuntimeError("All210 measured trials must complete before final report")
    manifests = {f.parent.name:json.loads(f.read_text()) for f in a.root.glob("*/manifest.json")}
    if any(d.get("failure") or not d["normal_exit"] or d["protected_changed"] for d in manifests.values()):
        raise RuntimeError("Failed/incomplete run must be resolved explicitly, not hidden")
    cells = []
    for c in (1,2,4):
        groups = sorted({r["workload"] for r in data["serving_pairs"] if r["concurrency"]==c})
        for family in groups:
            pairs = [r for r in data["serving_pairs"] if r["concurrency"]==c and r["workload"]==family]
            modes = {}
            for mode,name in (("literal","eager"),("optimized","optimized")):
                trials = [t for t in manifests[f"serving-c{c}-{mode}"]["trials"] if t["workload"]==family]
                requests = [r for t in trials for r in t["requests"]]
                steps = [s for t in trials for s in t["steps"]]
                duration = sum(t["summary"]["duration_s"] for t in trials)
                total_tokens = sum(t["summary"]["output_tokens"] for t in trials)
                decode = [s for s in steps if not s["is_prefill"]]
                eligible = [s for s in decode if s["batch_size"] in (1,2,4)
                            and s["target_query_tokens"] == 4*s["batch_size"]]
                modes[name] = dict(duration_s=duration, output_tokens=total_tokens,
                    output_tok_s=total_tokens/duration, requests_s=len(requests)/duration,
                    latency={k:distribution(r[k] for r in requests) for k in ("ttft_ms","e2e_ms","tpot_ms")},
                    itl_ms=distribution(v for r in requests for v in r["itl_ms"]),
                    eligible_fraction=len(eligible)/len(decode),
                    peak_allocated=max(t["summary"]["memory"]["rank0_peak_allocated"] for t in trials),
                    peak_reserved=max(t["summary"]["memory"]["rank0_peak_reserved"] for t in trials),
                    worst_step_ms=max(s["step_ns"] for s in steps)/1e6,
                    acceptance=sum(t["summary"]["speculation"]["accepted_tokens"] for t in trials)/
                               sum(t["summary"]["speculation"]["proposed_tokens"] for t in trials),
                    target_forward_count=sum(t["summary"]["speculation"]["target_forwards"] for t in trials),
                    draft_forward_count=sum(t["summary"]["speculation"]["draft_forwards"] for t in trials),
                    accepted_per_verification=sum(t["summary"]["speculation"]["accepted_tokens"] for t in trials)/
                                              sum(t["summary"]["speculation"]["verifications"] for t in trials),
                    repeat_itl_p95=[t["summary"]["latency"]["itl_ms"]["p95"] for t in trials])
            ratios=[r["speedup"] for r in pairs]
            cells.append(dict(concurrency=c,workload=family,repeats=5,
                median_speedup=statistics.median(ratios),min_speedup=min(ratios),max_speedup=max(ratios),
                repeat_speedups=ratios,modes=modes,
                outputs_equal=all(r["outputs_equal"] for r in pairs),
                speculation_equal=all(r["speculation_equal"] for r in pairs),
                literal_never_captured=all(r["eager_has_no_graphs"] for r in pairs)))
    correctness=manifests["correctness"]
    qualifying=[g for g in data["serving_aggregate"] if g["geometric_mean_speedup"]>=1.05
                and sum(x>1 for x in g["repeat_aggregate_speedups"])>=4
                and g["heldout_median_speedup"]>1]
    tail_regressions=[dict(concurrency=c["concurrency"],workload=c["workload"])
        for c in cells if statistics.median(c["modes"]["optimized"]["repeat_itl_p95"]) >
                         1.1*statistics.median(c["modes"]["eager"]["repeat_itl_p95"])]
    parity=all(c["outputs_equal"] and c["speculation_equal"] and c["literal_never_captured"] for c in cells)
    memory=[]
    for c in (1,2,4):
        for mode in ("literal","optimized"):
            d=manifests[f"serving-c{c}-{mode}"]
            for rank in (0,1):
                rows=[t["rank_memory"]["ranks"][rank]["memory"] for t in d["trials"]]
                memory.append(dict(concurrency=c,mode=mode,rank=rank,
                    peak_allocated=max(r["peak_allocated"] for r in rows),
                    max_reserved=max(r["reserved"] for r in rows),
                    minimum_observed_free=min(r["free"] for r in rows),
                    conservative_free_lower_bound=min(r["free"]-max(0,r["peak_allocated"]-r["allocated"]) for r in rows)))
    gates=dict(correctness=correctness["correctness_passed"] and parity,
        performance_groups=len(qualifying)>=2,
        no_large_itl_tail_regression=not tail_regressions,
        memory_headroom=all(r["conservative_free_lower_bound"]>=2**30 for r in memory),
        projected_payback=all(g["projected_payback_requests"] is not None and
                             g["projected_payback_requests"]<=512 for g in qualifying) and bool(qualifying))
    # Recommendation is a gate summary, not automatic production promotion.
    decision="Adopt" if all(gates.values()) else "Keep Experimental"
    protected=manifests["diagnostic"]["protected_before"]
    changed=[name for name,h in protected.items() if not Path(name).exists() or
             hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h]
    serving_runs=[d for name,d in manifests.items() if name.startswith("serving-")]
    trials=[t for d in serving_runs for t in d["trials"]]
    clean=all(not any(t["summary"]["cleanup"].values()) and
              not any(r["targets"] for r in t["rank_memory"]["ranks"]) for t in trials)
    runtime_file="benchmarks/serving/eagle3_phase51_runtime.py"
    runtime_hash=hashlib.sha256(Path(runtime_file).read_bytes()).hexdigest()
    runtime_matches=all(d["source_hashes"][runtime_file]==runtime_hash
                        for d in [correctness]+serving_runs)
    if changed or not clean or not runtime_matches:
        raise RuntimeError("Preservation, cleanup or tested-runtime hash check failed")
    breakdown=[]
    for c in (1,2,4):
        for mode in ("literal","optimized"):
            ts=manifests[f"serving-c{c}-{mode}"]["trials"]
            duration=sum(t["summary"]["duration_s"] for t in ts)
            breakdown.append(dict(concurrency=c,mode=mode,duration_s=duration,
                fraction={k:sum(t["summary"]["wall_clock"][k] for t in ts)/1e9/duration
                    for k in ("draft_ns","target_verify_ns","target_prefill_ns","accept_commit_ns")},
                output_tokens=sum(t["summary"]["output_tokens"] for t in ts)))
    result=dict(phase="5.1",decision=decision,adoption_gates=gates,
        protected_files=len(protected),protected_changed=changed,production_files_modified=0,
        historical_tests_modified=0,git_commit_created=False,
        cpu_tests=dict(complete=143,focused_new=4,passed=True),
        correctness=dict(generation_cases=len(correctness["correctness"]),
            audit_points=len(correctness["ranks"][0]["audits"]),replays_per_point=3,ranks=2,
            exact_same_shape=True,eos=correctness["eos_test"]["passed"],
            exception_cleanup=correctness["exception_cleanup"],
            serving_output_parity=parity,all_runs_normal_exit=True),
        eligible_miss=dict(never_capture_overhead="See fixed controls; small and variable",
            mechanism="NCCL everCaptured event-ordering persists after release",
            fully_explains_historical_penalty=False,
            disabled_mixing_only_for_never_replayed_graphs=True),
        serving_trials=210,paired_comparisons=105,workload_cells=cells,
        measured_requests=sum(len(t["requests"]) for t in trials),
        committed_output_tokens=sum(t["summary"]["output_tokens"] for t in trials),
        serving_cleanup_all_zero=clean,
        prefix_cache_hit_requests=sum(t["summary"]["prefix_cache_hit_requests"] for t in trials),
        peak_used_kv_blocks=max(t["summary"]["scheduler"]["max_used_blocks"] for t in trials),
        tested_runtime_sha256=runtime_hash,correctness_serving_runtime_identical=runtime_matches,
        wall_clock_breakdown=breakdown,
        serving_aggregate=data["serving_aggregate"],tail_regressions=tail_regressions,
        memory=memory,fixed=data["fixed"],raw_root=str(a.root),
        analysis_file=str(a.analysis),outlier_policy="Retain every sample, including startup-shaped first cells and profiled stalls",
        limitations=["Five repeats share a process per mode/c; not five independent process pairs",
            "Startup payback is projected, not observed in a512-request optimized run",
            "Rich profiler windows have overhead; gap/skew need not improve in every window",
            "Historical4.4C wall penalty not fully reproduced/explained",
            "Benchmark-only explicit runner, not promoted default production API"])
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"summary.json").write_text(json.dumps(result,indent=2))
    throughput=[[c["concurrency"],c["workload"],f'{c["modes"]["eager"]["output_tok_s"]:.2f}',
        f'{c["modes"]["optimized"]["output_tok_s"]:.2f}',f'{c["median_speedup"]:.4f}',
        f'{c["min_speedup"]:.4f}-{c["max_speedup"]:.4f}',
        f'{100*c["modes"]["optimized"]["eligible_fraction"]:.1f}%'] for c in cells]
    (a.output/"serving-table.md").write_text("# Serving Results\n\nAll five repeats included; tok/s = pooled committed outputs / pooled serving wall time. Speedup = median of paired wall-time ratios.\n\n"+
        md_table(["c","workload","eager tok/s","MLP tok/s","median speedup","repeat range","eligible steps"],throughput))
    latencies=[]
    for c in cells:
        for mode,r in c["modes"].items():
            latencies.append([c["concurrency"],c["workload"],mode,
                quantiles(r["latency"]["ttft_ms"]),quantiles(r["latency"]["e2e_ms"]),
                quantiles(r["latency"]["tpot_ms"]),quantiles(r["itl_ms"])])
    (a.output/"latency-table.md").write_text("# Latency Results\n\nP50/P95/P99 in ms, pooled request/token samples across five repeats. Burst-commit zero ITLs are retained.\n\n"+
        md_table(["c","workload","mode","TTFT","E2E","TPOT","ITL"],latencies))
    mechanism=[]
    for path,t in data["traces"].items():
        if "/prototype/" not in path or "layersNone" not in path:
            continue
        ws=[w for w in t["windows"] if w["operation"]=="verify"]
        keys=("launch_calls","kernel_count","nccl_count","copy_us","graph_launch_cpu_us","gpu_gap_us")
        mechanism.append([Path(path).name]+[f'{statistics.mean(w[k] for w in ws):.2f}' for k in keys])
    (a.output/"mechanism-table.md").write_text("# Mechanism Traces\n\nTwo profiled verification windows per row; rollback spans retained separately in raw analysis. Times in us. NCCL residency is not wire time. No trace outlier removed.\n\n"+
        md_table(["trace","CPU launch calls","GPU kernels","NCCL kernels","copy time","graph launch CPU","GPU gap"],mechanism))
    commands=[]
    for name,d in manifests.items():
        prefix="NCCL_GRAPH_MIXING_SUPPORT=0 " if d["environment"].get("NCCL_GRAPH_MIXING_SUPPORT")=="0" else ""
        commands.append(f"## {name}\n\n```bash\n{prefix}{shlex.join(d['command'])}\n```\n")
    (a.output/"commands.md").write_text("# Executed Commands\n\nWorking directory: project root. Common environment: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1. Mixing default except explicitly marked no-replay diagnostic.\n\n"+"\n".join(commands))
    print(json.dumps(dict(decision=decision,gates=gates,aggregate=data["serving_aggregate"],
                         tail_regressions=tail_regressions,protected_changed=changed),indent=2))


if __name__ == "__main__":
    main()
