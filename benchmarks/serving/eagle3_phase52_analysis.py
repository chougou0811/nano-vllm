"""Offline Phase5.2 wall fractions and rank-correlated profiler attribution."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from benchmarks.serving.eagle3_phase42 import distribution
from benchmarks.serving.eagle3_phase43_summary import analyze_trace, rank_comparison, contains


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"|".join("---" for _ in headers)+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])+"\n"


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    ds={mode:json.loads((a.root/mode/"manifest.json").read_text()) for mode in ("eager","mlp")}
    if any(d.get("failure") or not d["normal_exit"] or d["protected_changed"] for d in ds.values()):
        raise RuntimeError("Incomplete/failing run: do not silently summarize a subset")
    result=dict(phase="5.2",raw_root=str(a.root),serving=[],cells=[],profiles=[],metadata=[],
        validation=dict(all_normal_exit=True,protected_changed=[],pair_outputs_equal=True,
            pair_speculation_equal=True,cleanup_zero=True),external_transfer_implemented=False)
    for mode,d in ds.items():
        for c in (1,2,4):
            trials=[t for t in d["trials"] if t["concurrency"]==c]
            steps=[s for t in trials for s in t["steps"]]
            decode=[s for s in steps if not s["is_prefill"]]
            eligible=[s for s in decode if s["batch_size"] in (1,2,4) and s["target_query_tokens"]==4*s["batch_size"]]
            duration=sum(t["summary"]["duration_s"] for t in trials)
            times={k:sum(t["summary"]["wall_clock"][k] for t in trials)/1e9
                for k in ("draft_ns","draft_catchup_ns","target_verify_ns","target_prefill_ns",
                          "schedule_ns","reserve_ns","accept_commit_ns","sync_other_ns","driver_refill_ns")}
            times["draft_generation_only_ns"]=times["draft_ns"]-times["draft_catchup_ns"]
            result["serving"].append(dict(mode=mode,concurrency=c,trials=len(trials),duration_s=duration,
                output_tokens=sum(t["summary"]["output_tokens"] for t in trials),
                times_s=times,fractions={k:v/duration for k,v in times.items()},
                eligible_fraction=len(eligible)/len(decode),
                eligible_verify_time_fraction=sum(s["target_verify_ns"] for s in eligible)/sum(s["target_verify_ns"] for s in decode),
                mean_decode_batch=statistics.mean(s["batch_size"] for s in decode),
                mean_target_query_tokens=statistics.mean(s["target_query_tokens"] for s in decode),
                target_ms=distribution(s["target_verify_ns"]/1e6 for s in decode),
                draft_ms=distribution(s.get("draft_ns",0)/1e6 for s in decode),
                acceptance=sum(t["summary"]["speculation"]["accepted_tokens"] for t in trials)/sum(t["summary"]["speculation"]["proposed_tokens"] for t in trials),
                peak_used_blocks=max(t["summary"]["scheduler"]["max_used_blocks"] for t in trials),
                peak_allocated=max(t["summary"]["memory"]["rank0_peak_allocated"] for t in trials)))
    left={(t["concurrency"],t["workload"],t["repeat"]):t for t in ds["eager"]["trials"]}
    pairs=[]
    accounting={c:defaultdict(float) for c in (1,2,4)}
    for t in ds["mlp"]["trials"]:
        old=left[t["concurrency"],t["workload"],t["repeat"]]
        outputs=[r["output_token_ids"] for r in t["requests"]]==[r["output_token_ids"] for r in old["requests"]]
        specs=t["summary"]["speculation"]==old["summary"]["speculation"]
        acc=accounting[t["concurrency"]]
        acc["eager_wall_s"]+=old["summary"]["duration_s"]
        acc["mlp_wall_s"]+=t["summary"]["duration_s"]
        if len(t["steps"])!=len(old["steps"]):
            raise RuntimeError("Step alignment differs: paired component attribution invalid")
        for ls,rs in zip(old["steps"],t["steps"]):
            if any(ls.get(k)!=rs.get(k) for k in ("is_prefill","batch_size","target_query_tokens")):
                raise RuntimeError("Shape alignment differs")
            if ls["is_prefill"]:
                continue
            eligible=ls["batch_size"] in (1,2,4) and ls["target_query_tokens"]==4*ls["batch_size"]
            key="eligible" if eligible else "fallback"
            acc[f"eager_{key}_verify_s"]+=ls["target_verify_ns"]/1e9
            acc[f"mlp_{key}_verify_s"]+=rs["target_verify_ns"]/1e9
        for key in ("draft_ns","target_prefill_ns","accept_commit_ns","schedule_ns"):
            acc[f"saved_{key}_s"]+=(old["summary"]["wall_clock"][key]-t["summary"]["wall_clock"][key])/1e9
        result["validation"]["pair_outputs_equal"] &= outputs
        result["validation"]["pair_speculation_equal"] &= specs
        pairs.append(dict(concurrency=t["concurrency"],workload=t["workload"],repeat=t["repeat"],
            speedup=old["summary"]["duration_s"]/t["summary"]["duration_s"],outputs_equal=outputs,speculation_equal=specs))
    result["pairs"]=pairs
    result["paired_accounting"]={str(c):dict(row) for c,row in accounting.items()}
    for c in (1,2,4):
        for family in sorted({r["workload"] for r in pairs}):
            rows=[r for r in pairs if r["concurrency"]==c and r["workload"]==family]
            result["cells"].append(dict(concurrency=c,workload=family,speedups=[r["speedup"] for r in rows],
                median_speedup=statistics.median(r["speedup"] for r in rows)))
    result["speedups"]={str(c):math.exp(statistics.mean(math.log(r["speedup"]) for r in pairs if r["concurrency"]==c)) for c in (1,2,4)}
    for mode,d in ds.items():
        for t in d["trials"]:
            if any(t["summary"]["cleanup"].values()) or any(r["targets"] for r in t["rank_cleanup"]["ranks"]):
                result["validation"]["cleanup_zero"]=False
        for path in sorted((a.root/mode).glob("*-rank0.trace.json")):
            analyses=[]
            record=dict(mode=mode,trace=path.name,ranks=[])
            for rank in (0,1):
                tracepath=Path(str(path).replace("rank0",f"rank{rank}"))
                meta=json.loads(Path(str(tracepath).replace(".trace.json",".json")).read_text())
                trace=json.loads(tracepath.read_text())
                analysis=analyze_trace(trace)
                analyses.append(analysis)
                scopes=sorted([e for e in trace["traceEvents"] if e.get("cat")=="user_annotation" and e["name"].startswith("diag.target.")],key=lambda e:e["ts"])
                for target,info,scope in zip(analysis["targets"],meta["targets"],scopes):
                    target.update(rows=info["rows"],batch=info["batch"],operation=info["operation"],gpu_event_ms=info["gpu_event_ms"])
                    apis=[e for e in trace["traceEvents"] if e.get("cat") in ("cuda_runtime","cuda_driver") and e.get("ph")=="X" and contains(scope,e)]
                    target["graph_launch_count"]=sum("GraphLaunch" in e["name"] for e in apis)
                    target["graph_launch_cpu_us"]=sum(e["dur"] for e in apis if "GraphLaunch" in e["name"])
                record["ranks"].append(dict(rank=rank,targets=analysis["targets"],
                    trace_gpu_events=analysis["trace_gpu_events"],target_gpu_events=analysis["target_gpu_events"]))
            record["rank_comparison"]=rank_comparison(analyses)
            # Full event-level data is an external artifact, not tracked report bulk.
            path.with_name(path.name.replace("-rank0.trace.json","-classified.json")).write_text(json.dumps(analyses))
            for rank in record["ranks"]:
                for target in rank["targets"]:
                    target.pop("kernels")
            result["profiles"].append(record)
        for path in sorted((a.root/mode).glob("*-timing-rank*.json")):
            d=json.loads(path.read_text())
            targets=[]
            for t in d["targets"]:
                rows=[r for r in d["metadata"] if t["start_ns"]<=r["start_ns"] and r["end_ns"]<=t["end_ns"]]
                sums=defaultdict(float)
                for r in rows:
                    sums[r["name"]]+=(r["end_ns"]-r["start_ns"])/1e6
                targets.append(dict(**t,host_ms=(t["end_ns"]-t["start_ns"])/1e6,metadata_ms=dict(sums)))
            result["metadata"].append(dict(mode=mode,file=path.name,rank=d["rank"],targets=targets,
                rpc=d["rpc"]))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"summary.json").write_text(json.dumps(result,indent=2))
    rows=[]
    for r in result["serving"]:
        f=r["fractions"]
        rows.append([r["mode"],r["concurrency"],f'{r["output_tokens"]/r["duration_s"]:.2f}']+
            [f'{100*f[k]:.2f}%' for k in ("target_verify_ns","draft_ns","draft_catchup_ns","target_prefill_ns","accept_commit_ns","schedule_ns")]+
            [f'{100*(1-r["eligible_fraction"]):.2f}%'])
    (a.output/"wall-table.md").write_text("# New Unprofiled Serving Measurements\n\nCatch-up is nested inside draft, not additive. All three repeats retained.\n\n"+table(
        ["mode","c","tok/s","verify","draft total","catch-up subset","prefill","accept/commit","schedule","ineligible steps"],rows))
    print(json.dumps(dict(speedups=result["speedups"],validation=result["validation"],profiles=len(result["profiles"])),indent=2))


if __name__ == "__main__":
    main()
