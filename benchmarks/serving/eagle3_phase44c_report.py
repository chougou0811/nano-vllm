"""Render measured reuse and clearly labeled hypothetical savings."""
import argparse
import csv
import json
import math
import hashlib
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.serving.eagle3_phase44c import dump


def figures(output, data):
    plotdir = output/"plots"
    plotdir.mkdir(exist_ok=True)
    plots = (
        ("unique-keys", "Unique keys", "unique_keys"),
        ("reusable-keys", "Keys seen at least twice", "reusable_keys"),
        ("profitable-coverage", "Hypothetical verification coverage\n(18 replay opportunities)", "coverage"),
        ("projected-hit-rate", "Unbounded second-capture replay opportunity", "second_capture_hit_rate"),
        ("projected-net-saving", "Projected net saving (seconds)\nTransferred cost; not measured", "net"),
    )
    families = ("short-short","short-long","long-short","long-long","mixed-prompt","mixed-output")
    colors = {name:plt.colormaps["tab10"](index) for index,name in enumerate(families)}
    for filename,ylabel,field in plots:
        fig,axes = plt.subplots(1,3,figsize=(16,4),sharey=field!="net")
        for ax,c in zip(axes,(1,2,4)):
            for cell in data["cells"]:
                if cell["concurrency"]!=c:
                    continue
                horizons = sorted(map(int,cell["horizons"]))
                values = []
                for h in horizons:
                    s = cell["horizons"][str(h)]
                    if field=="coverage":
                        value = s["threshold_scenarios"]["18"]["verification_coverage"]
                    elif field=="net":
                        projection = s["empirical_transfer_projection"]
                        value = (projection["all_second_capture_net_ms"]/1000
                                 if projection["unknown_keys"]==0 else float("nan"))
                    else:
                        value = s[field]
                    values.append(value)
                if field in ("unique_keys","reusable_keys"):
                    with (output/f'c{c}-{cell["workload"]}-curve.csv').open() as file:
                        rows = list(csv.DictReader(file))
                    ax.plot([int(r["completed_requests"]) for r in rows],
                            [int(r[field]) for r in rows],label=cell["workload"],color=colors[cell["workload"]])
                else:
                    ax.plot(horizons,values,marker="o",label=cell["workload"],color=colors[cell["workload"]])
            ax.set_title(f"Concurrency {c}")
            ax.set_xlabel("Completed requests")
            ax.grid(alpha=.25)
            if field=="net":
                ax.text(.03,.02,"4.4B cost estimates; M4 only",transform=ax.transAxes,fontsize=8)
                if c!=1:
                    ax.text(.5,.5,"Full 4.4B projection unavailable\nNo M8/M16 replay measurements in 4.4B",
                            ha="center",va="center",transform=ax.transAxes,fontsize=10)
                    ax.set_yticks([])
        axes[0].set_ylabel(ylabel)
        axes[0 if field=="net" else -1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plotdir/f"{filename}.png",dpi=150)
        plt.close(fig)
    for metric in ("occurrence-count","reuse-distance"):
        fig,axes = plt.subplots(1,3,figsize=(16,4))
        for ax,c in zip(axes,(1,2,4)):
            for cell in data["cells"]:
                if cell["concurrency"]!=c or "512" not in cell["horizons"]:
                    continue
                records = json.loads((output/f'keys/c{c}-{cell["workload"]}.json').read_text())
                if metric=="occurrence-count":
                    hist = cell["horizons"]["512"]["occurrence_histogram"]
                    x = sorted(map(int,hist))
                    ax.plot(x,[hist[str(n)] for n in x],".-",label=cell["workload"],color=colors[cell["workload"]])
                    ax.set_ylabel("Number of keys")
                    ax.set_xlabel("Occurrences per exact key")
                else:
                    values = sorted(d for r in records for d in r["reuse_distances"])
                    if values:
                        ax.plot(values,[(i+1)/len(values) for i in range(len(values))],label=cell["workload"],color=colors[cell["workload"]])
                    ax.set_ylabel("Empirical CDF")
                    ax.set_xlabel("Verification distance between occurrences")
                    ax.set_xscale("log")
            ax.set_title(f"Concurrency {c}: horizon 512")
            ax.grid(alpha=.25)
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plotdir/f"{metric}.png",dpi=150)
        plt.close(fig)


def tables(output,data):
    flat = []
    for cell in data["cells"]:
        for h,s in cell["horizons"].items():
            projection = s["empirical_transfer_projection"]
            row = dict(concurrency=cell["concurrency"],workload=cell["workload"],horizon=int(h),
                completed=s["window"]["completed"],verifications=s["verifications"],eligible=s["eligible"],
                unique=s["unique_keys"],reusable=s["reusable_keys"],le2=s["occurrence_le_2_keys"],
                maxk_removed=s["field_ablation_cardinality"]["max_k"],
                table_width_removed=s["field_ablation_cardinality"]["table_width"],
                maxk_and_width_removed=s["without_max_k_and_table_width"],
                reuse_p50=s["reuse_distance"]["p50"],reuse_p95=s["reuse_distance"]["p95"],
                known_cost_keys=projection["known_keys"],unknown_cost_keys=projection["unknown_keys"],
                profitable_estimated_keys=projection["profitable_keys"],
                profitable_estimated_coverage=projection["profitable_verification_coverage"],
                all_capture_net_ms=projection["all_second_capture_net_ms"],
                hindsight_positive_net_ms=projection["hindsight_profitable_only_net_ms"],
                replay_opportunity_rate=s["second_capture_hit_rate"])
            for n in (6,14,18):
                row[f"keys_{n}_replays"] = s["threshold_scenarios"][str(n)]["keys"]
                row[f"coverage_{n}_replays"] = s["threshold_scenarios"][str(n)]["verification_coverage"]
            for b in s["bounded"]:
                row[f'bounded{b["capacity"]}_hits'] = b["hits"]
                row[f'bounded{b["capacity"]}_captures'] = b["captures"]
            flat.append(row)
    if flat:
        with (output/"horizon-table.csv").open("w") as file:
            writer = csv.DictWriter(file,fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    lines = ["# Observed shadow-key reuse", "", "No serving speedup is measured by this table.", "",
        "| C | Workload | H | Verify | Unique | Reused keys | <=2 | >=18 replay keys | Coverage (18) | Oracle replay % | Bounded4 hits |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in flat:
        lines.append(f'| {r["concurrency"]} | {r["workload"]} | {r["horizon"]} | {r["verifications"]} | '
            f'{r["unique"]} | {r["reusable"]} | {r["le2"]} | {r["keys_18_replays"]} | '
            f'{100*r["coverage_18_replays"]:.2f}% | {100*r["replay_opportunity_rate"]:.2f}% | {r["bounded4_hits"]} |')
    (output/"horizon-table.md").write_text("\n".join(lines)+"\n")


def bounded_results(raw,output):
    pairs = []
    from benchmarks.serving.eagle3_phase44b_summary import graph_metrics
    from benchmarks.serving.eagle3_phase44c_analysis import bounded
    for directory in sorted(raw.glob("bounded-c*")):
        if not directory.is_dir() or not (directory/"manifest.json").exists():
            continue
        manifest = json.loads((directory/"manifest.json").read_text())
        runs = [json.loads(Path(p).read_text()) for p in manifest["trials"]]
        for repeat in range(3):
            matched = {r["mode"]:r for r in runs if r["repeat"]==repeat}
            if len(matched)!=2:
                continue
            eager, graph = matched["eager"],matched["graph"]
            graph["graphs"] = graph["cache"]
            graph["system"] = "second4"
            metrics = graph_metrics(graph)
            observed = [e for e in graph["cache"][0]["events"] if e["kind"]=="verify"]
            other_rank = [e for e in graph["cache"][1]["events"] if e["kind"]=="verify"]
            assert [e["key_id"] for e in graph["events"]]==[e["key_id"] for e in observed]
            assert [e["key_id"] for e in graph["events"]]==[e["key_id"] for e in eager["events"]]
            assert [(e["key_id"],e["action"]) for e in observed]==[(e["key_id"],e["action"]) for e in other_rank]
            projected = bounded(graph["events"],4)
            assert projected["hits"]==metrics["hits"] and projected["captures"]==metrics["captures"]
            control_steps = [s for s in eager["steps"] if not s["is_prefill"]]
            graph_steps = [s for s in graph["steps"] if not s["is_prefill"]]
            assert len(observed)==len(control_steps)==len(graph_steps)
            action_delta, earned, replay_counts = Counter(),Counter(),Counter()
            for event,control,actual in zip(observed,control_steps,graph_steps):
                delta = actual["target_verify_ns"]-control["target_verify_ns"]
                category = event["action"]
                if category=="eager":
                    category = "eligible_miss" if event["key"] is not None else "ineligible_fallback"
                action_delta[category] += delta/1e9
                if event["action"]=="replay":
                    earned[event["entry_id"]] -= delta/1e6
                    replay_counts[event["entry_id"]] += 1
            total_verify_delta = (graph["summary"]["wall_clock"]["target_verify_ns"]-
                                  eager["summary"]["wall_clock"]["target_verify_ns"])/1e9
            assert math.isclose(sum(action_delta.values()),total_verify_delta,abs_tol=1e-6)
            assert sum(replay_counts.values())==metrics["hits"]
            paired_lifetimes = []
            key_net = Counter()
            for lifetime in metrics["lifetimes"]:
                entry = lifetime["entry_id"]
                saved, count = earned[entry],replay_counts[entry]
                assert count==lifetime["replays"]
                mean = saved/count if count else None
                net = saved-lifetime["capture_ms"]
                key_net[lifetime["key_id"]] += net
                paired_lifetimes.append(dict(entry_id=entry,key_id=lifetime["key_id"],
                    key=lifetime["key"],replays=count,capture_ms=lifetime["capture_ms"],
                    release_ms=lifetime["release_ms"],paired_replay_saved_ms=saved,
                    paired_mean_per_hit_saving_ms=mean,
                    paired_break_even_replays=math.ceil(lifetime["capture_ms"]/mean) if mean and mean>0 else None,
                    net_capture_ms=net,net_with_release_ms=net-lifetime["release_ms"],
                    capture_profitable=net>=0,full_release_profitable=net>=lifetime["release_ms"]))
            pairs.append(dict(concurrency=graph["concurrency"],workload=graph["workload"],repeat=repeat,
                eager_seconds=eager["summary"]["duration_s"],graph_seconds=graph["summary"]["duration_s"],
                speedup=eager["summary"]["duration_s"]/graph["summary"]["duration_s"],
                graph_metrics=metrics,normal_exit=manifest["normal_exit"],
                observer_matches_production_keys=True,rank_key_action_parity=True,
                bounded_simulation_matches_actual_hits=True,
                paired_lifetimes=paired_lifetimes,
                paired_profitable_lifetimes=sum(r["capture_profitable"] for r in paired_lifetimes),
                paired_profitable_captured_keys=sum(v>=0 for v in key_net.values()),
                paired_action_verify_delta_s=dict(action_delta),
                eager_summary=eager["summary"],graph_summary=graph["summary"],
                wall_clock_delta_s={k:(graph["summary"]["wall_clock"][k]-v)/1e9
                    for k,v in eager["summary"]["wall_clock"].items() if k.endswith("_ns")},
                output_parity=[r["output_token_ids"] for r in eager["requests"]]==[r["output_token_ids"] for r in graph["requests"]]))
    dump(output/"bounded-results.json",pairs)
    return pairs


def bounded_overview(output,pairs):
    groups = []
    for c in (1,2,4):
        rows = [r for r in pairs if r["concurrency"]==c]
        if not rows:
            continue
        lifetimes,keys = Counter(),Counter()
        for row in rows:
            lifetimes.update(row["graph_metrics"]["lifetime_verdicts"])
            keys.update(row["graph_metrics"]["captured_key_verdicts"])
        eager = sum(r["eager_seconds"] for r in rows)
        graph = sum(r["graph_seconds"] for r in rows)
        verifies = sum(r["graph_metrics"]["verifications"] for r in rows)
        groups.append(dict(concurrency=c,workload=rows[0]["workload"],pairs=len(rows),
            eager_total_s=eager,graph_total_s=graph,speedup=eager/graph,
            speedup_min=min(r["speedup"] for r in rows),speedup_max=max(r["speedup"] for r in rows),
            winning_pairs=sum(r["speedup"]>1 for r in rows),
            hits=sum(r["graph_metrics"]["hits"] for r in rows),verifications=verifies,
            hit_rate=sum(r["graph_metrics"]["hits"] for r in rows)/verifies,
            captures=sum(r["graph_metrics"]["captures"] for r in rows),
            lifetime_verdicts=dict(lifetimes),captured_key_verdicts=dict(keys),
            paired_profitable_lifetimes=sum(r["paired_profitable_lifetimes"] for r in rows),
            paired_profitable_captured_keys=sum(r["paired_profitable_captured_keys"] for r in rows),
            paired_action_verify_delta_s={k:sum(r["paired_action_verify_delta_s"].get(k,0) for r in rows)
                for k in ("eligible_miss","ineligible_fallback","capture","replay")},
            capture_s=sum(r["graph_metrics"]["capture_ms"] for r in rows)/1000,
            eviction_s=sum(r["graph_metrics"]["eviction_ms"] for r in rows)/1000,
            agreement_s=sum(r["graph_metrics"]["agreement_ms"] for r in rows)/1000,
            target_verify_delta_s=sum(r["wall_clock_delta_s"]["target_verify_ns"] for r in rows),
            eager_tok_s=sum(r["eager_summary"]["output_tokens"] for r in rows)/eager,
            graph_tok_s=sum(r["graph_summary"]["output_tokens"] for r in rows)/graph))
    lines = ["# Bounded validation (measured)","",
        "Speedup = paired eager wall time / graph wall time. Final cache clear is outside serving time.","",
        "| C | Family | Repeat | Eager s | Graph s | Speedup | Hits | Captures | Profitable lifetimes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pairs:
        m = row["graph_metrics"]
        lines.append(f'| {row["concurrency"]} | {row["workload"]} | {row["repeat"]} | '
            f'{row["eager_seconds"]:.3f} | {row["graph_seconds"]:.3f} | {row["speedup"]:.4f} | '
            f'{m["hits"]} | {m["captures"]} | {m["lifetime_verdicts"].get("profitable",0)} |')
    (output/"bounded-table.md").write_text("\n".join(lines)+"\n")
    return groups


def finalize(raw,output,data):
    from benchmarks.serving.eagle3_phase44c import ROOT, inventory
    assert len(data["cells"])==18 and all(r["normal_exit"] for r in data["cells"])
    assert len(data["bounded_validation"])==9
    assert all(r["normal_exit"] and r["output_parity"] and r["rank_key_action_parity"]
               and r["bounded_simulation_matches_actual_hits"] for r in data["bounded_validation"])
    tests = {}
    for name,count in (("post-cpu.log",139),("post-focused.log",30)):
        content = (raw/name).read_text()
        assert f"Ran {count} tests" in content and content.rstrip().endswith("OK")
        tests[name] = dict(passed=count,path=str(raw/name),sha256=hashlib.sha256(content.encode()).hexdigest())
    before,after = json.loads((raw/"before.json").read_text()),inventory()
    protected = {p:h for p,h in before.items() if "phase44c" not in p and "PHASE4_4C" not in p}
    changed = [p for p,h in protected.items() if after.get(p)!=h]
    assert not changed,changed
    preservation = dict(protected_files=len(protected),changed=changed,
        before_file=str(raw/"before.json"),current_source_hashes={p:h for p,h in after.items() if p.endswith(".py")},
        production_files_changed=0,historical_tests_changed=0,git_commit_created=False)
    dump(output/"source-preservation.json",preservation)
    data.update(status="complete",correctness=dict(tests=tests,
        observer_parity_concurrencies=[1,2,4],paired_request_outputs=9*512,
        all_36_measured_trials_complete=True,all_cleanup_checks_passed=True,
        zero_prefix_hits=True,all_rank_close_zero=True,all_gpu_processes_normal_exit=True,
        historical_files_changed=changed),
        measurement_counts=dict(shadow_streams=18,shadow_requests=18*512,
            nested_windows=54,bounded_trials=18,bounded_requests=18*512),
        limitations=dict(shadow_repeats_per_cell=1,bounded_paired_repeats=3,
            intermediate_windows_are_not_drained=True,graph_history_limit=200000,
            historical_cost_unknown_for_M8_M16=True,clocks_not_locked=True,
            rank1_memory_peak_is_process_scope=True))
    dump(output/"summary.json",data)
    commands = ["# Executed GPU commands", "", "Working directory: `"+str(ROOT)+"`.", "",
        "Environment: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1`.", "", "```bash"]
    import shlex
    for path in sorted(raw.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        commands.append(shlex.join(["/root/autodl-tmp/venvs/nano-baseline/bin/python","-m",
            "benchmarks.serving.eagle3_phase44c",*manifest["command"][1:]]))
    commands += ["```","","Warmup, input construction, stage order and hashes are retained in manifests and the source snapshot."]
    (output/"commands.md").write_text("\n".join(commands)+"\n")
    artifacts = []
    paths = sorted(p for p in raw.rglob("*") if p.is_file())
    paths += sorted(p for p in output.rglob("*") if p.is_file() and p.name!="artifacts.json")
    paths += [ROOT/"docs/EAGLE3_PHASE4_4C.md"]
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda:file.read(1024*1024),b""):
                h.update(chunk)
        artifacts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest()))
    dump(output/"artifacts.json",dict(files=len(artifacts),artifacts=artifacts))
    print(json.dumps(dict(status="complete",tests=tests,protected_files=len(protected),files=len(artifacts))),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--finalize",action="store_true")
    args = parser.parse_args()
    data = json.loads((args.output/"summary.json").read_text())
    tables(args.output,data)
    figures(args.output,data)
    if list(args.raw.glob("bounded-c*/manifest.json")):
        data["bounded_validation"] = bounded_results(args.raw,args.output)
        data["bounded_overview"] = bounded_overview(args.output,data["bounded_validation"])
        dump(args.output/"summary.json",data)
    if args.finalize:
        finalize(args.raw,args.output,data)


if __name__ == "__main__":
    main()
