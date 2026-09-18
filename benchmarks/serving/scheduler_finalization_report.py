"""Audited frozen-policy comparisons with explicit overload-state semantics."""
import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import tarfile

from .metrics import distribution
from .report import write_json
from .scheduler_v2_report import aggregate
from .scheduler_finalization import assert_frozen


def audit_finalization(root, archive):
    repo = Path(__file__).resolve().parents[2]
    with tarfile.open(archive) as saved:
        old = {m.name.removeprefix(repo.name+"/"): saved.extractfile(m).read()
               for m in saved.getmembers() if m.isfile() and "/.git/" not in m.name
               and Path(m.name).suffix in [".py",".md",".toml"]}
    current = {str(p.relative_to(repo)):p.read_bytes() for p in repo.rglob("*")
               if p.is_file() and ".git" not in p.parts and p.suffix in [".py",".md",".toml"]}
    changed = [p for p in sorted(set(old)|set(current)) if old.get(p) != current.get(p)]
    assert not any(p.startswith("nanovllm/") for p in changed)
    patch = "".join("".join(difflib.unified_diff(old.get(p,b"").decode().splitlines(True),
        current.get(p,b"").decode().splitlines(True),fromfile="frozen-v2/"+p,tofile="finalization/"+p)) for p in changed)
    (root/"v2.1-only.diff").write_text(patch)
    for p in changed:
        if p in current:
            target = root/"finalization_source"/p
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(current[p])
    write_json(root/"source-scope-audit.json",dict(production_unchanged=True,changed_files=changed,
        archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        finalization_source_hashes={p:hashlib.sha256(current[p]).hexdigest() for p in changed if p in current}))


def paired_stats(deltas):
    """Exact for five repeats; bound work for larger optional repeat counts."""
    if len(deltas) <= 6:
        samples = itertools.product(deltas,repeat=len(deltas))
    else:
        rng = random.Random(219)
        samples = (rng.choices(deltas,k=len(deltas)) for _ in range(10000))
    means = [statistics.mean(sample) for sample in samples]
    d = distribution(means)
    means.sort()
    def percentile(q):
        index = (len(means)-1)*q
        low = int(index)
        return means[low]+(means[min(low+1,len(means)-1)]-means[low])*(index-low)
    return dict(differences=deltas,mean=statistics.mean(deltas),
                positive=sum(x>0 for x in deltas),negative=sum(x<0 for x in deltas),
                bootstrap_mean_ci95=[percentile(.025),percentile(.975)],bootstrap_samples=d["count"])


def overload_stats(steps):
    evaluated = [s for s in steps if s["policy_decision"].get("overload_evaluated",False)]
    on = lambda s: bool(s["policy_decision"].get("overloaded",False))
    duration = sum(s["step_latency_ms"] for s in steps)
    eval_duration = sum(s["step_latency_ms"] for s in evaluated)
    return dict(steps=len(steps),evaluated_steps=len(evaluated),
        recorded_fraction=sum(map(on,steps))/len(steps),
        recorded_step_time_fraction=sum(s["step_latency_ms"] for s in steps if on(s))/duration,
        evaluated_fraction=sum(map(on,evaluated))/len(evaluated) if evaluated else None,
        evaluated_step_time_fraction=sum(s["step_latency_ms"] for s in evaluated if on(s))/eval_duration if eval_duration else None,
        retained_on_without_waiting=sum(on(s) and not s["policy_decision"].get("overload_evaluated",False) for s in steps),
        transitions=sum(s["policy_decision"].get("overload_transition",0) for s in steps),
        predicates=dict(Counter(
            "both" if s["policy_decision"].get("overload_decode_predicate") and s["policy_decision"].get("overload_backlog_predicate")
            else "decode" if s["policy_decision"].get("overload_decode_predicate")
            else "backlog" if s["policy_decision"].get("overload_backlog_predicate") else "neither"
            for s in evaluated)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--backup-tar",required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    manifest = json.loads((root/"manifest.json").read_text())
    repeats = manifest["arguments"]["repeats"]
    assert manifest["status"] == "completed" and manifest["normal_exit"]
    assert_frozen(manifest["arguments"]["frozen_manifest"])
    audit_finalization(root,args.backup_tar)
    plans = json.loads((root/"planned-workloads.json").read_text())
    digest = hashlib.sha256(json.dumps(plans,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert digest == manifest["workload_sha256"]
    expected = {f"{p['stage']}/arrival-{p['interval']}/repeat-{p['repeat']}/{p['policy']}":p for p in plans}
    trials = json.loads((root/"trials.json").read_text())
    assert len(trials) == len(expected) == len(plans)
    assert {t["name"] for t in trials} == set(expected)
    for trial in trials:
        path = root/trial["name"]
        assert json.loads((path/"workload.json").read_text()) == expected[trial["name"]]["requests"]
        assert trial["summary"]["arrival_mode"] == "open-loop"
    groups = aggregate(root)
    trial_data = defaultdict(list)
    warm_states = defaultdict(list)
    for trial in trials:
        s = trial["summary"]
        steps = json.loads((root/trial["name"]/"steps.json").read_text())
        key = s["stage"],s["interval_ms"],s["policy"]
        trial_data[key].append((s,steps))
        warm = json.loads((root/trial["name"]/"warmup-steps.json").read_text())
        warm_states[key].append(dict(repeat=s["repeat"],
            last_warmup_overloaded=warm[-1]["policy_decision"].get("overloaded"),
            warmup_steps=len(warm)))
    for g in groups:
        group = trial_data[g["stage"],g["interval_ms"],g["policy"]]
        assert len(group) == repeats and repeats >= 5
        steps = [step for _,ss in group for step in ss]
        g["overload"] = overload_stats(steps) if g["policy"] in ["frozen-v2","v2-no-overload"] else None
        g["controller_initial_state"] = warm_states[g["stage"],g["interval_ms"],g["policy"]]
        g["queue_depth"] = {k: distribution(s["queue_before"][k] for s in steps)
                            for k in ["waiting","running","waiting_tokens"]}
        total_ms = sum(s["step_latency_ms"] for s in steps)
        g["queue_step_time_weighted_mean"] = {k: sum(s["queue_before"][k]*s["step_latency_ms"] for s in steps)/total_ms
                                             for k in ["waiting","running","waiting_tokens"]}
        g["maximum_step_latency_ms"] = max(s["step_latency_ms"] for s in steps)
        g["per_repeat_overload"] = [dict(repeat=s["repeat"],**overload_stats(ss)) for s,ss in group]
        g["per_repeat_budget_counts"] = [dict(repeat=s["repeat"],budgets=dict(Counter(
            step["policy_decision"].get("prefill_budget",sum(step["scheduled_tokens"]))
            for step in ss if step["is_prefill"]))) for s,ss in group]
        longest = 0
        for _,ss in group:
            streak = 0
            for step in ss:
                if step["is_prefill"]:
                    streak = streak+1 if step["policy_decision"].get("prefill_budget") == 256 else 0
                    longest = max(longest,streak)
        g["longest_min_budget_prefill_streak"] = longest
    paired = []
    indexed = {(g["stage"],g["interval_ms"],g["policy"]):g for g in groups}
    for stage,interval in sorted({(g["stage"],g["interval_ms"]) for g in groups}):
        for candidate,baseline in [("frozen-v2","static-512"),("v2-no-overload","frozen-v2")]:
            a = {r["repeat"]:r for r in indexed[stage,interval,candidate]["per_repeat"]}
            b = {r["repeat"]:r for r in indexed[stage,interval,baseline]["per_repeat"]}
            assert set(a) == set(b)
            metrics = {k: paired_stats([a[r][k]-b[r][k] for r in sorted(a)])
                       for k in ["goodput","tps","ttft_p99","itl_p99"]}
            paired.append(dict(stage=stage,interval_ms=interval,candidate=candidate,baseline=baseline,metrics=metrics))
    correctness = json.loads((root/"correctness/acceptance.json").read_text())
    assert correctness["passed"]
    write_json(root/"summary.json",dict(audit_passed=True,groups=groups,paired=paired,
        correctness_passed=True,manifest_status=manifest["status"],
        totals={k:sum(g[k] for g in groups) for k in ["repeats","requests","output_tokens"]},
        uncertainty="Paired empirical bootstrap: exact up to six repeats, otherwise 10000 seeded resamples; descriptive only, no multiple-comparison correction."))
    lines = ["# Frozen Scheduler Finalization: Detailed Results","",
        f"All raw samples retained. {repeats} paired repeats per configuration; pooled request/token quantiles.",
        "Throughput/goodput use summed counts divided by summed windows, including arrivals and drain.",
        "SLO: TTFT <= 2000 ms and request-average TPOT <= 100 ms, not every-token ITL <= 100 ms.","",
        "## Latency and Throughput","",
        "Latencies in ms; output throughput tokens/s; goodput requests/s.","",
        "| Mix / arrival ms / policy | TTFT P50/P95/P99 | TPOT P50/P95/P99 | ITL P50/P95/P99 | Tokens/s | Goodput | Violation % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def quantiles(d):
        return "/".join(f"{d[k]:.2f}" if d[k] is not None else "N/A" for k in ["p50","p95","p99"])
    for g in groups:
        l = g["latency"]
        lines.append(f"| {g['stage']} / {g['interval_ms']} / {g['policy']} | {quantiles(l['ttft_ms'])} | {quantiles(l['tpot_ms'])} | {quantiles(l['itl_ms'])} | {g['token_throughput']:.2f} | {g['goodput']:.3f} | {g['violation_rate']*100:.1f} |")
    lines += ["","## Queue and Service","",
        "Queue peaks are pre-step samples, not a continuous-time maximum. Admission lag is benchmark/step-boundary lag, not engine waiting. Partial-prefill time is included in TTFT but not engine initial waiting.","",
        "| Mix / arrival / policy | Peak waiting / running | Peak waiting tokens | Admission P95 | Engine wait P95 | First-decode wait P50/P95/P99 | Decode service gap P50/P95/P99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        l = g["latency"]
        lines.append(f"| {g['stage']} / {g['interval_ms']} / {g['policy']} | {g['peak_waiting']}/{g['peak_running']} | {g['peak_waiting_tokens']} | {l['ingress_queue_delay_ms']['p95']:.2f} | {l['engine_queue_delay_ms']['p95']:.2f} | {quantiles(l['first_decode_wait_ms'])} | {quantiles(l['decode_service_gap_ms'])} |")
    lines += ["","## Budgets and Overload","",
        "Recorded occupancy includes state retained when no waiting exists. Evaluated occupancy only counts waiting-present decisions. Time-weighting covers engine steps, not idle time. Budget is capacity, not actual tokens consumed. Controller state (including overload) inherits the policy warmup; per-repeat initial states are in summary.json.","",
        "| Mix / arrival / policy | Budget counts | Recorded / evaluated overload fraction | Transitions | Predicate counts |",
        "| --- | --- | --- | ---: | --- |"]
    for g in groups:
        o = g["overload"]
        fractions = f"{o['recorded_fraction']:.3f}/{o['evaluated_fraction']:.3f}" if o and o['evaluated_fraction'] is not None else "N/A"
        lines.append(f"| {g['stage']} / {g['interval_ms']} / {g['policy']} | {g['prefill_budgets']} | {fractions} | {o['transitions'] if o else 'N/A'} | {o['predicates'] if o else 'N/A'} |")
    lines += ["","## Paired Repeat Differences","",
        "Candidate minus baseline. TTFT/ITL lower is better; throughput/goodput higher is better. Values are means [empirical-bootstrap 95% interval]. Small repeat counts and many comparisons do not establish statistical significance.","",
        "| Mix / arrival / candidate - baseline | Goodput | Tokens/s | TTFT P99 | ITL P99 |",
        "| --- | --- | --- | --- | --- |"]
    for p in paired:
        cells=[]
        for k in ["goodput","tps","ttft_p99","itl_p99"]:
            d=p["metrics"][k]
            cells.append(f"{d['mean']:.3f} [{d['bootstrap_mean_ci95'][0]:.3f}, {d['bootstrap_mean_ci95'][1]:.3f}]")
        lines.append(f"| {p['stage']} / {p['interval_ms']} / {p['candidate']} - {p['baseline']} | "+" | ".join(cells)+" |")
    lines += ["","## Correctness and Reproduction","",
        "Fixed-rule correctness passed for all five configurations plus original repeat. Exact original parity, finite values, coverage and KV release checked. No numerical thresholds recalibrated.",
        "All measured requests complete; initial prefix-cache hits are zero; all KV references drain. This is finite evidence, not proof of indefinite overload stability.",
        "See correctness/acceptance.json, per-trial raw JSON/CSV, all-steps.csv, freeze.json, manifest.json and source_snapshot/.","",
        "```bash",manifest["command"],"```","",
        "Use a fresh output directory. The repository protocol specifies frozen settings and warmup. Production code is unchanged from V2; diagnostics add small host overhead included in measured times."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(groups=len(groups),trials=sum(g["repeats"] for g in groups),audit_passed=True)))


if __name__ == "__main__":
    main()
