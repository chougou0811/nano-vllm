"""Audit raw scheduler trials and aggregate without discarding slow samples."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import difflib
import json
import math
from pathlib import Path
import tarfile

from .metrics import distribution
from .report import write_json


def audit_source(archive, output):
    repo = Path(__file__).resolve().parents[2]
    prefix = repo.name + "/"
    with tarfile.open(archive) as saved:
        old = {m.name.removeprefix(prefix): saved.extractfile(m).read().decode()
               for m in saved.getmembers() if m.isfile() and m.name.startswith(prefix)
               and "/.git/" not in m.name and Path(m.name).suffix in [".py", ".md", ".toml"]}
    current = {str(p.relative_to(repo)): p.read_text() for p in repo.rglob("*")
               if p.is_file() and ".git" not in p.parts and p.suffix in [".py", ".md", ".toml"]}
    protected = [p for p in old if p.startswith(("nanovllm/layers/", "nanovllm/models/")) or p in [
        "nanovllm/engine/model_runner.py", "nanovllm/engine/block_manager.py", "nanovllm/engine/sequence.py",
        "nanovllm/engine/scheduler.py", "nanovllm/sampling_params.py"]]
    assert all(current[p] == old[p] for p in protected)
    def classes(source):
        return {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.ClassDef)}
    v1 = "nanovllm/engine/policy_scheduler.py"
    assert classes(old[v1]) == classes(current[v1]), "V1 implementation changed"
    changed = [p for p in sorted(set(old)|set(current)) if old.get(p) != current.get(p)]
    patch = "".join("".join(difflib.unified_diff(old.get(p, "").splitlines(True), current.get(p, "").splitlines(True),
                    fromfile="V1/"+p, tofile="V2/"+p)) for p in changed)
    (output/"v2-only.diff").write_text(patch)
    write_json(output/"source-scope-audit.json", dict(v1_classes_unchanged=True,
               protected_files_unchanged=protected, changed_since_v1=changed))
    return changed


def v1_causes(root):
    run = root/"14b-tp2-exploratory"
    data = json.loads((run/"summary.json").read_text())
    results = []
    for interval in ["mixed-350ms", "mixed-100ms"]:
        slo, static = [], []
        for trial in data:
            if trial["summary"]["profile"] != interval:
                continue
            steps = json.loads((run/trial["name"]/"steps.json").read_text())
            if trial["summary"]["policy"] == "slo-aware":
                slo += steps
            if trial["summary"]["policy"] == "static":
                static += steps
        prefill = [s for s in slo if s["is_prefill"]]
        decisions = [s["policy_decision"] for s in slo if s["policy_decision"].get("decode_slack_ns") is not None]
        results.append(dict(profile=interval, prefill_steps=len(prefill),
            prefill_budgets=dict(Counter(s["policy_decision"]["prefill_budget"] for s in prefill)),
            reasons=dict(Counter(s["policy_decision"]["reason"] for s in prefill)),
            decode_slack_ms=distribution(d["decode_slack_ns"]/1e6 for d in decisions),
            predicted_256_ms=distribution(max(d["predicted_decode_ns"], 256*d["prefill_ns_per_token"])/1e6 for d in decisions),
            actual_v1_128_ms=distribution(s["step_latency_ms"] for s in prefill if sum(s["scheduled_tokens"]) == 128),
            actual_static_256_ms=distribution(s["step_latency_ms"] for s in static
                                             if s["is_prefill"] and sum(s["scheduled_tokens"]) == 256)))
    return results


def aggregate(run):
    trials = json.loads((run/"trials.json").read_text())
    groups = defaultdict(list)
    all_steps = []
    for trial in trials:
        path = run/trial["name"]
        rows = json.loads((path/"requests.json").read_text())
        steps = json.loads((path/"steps.json").read_text())
        summary = trial["summary"]
        assert summary["complete"] and summary["kv_released"]
        assert summary["initial_prefix_cache_blocks"] == 0
        for row in rows:
            times = row["token_times_ns"]
            assert row["status"] == "completed"
            assert len(times) == len(row["output_token_ids"]) == row["requested_output_length"]
            assert times == sorted(times) and times[-1] == row["finish_ns"]
            assert row["arrival_ns"] <= row["admitted_ns"] <= row["enqueued_ns"] <= row["first_scheduled_ns"] <= times[0]
            assert math.isclose(row["queue_delay_ms"], row["ingress_queue_delay_ms"]+row["submission_ms"]+row["engine_queue_delay_ms"])
            assert row["num_schedule_steps"] == row["num_prefill_steps"]+row["num_decode_steps"]
        assert sum(len(r["output_token_ids"]) for r in rows) == summary["output_tokens"]
        key = summary["stage"], summary["interval_ms"], summary["policy"]
        groups[key].append((summary, rows, steps))
        all_steps.extend(dict(trial=trial["name"], **s) for s in steps)
    output = []
    for (stage, interval, policy), group in sorted(groups.items()):
        rows = [r for _,rs,_ in group for r in rs]
        steps = [s for _,_,ss in group for s in ss]
        duration = sum(s["duration_s"] for s,_,_ in group)
        good = sum(r["slo_met"] for r in rows)
        latency = {name: distribution(r[name] for r in rows) for name in ["ttft_ms", "tpot_ms", "e2e_ms",
                   "ingress_queue_delay_ms", "engine_queue_delay_ms", "queue_delay_ms"]}
        latency["itl_ms"] = distribution(v for r in rows for v in r["itl_ms"])
        decode_gaps, initial_decode_waits = [], []
        for _, requests, ss in group:
            events = defaultdict(list)
            for s in ss:
                if not s["is_prefill"]:
                    for rid in s["request_ids"]:
                        events[rid].append(s["scheduled_ns"])
            for r in requests:
                timestamps = events[r["request_id"]]
                if timestamps:
                    initial_decode_waits.append((timestamps[0]-r["first_token_ns"])/1e6)
                    decode_gaps.extend((b-a)/1e6 for a,b in zip(timestamps,timestamps[1:]))
        latency["decode_service_gap_ms"] = distribution(decode_gaps)
        latency["first_decode_wait_ms"] = distribution(initial_decode_waits)
        prefill = [s for s in steps if s["is_prefill"]]
        output.append(dict(stage=stage, interval_ms=interval, policy=policy, repeats=len(group),
            requests=len(rows), output_tokens=sum(len(r["output_token_ids"]) for r in rows),
            duration_s=duration, request_throughput=len(rows)/duration,
            token_throughput=sum(len(r["output_token_ids"]) for r in rows)/duration,
            good_requests=good, goodput=good/duration, violation_rate=1-good/len(rows), latency=latency,
            peak_waiting=max(s["queue_before"]["waiting"] for s in steps),
            peak_running=max(s["queue_before"]["running"] for s in steps),
            peak_waiting_tokens=max(s["queue_before"]["waiting_tokens"] for s in steps),
            prefill_steps=len(prefill), decode_steps=len(steps)-len(prefill),
            prefill_budgets=dict(Counter(s["policy_decision"].get("prefill_budget",sum(s["scheduled_tokens"])) for s in prefill)),
            actual_prefill_tokens=dict(Counter(sum(s["scheduled_tokens"]) for s in prefill)),
            reasons=dict(Counter(s["policy_decision"].get("reason", "original") for s in steps)),
            overload_steps=sum(s["policy_decision"].get("overloaded",False) for s in steps),
            observed_reallocations=sum(r["num_kv_allocations"]-1 for r in rows),
            step_latency_by_phase_batch={f"{phase}-{batch}": distribution(s["step_latency_ms"] for s in steps
                if s["is_prefill"] == (phase=="prefill") and s["batch_size"] == batch)
                for phase,batch in sorted({("prefill" if s["is_prefill"] else "decode",s["batch_size"]) for s in steps})},
            per_repeat=[dict(repeat=s["repeat"], duration=s["duration_s"], goodput=s["slo_goodput_rps"],
                tps=s["output_token_throughput_tps"], ttft_p99=s["latency"]["ttft_ms"]["p99"],
                itl_p99=s["latency"]["itl_ms"]["p99"]) for s,_,_ in group]))
    with (run/"all-steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_steps[0]))
        writer.writeheader()
        for row in all_steps:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--v1-dir", required=True)
    parser.add_argument("--backup-tar")
    args = parser.parse_args()
    run = Path(args.run_dir)
    changed = audit_source(args.backup_tar, run) if args.backup_tar else []
    groups = aggregate(run)
    causes = v1_causes(Path(args.v1_dir))
    manifest = json.loads((run/"manifest.json").read_text())
    frozen = json.loads((run/"frozen-config.json").read_text())
    assert manifest["status"] == "completed" and manifest["normal_exit"]
    write_json(run/"summary.json", dict(audit_passed=True, groups=groups, v1_root_cause=causes,
                                        frozen=frozen, manifest_status=manifest["status"]))
    lines = ["# Scheduler V2 Report", "", "All samples retained. Percentiles pool request or token observations across repeats; throughput/goodput divide total counts by summed windows. No significance claim from small repeat counts.", "",
        "## Preservation", "", "V1 dirty worktree archived before changes; V1 artifacts untouched. No Git commit or default policy switch. See pre-v2/worktree.tar.gz in the parent directory. TP, Attention, BlockManager, ModelRunner, Sequence and sampling source unchanged.", "",
        "## Protocol", "", f"Model revision: `{manifest['model_revision']}`. Git base: `{manifest['nano_vllm_commit']}` plus recorded dirty source snapshot. BF16 eager, TP=2, 2 x 4090, 2048 context/token budget, batch cap 4. Full environment, model hashes and command are in manifest.json.", "",
        "Tuning is separate from held-out. All workload tokens/order/arrival offsets were saved before execution. The minimum-chunk choice is frozen before any held-out timing. Targets: TTFT 2000 ms, request TPOT 100 ms. Open-loop step-boundary arrival; no external concurrency gate. Fresh prefix metadata and isolated warmups per trial, no initial cache hits.", "",
        "```json", json.dumps(frozen, indent=2), "```", "",
        "## V1 Root Cause", "",
        "Expired TTFT and class-progress overrides only grant the minimum chunk; per-token EWMA charges fixed step costs as a slope. Earliest decode slack becomes infeasible under overload, and V1 has no balanced fallback. Candidate chunk cost forecasts omit repeated interleaved decode turns. These source-level mechanisms are supported by the following traces, not claimed as isolated effects without ablations.", "",
        "| V1 profile | Prefill steps | Budget counts | Predicted 256 P50 ms | Actual static 256 P50 ms | Decode slack P50 ms |",
        "| --- | ---: | --- | ---: | ---: | ---: |"]
    for c in causes:
        lines.append(f"| {c['profile']} | {c['prefill_steps']} | {c['prefill_budgets']} | {c['predicted_256_ms']['p50']:.2f} | {c['actual_static_256_ms']['p50']:.2f} | {c['decode_slack_ms']['p50']:.2f} |")
    lines += ["", "## Algorithm", "", "V2 alternates feasible prefill/decode turns, chooses chunks using remaining-work completion forecasts and waiting age, estimates costs in phase/chunk/batch/context buckets, and applies an efficiency-chunk overload fallback with a soft per-step cap. Static and V2 share conservative reservation and decode rotation. Each feasible class receives a turn within two steps; wall-time bounds require bounded step duration. Arbitrary unbounded offered overload cannot have bounded request latency. Details and exact formulas are in docs/SCHEDULER_V2.md.", ""]
    lines += ["## Files Changed Since Preserved V1", "", *[f"- `{p}`" for p in changed], "",
              "`v2-only.diff` compares with the actual uncommitted V1 backup, not only Git HEAD. `source-scope-audit.json` checks the unchanged V1 class AST and protected source files.", ""]
    for stage in ["tuning", "held-out", "ablation", "sustained"]:
        lines += [f"## {stage}", "", "Latency ms; output throughput tokens/s; goodput requests/s. 0 ms arrival is a burst, not a finite offered rate.", "",
            "| Arrival ms / policy | Repeats | TTFT P95/P99 | TPOT P95/P99 | ITL P95/P99 | tokens/s | goodput | violation % | Peak waiting | Decode gap P99 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for g in groups:
            if g["stage"] != stage:
                continue
            def pair(name):
                return f"{g['latency'][name]['p95']:.2f}/{g['latency'][name]['p99']:.2f}"
            gap = g['latency']['decode_service_gap_ms']['p99']
            lines.append(f"| {g['interval_ms']} / {g['policy']} | {g['repeats']} | {pair('ttft_ms')} | {pair('tpot_ms')} | {pair('itl_ms')} | {g['token_throughput']:.2f} | {g['goodput']:.3f} | {g['violation_rate']*100:.2f} | {g['peak_waiting']} | {gap:.2f} |")
    lines += ["", "## Queue and Budget Details", "", "Arrival -> admitted is benchmark/step-boundary lag, not engine queue wait. Enqueued -> first_scheduled is engine initial waiting. Partial prefill continuation is not included in first-schedule queue delay, but is included in TTFT. Running can exceed max_num_seqs because the latter caps scheduled batch size.", "",
        "| Stage / arrival / policy | Admission P95 ms | Engine wait P95 ms | Peak waiting tokens | Prefill budget counts | Overload steps |",
        "| --- | ---: | ---: | ---: | --- | ---: |"]
    for g in groups:
        lines.append(f"| {g['stage']} / {g['interval_ms']} / {g['policy']} | {g['latency']['ingress_queue_delay_ms']['p95']:.2f} | {g['latency']['engine_queue_delay_ms']['p95']:.2f} | {g['peak_waiting_tokens']} | {g['prefill_budgets']} | {g['overload_steps']} |")
    lines += ["", "## Correctness and Safety", "",
        "See runtime-correctness/acceptance-probability.json and frozen-acceptance.json for independent HF/teacher-prefix validation. The model-specific small FP32 control and prior failed metric calibration are in sibling correctness directories. Original exact parity, NaN/Inf, position coverage and KV release are checked. Rules are calibrated using unchanged-original chunk/batch controls, not candidate errors. Raw absolute and unweighted L2 discrepancies remain reported; these criteria are finite numerical evidence, not a proof of semantic equivalence.", "",
        "Every measured request completed and all used KV blocks were released. No initial prefix reuse; raw timestamps/output counts/queue decomposition audited. This finite experiment cannot prove stability under indefinite overload. Reservation-removal ablation is only run when the entire finite workload's worst-case KV fits; it does not establish safety under memory pressure.", "",
        "## Reproduction and Limits", "", "```bash", manifest["command"], "```", "",
        "Use a new output directory and restore manifest environment/source snapshot. All per-trial requests.json/csv, tokens.csv, steps.json/csv and configs are retained. summary.json contains P50/P95/P99, per-repeat outcomes, service gaps, phase/batch latencies and budget distributions; all-steps.csv contains every step. Latencies are CPU Sequence commit times, not GPU-kernel or HTTP-client times.", "",
        "Only a synthetic finite workload family, three held-out repeats and two ablation repeats are measured. Thermal/host variation, token content and ordering remain confounders; paired data and rotation reduce but do not eliminate them. The sustained run is one diagnostic repeat. Do not infer a broad performance improvement, overload stability or readiness for EAGLE solely from this table.", ""]
    (run/"report.md").write_text("\n".join(lines))
    for g in groups:
        if g["stage"] == "held-out":
            print(g["interval_ms"], g["policy"], "TTFT99", round(g["latency"]["ttft_ms"]["p99"],2),
                  "ITL99", round(g["latency"]["itl_ms"]["p99"],2), "goodput", round(g["goodput"],3),
                  "tps", round(g["token_throughput"],2))


if __name__ == "__main__":
    main()
