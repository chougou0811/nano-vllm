"""Render compact diagnostic tables and final provenance after all runs finish."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

from benchmarks.serving.eagle3_phase52_analysis import table


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    path=a.output/"summary.json"
    d=json.loads(path.read_text())
    root=Path(d["raw_root"])
    manifest=json.loads((root/"eager/manifest.json").read_text())
    changed=[p for p,h in manifest["protected_before"].items()
             if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    flags=("all_normal_exit","pair_outputs_equal","pair_speculation_equal","cleanup_zero")
    if changed or d["validation"]["protected_changed"] or not all(d["validation"][k] for k in flags):
        raise RuntimeError(f"Validation failure: {changed}, {d['validation']}")
    test=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-q"],
                         text=True,capture_output=True)
    (root/"post-cpu-tests.log").write_text(test.stdout+test.stderr)
    if test.returncode:
        raise RuntimeError(test.stderr)
    count=int(re.search(r"Ran (\d+) tests",test.stderr)[1])
    d["validation"].update(cpu_tests=count,cpu_tests_passed=True,
        protected_files=len(manifest["protected_before"]),production_edits=0,historical_test_edits=0,
        git_commit_created=False)
    d["decision"]=dict(primary="Cross-request EAGLE-3 draft-step batching with serial conditioning first",
        secondary="Execution-mode-aware BF16 target MLP GEMM tactic selection",
        implementation_started=False,
        mlp_replay_status="Experimental opt-in with low-concurrency evidence, not main default optimization",
        largest_wall_component_all_concurrencies="Target verification",
        draft_is_largest_wall_component=False)
    rows=[]
    for c in (1,2,4):
        r=d["paired_accounting"][str(c)]
        rows.append([c,f'{d["speedups"][str(c)]:.4f}',
            f'{r["eager_wall_s"]-r["mlp_wall_s"]:.4f}',
            f'{r["eager_eligible_verify_s"]-r["mlp_eligible_verify_s"]:.4f}',
            f'{r["eager_fallback_verify_s"]-r["mlp_fallback_verify_s"]:.4f}',
            f'{r["saved_draft_ns_s"]:.4f}',f'{r["saved_target_prefill_ns_s"]:.4f}'])
    (a.output/"paired-table.md").write_text("# Paired Accounting\n\nSeconds saved over all21 trials/c, not per request. Positive saves time. All samples retained.\n\n"+
        table(["c","geomean speedup","total saved","eligible verify saved","fallback verify saved","draft saved","prefill saved"],rows))
    rows=[]
    for profile in d["profiles"]:
        if "-minimal-" not in profile["trace"]:
            continue
        for rank in profile["ranks"]:
            ts=rank["targets"]
            mean=lambda k: statistics.mean(t[k] for t in ts)
            rows.append([profile["mode"],profile["trace"].split("-minimal")[0],rank["rank"],
                *[f'{mean(k)/1000:.3f}' for k in ("host_span_us","gpu_envelope_us","gpu_idle_gap_us","communication_union_us")],
                f'{mean("launch_count")+mean("graph_launch_count"):.0f}',f'{mean("graph_launch_cpu_us")/1000:.3f}',
                f'{statistics.mean(t["collective_start_skew_mean_us"] for t in profile["rank_comparison"])/1000:.3f}',
                f'{statistics.mean(t["joint_nccl_residency_us"] for t in profile["rank_comparison"])/1000:.3f}'])
    (a.output/"profile-table.md").write_text("# Minimal Trace Diagnostics\n\nTimes in ms. Two verification windows/row. Profiler timings are perturbed; NCCL residence is not transport time. Rank comparison columns are shared across the paired ranks.\n\n"+
        table(["mode","cell","rank","host","GPU envelope","GPU idle gap","NCCL residence","launch APIs","graph launch CPU","mean rank skew","joint NCCL residence"],rows))
    rows=[]
    for profile in d["profiles"]:
        if "-profile-" not in profile["trace"]:
            continue
        ts=profile["ranks"][0]["targets"]
        keys=("mlp_region","mlp_gate_up","mlp_down","attention_varlen","qkv","output_projection","lm_head")
        rows.append([profile["mode"],profile["trace"].split("-profile")[0]]+
            [f'{statistics.mean(t["categories"].get(k,{}).get("sum_us",0) for t in ts)/1000:.3f}' for k in keys])
    (a.output/"operator-table.md").write_text("# Rich Trace Operator Attribution\n\nRank0 GPU kernel service sums in ms, not serving wall fractions. Captured MLP has no individual Python leaf calls; its graph kernels are attributed to mlp_region. Zero in a leaf column does not mean that computation disappeared.\n\n"+
        table(["mode","cell","MLP region","gate/up","down","attention","QKV","o_proj","LM head"],rows))
    rows=[]
    for r in d["metadata"]:
        if r["rank"]!=0:
            continue
        means={k:statistics.mean(t["metadata_ms"].get(k,0) for t in r["targets"])
               for k in ("verification_layout","_descriptors","prepare_prefill","_status")}
        rpc=[(v["end_ns"]-v["start_ns"])/1e6 for v in r["rpc"] if v["operation"]=="verify"]
        rows.append([r["mode"],r["file"].split("-timing")[0]]+[f'{v:.4f}' for v in means.values()]+[f'{statistics.mean(rpc):.4f}'])
    (a.output/"metadata-table.md").write_text("# Event-only Host Diagnostics\n\nRank0 ms; nested within target wall time. prepare_prefill includes tensor construction/copies, not pure CPU loops. _status includes synchronization. Dispatch is shared-memory packing/publication; rank1 read waits are not counted as extra serving latency.\n\n"+
        table(["mode","cell","layout","descriptors","prepare input","status","dispatch"],rows))
    d["environment"]={k:manifest[k] for k in ("git","config","target_revision","draft_revision","versions","topology")}
    d["hardware_check"]=json.loads((a.output/"hardware-check.json").read_text())
    d["limitations"]=["Three within-process repeats; two fresh systems, not independent fresh-process pairs per repeat",
        "Rich/minimal profiler overhead is substantial and rank-asymmetric; wall fractions use unprofiled trials",
        "No transport-only NCCL attribution, GPU hardware-counter roofline, or new feature speedup claim",
        "External source pins are audit-time default HEAD, not installed/tested serving stacks",
        "Current high-acceptance short/medium-context families do not establish all-workload generality"]
    path.write_text(json.dumps(d,indent=2))
    print(json.dumps(dict(decision=d["decision"],validation=d["validation"]),indent=2))


if __name__ == "__main__":
    main()
