import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from time import get_clock_info


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def command(args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def capture_manifest(args, engine_config, specs, output_dir, argv):
    import os
    import torch

    repo = Path(__file__).resolve().parents[2]
    model = Path(args.model).resolve()
    files = sorted(p for p in model.iterdir() if p.is_file() and
                   (p.suffix in [".json", ".safetensors", ".txt", ".jinja"] or p.name == "LICENSE"))
    model_files = {}
    for file in files:
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        model_files[file.name] = dict(size=file.stat().st_size, sha256=digest)
    if not any(name.endswith(".safetensors") for name in model_files):
        raise ValueError("Model directory has no safetensors weights")
    verified = False
    if args.model_manifest:
        provenance = json.loads(Path(args.model_manifest).read_text())
        if provenance["revision"] != args.model_revision:
            raise ValueError("Requested revision differs from model manifest")
        expected = {f["file"]: f for f in provenance["files"]}
        for name, actual in model_files.items():
            entry = expected.get(name)
            if entry is None:
                raise ValueError(f"Model manifest does not cover {name}")
            if actual["size"] != entry["bytes"]:
                raise ValueError(f"Model size mismatch: {name}")
            if entry.get("sha256"):
                if actual["sha256"] != entry["sha256"]:
                    raise ValueError(f"Model SHA256 mismatch: {name}")
            elif entry.get("git_blob"):
                data = (model / name).read_bytes()
                blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                if blob != entry["git_blob"]:
                    raise ValueError(f"Model Git blob mismatch: {name}")
            else:
                raise ValueError(f"No expected checksum for {name}")
        verified = True
        write_json(output_dir / "model_manifest.json", provenance)
    snapshot = output_dir / "source_snapshot"
    sources = set(repo.glob("nanovllm/**/*.py")) | set(repo.glob("benchmarks/**/*.py"))
    sources |= set(repo.glob("tests/test_*.py"))
    sources |= set(repo.glob("docs/*.md"))
    sources |= {repo / "pyproject.toml", repo / "AGENTS.md", repo / "benchmarks/serving/README.md"}
    hashes = {}
    for path in sorted(sources):
        if not path.is_file():
            continue
        relative = path.relative_to(repo)
        data = path.read_bytes()
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    diff = command(["git", "diff", "HEAD"], repo)
    (output_dir / "source.diff").write_text(diff + "\n")
    (output_dir / "pip_freeze.txt").write_text(command([sys.executable, "-m", "pip", "freeze"]) + "\n")
    clock = get_clock_info("perf_counter")
    workload_json = json.dumps(specs, sort_keys=True, separators=(",", ":"))
    packages = {}
    for name in ["torch", "triton", "transformers", "flash-attn", "xxhash", "safetensors"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(
        schema_version=2, command=shlex.join([sys.executable, "-m", "benchmarks.serving", *argv]),
        arguments=vars(args), requested_engine_config=engine_config,
        model_path=str(model), model_revision=args.model_revision,
        model_revision_verified_against_manifest=verified, model_files=model_files,
        nano_vllm_commit=command(["git", "rev-parse", "HEAD"], repo),
        git_status=command(["git", "status", "--short"], repo), source_hashes=hashes,
        workload_sha256=hashlib.sha256(workload_json.encode()).hexdigest(),
        python=sys.version, platform=platform.platform(), packages=packages,
        torch_cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
        gpu=command(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"]),
        topology=command(["nvidia-smi", "topo", "-m"]),
        environment={k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "TORCH", "CUDA_", "HF_")) or k in ["OMP_NUM_THREADS", "TRITON_CACHE_DIR", "TMPDIR", "TRANSFORMERS_OFFLINE"]},
        clock=dict(name="perf_counter_ns", implementation=clock.implementation,
                   monotonic=clock.monotonic, resolution_s=clock.resolution),
        measurement_boundary="logical arrival to committed CPU Sequence output; excludes network and text processing",
        prefix_cache="Original cache retained; exact workload and warmup tokens saved",
        status="prepared",
        compiler_cache_initial={name: cache_inventory(os.environ[name]) for name in
                                ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"] if name in os.environ},
    )


def cache_inventory(directory):
    root = Path(directory)
    return [{"file": str(p.relative_to(root)), "size": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob("*")) if p.is_file()]


def write_artifacts(output_dir, result, summary, rows, manifest):
    write_json(output_dir / "requests.json", rows)
    write_json(output_dir / "steps.json", result.steps)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)
    with (output_dir / "steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.steps[0]) if result.steps else [])
        writer.writeheader()
        for row in result.steps:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    with (output_dir / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    with (output_dir / "tokens.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["request_id", "token_index", "token_id", "commit_ns", "itl_ns"])
        for row in rows:
            previous = None
            for index, (token, timestamp) in enumerate(zip(row["output_token_ids"], row["token_times_ns"])):
                writer.writerow([row["request_id"], index, token, timestamp,
                                 None if previous is None else timestamp - previous])
                previous = timestamp

    def number(value):
        return "N/A" if value is None else f"{value:.6f}"

    lines = ["# Serving Benchmark Report", "",
             "Framework validation only. These measurements do not establish a performance conclusion.", "",
             f"- Complete: {summary['complete']}; error: {summary['error'] or 'none'}.",
             f"- Arrival mode: {summary['arrival_mode']}; open-loop ignores concurrency admission cap.",
             f"- Prefix cache blocks reused: {summary['prefix_cache_blocks']}; initial hit requests: {summary['prefix_cache_hit_requests']}.",
             f"- Maximum waiting before / after schedule: {summary['max_waiting']} / {summary['max_waiting_after_schedule']}.",
             f"- Model: {manifest.get('model_path')}; revision: `{manifest.get('model_revision')}`.",
             f"- nano-vllm commit: `{manifest.get('nano_vllm_commit')}`; dirty/untracked source snapshot included.",
             f"- Planned / arrived / completed: {summary['planned_requests']} / {summary['arrived_requests']} / {summary['completed_requests']}.",
             f"- Failed / timed out / not arrived: {summary['failed_requests']} / {summary['timed_out_requests']} / {summary['not_arrived_requests']}.",
             f"- Window: {number(summary['duration_s'])} s, including logical-arrival gaps and drain; excludes initialization/warmup.",
             f"- Request throughput: {number(summary['request_throughput_rps'])} requests/s.",
             f"- Output token throughput: {number(summary['output_token_throughput_tps'])} tokens/s; includes EOS and any tokens committed before failure.", "",
             "## Latencies", "",
             "Completed requests only. TPOT is request-weighted; ITL pools individual token intervals. All values below are milliseconds, with linear percentiles.", "",
             "| Metric | Count | Mean | P50 | P95 | P99 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, stats in summary["latency"].items():
        lines.append(f"| {name} | {stats['count']} | {number(stats['mean'])} | {number(stats['p50'])} | {number(stats['p95'])} | {number(stats['p99'])} |")
    lines += ["", "Total queue delay = first schedule - logical arrival; it must NOT all be attributed to the Scheduler. Arrival -> admitted is step-boundary / benchmark admission lag (also concurrency gating in closed mode). Admitted -> enqueued is submission overhead. Enqueued -> first scheduled is engine initial scheduling wait. The compatibility field scheduler_queue_delay_ms uses admitted -> first scheduled, including submission overhead. These are initial waits, not cumulative preemption/decode wait.",
              "", "## SLO", "",
              f"- TTFT threshold: {number(summary['ttft_slo_ms'])} ms; request TPOT threshold: {number(summary['tpot_slo_ms'])} ms.",
              f"- Violation rate: {number(summary['slo_violation_rate'])}; goodput: {number(summary['slo_goodput_rps'])} requests/s.",
              "- Denominator is arrived requests; failures/timeouts violate configured SLOs. TPOT is inapplicable to one-token outputs. No thresholds means N/A, not zero violations.",
              "", "## Measurement Boundaries", "",
              "Arrivals are deterministic logical offsets, not observed network receipt. Admission happens between original steps. Closed mode gates active requests; open-loop submits all due requests regardless of active count. Step-boundary lateness is retained, never assigned to engine queue time. Initialization and all warmup are excluded.",
              "", "Tokens are timestamped after original postprocess commits them to Sequence. Intermediate prefill samples and uncommitted proposals are excluded. Multiple committed tokens in one step share a timestamp, so their internal ITLs are zero, not reconstructed GPU times.",
              "", "No extra CUDA synchronization or token streaming transport is used. Optional diagnostics add host spans and telemetry, are explicitly marked and are not comparative performance trials. These are in-process token-commit latencies, not GPU kernel timings or full client-visible HTTP/text latencies.",
              "", f"Observer bookkeeping host time: {summary['observer_bookkeeping_ns']} ns across {summary['observer_calls']} delegated calls. This excludes original scheduler work and the driver; it is not a claim of total zero perturbation.",
              "", "## Reproduction", "", "```bash", manifest.get("command", ""), "```", "",
              "Use a new output directory for reproduction. See manifest.json, workload.json, warmup.json, source_snapshot/, source.diff and pip_freeze.txt. Raw records: requests.json/requests.csv, tokens.csv and steps.json."]
    if manifest.get("observer_verification") is not None:
        lines += ["", "## Observer Verification", "", "```json",
                  json.dumps(manifest["observer_verification"], indent=2), "```"]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_repeat_report(output, trials, manifest):
    write_json(output / "summary.json", dict(schema_version=2, status=manifest["status"],
               arrival_mode=manifest["arrival_mode"], warmup_mode=manifest["warmup_mode"],
               diagnostics=manifest["diagnostics"], trials=trials, probes=manifest.get("probes", [])))
    lines = ["# Repeated Framework Validation", "", "No performance improvement claim. No samples removed.", "",
             f"Mode: {manifest['arrival_mode']}; warmup: {manifest['warmup_mode']}; diagnostic: {manifest['diagnostics']}; normal exit: {manifest.get('normal_exit')}.", "",
             "Arrival -> admitted is benchmark/step-boundary lag, not Scheduler wait. Admitted -> enqueued is submission overhead; enqueued -> first scheduled is engine initial waiting. Total queue delay combines these and must not be attributed entirely to Scheduler.", "",
             "| Repeat | Complete | Requests | Admission P95 ms | Engine wait P95 ms | TTFT P95 ms | Max decode ms | Waiting peak | Prefix blocks reused |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for trial in trials:
        s = trial["summary"]
        lat = s["latency"]
        maximum = max((v["max"] for v in s["step_latency_by_batch"].values() if v["max"] is not None), default=0)
        def value(key):
            v = lat[key]["p95"]
            return "N/A" if v is None else f"{v:.3f}"
        lines.append(f"| {trial['name']} | {s['complete']} | {s['completed_requests']} | {value('ingress_queue_delay_ms')} | {value('engine_queue_delay_ms')} | {value('ttft_ms')} | {maximum:.3f} | {s['max_waiting']} | {s['prefix_cache_blocks']} |")
    lines += ["", "## Decode Step Details", "",
              "| Repeat | Batch size | Count | P50 ms | P95 ms | P99 ms | Max ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for trial in trials:
        for batch, stats in trial["summary"]["step_latency_by_batch"].items():
            lines.append(f"| {trial['name']} | {batch} | {stats['count']} | {stats['p50']:.3f} | {stats['p95']:.3f} | {stats['p99']:.3f} | {stats['max']:.3f} |")
    lines += ["", "Warmup is outside all measurement windows. All requests have unique first tokens across warmup/repeats/probes; shapes, lengths, arrival distribution and sampling configuration are kept fixed. Exact content varies across repeats to prevent prefix reuse, not as a quality test. Actual reuse is observed at the original BlockManager.allocate call. Initial hits and reallocation hits are separately recorded.",
              "", "See each repeat's report.md, requests.json/csv, tokens.csv and steps.json/csv. Root manifest includes source snapshot hashes, model revision, compiler cache inventory, warmup coverage and final cleanup status. Root workload.json saves every repeat; warmup/ contains untimed records. Optional diagnostics are separate from ordinary trials."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
