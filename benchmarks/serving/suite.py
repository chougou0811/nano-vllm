"""Serial, isolated-cache framework validation; not a performance comparison."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .report import cache_inventory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "model-revision", "model-manifest", "output-dir", "inductor-cache", "triton-cache"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--groups", nargs="+", choices=["A", "B", "C", "D"], default=["A", "B", "C", "D"])
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    seed = root / "cache-seed"
    for name, source in [("inductor", args.inductor_cache), ("triton", args.triton_cache)]:
        shutil.copytree(source, seed / name)
    write_json(root / "cache-seed.json", {name: cache_inventory(seed / name) for name in ["inductor", "triton"]})
    results = []
    for group in args.groups:
        caches = root / "caches" / group
        shutil.copytree(seed, caches)
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(caches / "inductor"),
                   TRITON_CACHE_DIR=str(caches / "triton"), PYTHONUNBUFFERED="1")
        env.pop("TORCH_LOGS", None)
        if group == "D":
            env["TORCH_LOGS"] = "recompiles,dynamo,inductor"
        argv = [sys.executable, "-u", "-m", "benchmarks.serving",
                "--model", args.model, "--model-revision", args.model_revision,
                "--model-manifest", args.model_manifest, "--output-dir", str(root / group),
                "--tensor-parallel-size", "2", "--dtype", "bfloat16", "--enforce-eager",
                "--max-model-len", "2048", "--max-num-batched-tokens", "2048", "--max-num-seqs", "4",
                "--gpu-memory-utilization", "0.85", "--num-requests", "12", "--prompt-length", "128",
                "--output-length", "32", "--concurrency", "4", "--arrival-interval-ms", "100",
                "--warmup-requests", "4", "--verify-observer", "--repeats", "3",
                "--warmup-mode", "legacy" if group in ["A", "D"] else "decode-shapes",
                "--arrival-mode", "open-loop" if group == "C" else "concurrency-gated",
                "--ttft-slo-ms", "5000", "--tpot-slo-ms", "500"]
        if group == "C":
            argv += ["--burst-probe", "--prefix-probe"]
        if group == "D":
            argv += ["--diagnostics"]
        print(f"Starting group {group}; log: {root / (group + '.log')}", flush=True)
        with (root / (group + ".log")).open("w") as stream:
            try:
                completed = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
        results.append(dict(group=group, command=argv, returncode=returncode))
        write_json(root / "suite-execution.json", results)
        print(f"Group {group} exit: {returncode}", flush=True)
        if returncode:
            return returncode
    reports = {group: json.loads((root / group / "summary.json").read_text()) for group in args.groups}
    write_json(root / "summary.json", reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
