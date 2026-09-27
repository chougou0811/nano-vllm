"""Serial GPU orchestration; abort on error, never silently discard a run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.serving.eagle3_phase44c import ROOT, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--wait-pid",type=int,required=True)
    args = parser.parse_args()
    env = os.environ | dict(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",NCCL_DEBUG="WARN",TORCH_DISABLE_ADDR2LINE="1")
    output = ROOT/"benchmarks/eagle3-phase4_4c"
    jobs = []

    def run(module, arguments, name):
        command = [sys.executable,"-m",module,*map(str,arguments)]
        with (args.raw/f"{name}.log").open("w") as log:
            start = time.time()
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append(dict(command=command,returncode=result.returncode,
                         elapsed_s=time.time()-start,log=str(args.raw/f"{name}.log")))
        dump(args.raw/"pipeline.json",jobs)
        print(json.dumps(jobs[-1]),flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; artifacts retained")

    deadline = time.monotonic()+14400
    while True:
        path = args.raw/"shadow-c1/manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest.get("error"):
                raise RuntimeError(manifest["error"])
            if manifest.get("normal_exit"):
                break
        try:
            os.kill(args.wait_pid,0)
        except ProcessLookupError:
            time.sleep(2)
            if path.exists() and json.loads(path.read_text()).get("normal_exit"):
                break
            raise RuntimeError("First shadow process exited without completion")
        if time.monotonic()>deadline:
            raise TimeoutError("First shadow run not complete")
        time.sleep(20)
    for c in (2,4):
        run("benchmarks.serving.eagle3_phase44c",["--stage","shadow","--concurrency",c,
            "--count",512,"--output",args.raw/f"shadow-c{c}"],f"shadow-c{c}")
    analyze_args = ["--raw",args.raw,"--output",output]
    run("benchmarks.serving.eagle3_phase44c_analysis",analyze_args,"analysis")
    summary = json.loads((output/"summary.json").read_text())
    assert summary["status"]=="shadow_complete"
    dump(args.raw/"bounded-selection.json",summary["conditional_bounded_validation"])
    for cell in summary["conditional_bounded_validation"]:
        c = cell["concurrency"]
        run("benchmarks.serving.eagle3_phase44c",["--stage","bounded","--concurrency",c,
            "--workload",cell["workload"],"--count",512,"--output",args.raw/f"bounded-c{c}"],f"bounded-c{c}")
    run("benchmarks.serving.eagle3_phase44c_report",analyze_args,"render")


if __name__ == "__main__":
    main()
