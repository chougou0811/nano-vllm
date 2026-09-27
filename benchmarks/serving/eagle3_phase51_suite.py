"""Serial fresh-process serving suite; never shares a GPU engine across modes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads((args.root/"correctness/manifest.json").read_text())
    if not gate.get("correctness_passed") or not gate.get("normal_exit") or gate.get("failure"):
        raise RuntimeError("Correctness gate incomplete")
    orders = [(1, "literal"), (1, "optimized"), (2, "optimized"),
              (2, "literal"), (4, "literal"), (4, "optimized")]
    commands = []
    for concurrency, mode in orders:
        destination = args.root/f"serving-c{concurrency}-{mode}"
        command = [sys.executable, "-m", "benchmarks.serving.eagle3_phase51",
                   "--stage", "serving", "--serving-mode", mode,
                   "--concurrency", str(concurrency), "--output", str(destination)]
        commands.append(command)
        (args.root/"serving-commands.json").write_text(json.dumps(commands, indent=2))
        with (args.root/f"serving-c{concurrency}-{mode}.log").open("x") as log:
            print(json.dumps(dict(start=destination.name, command=command)), flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=os.environ.copy())
        result = json.loads((destination/"manifest.json").read_text())
        if result.get("failure") or not result["normal_exit"] or result["protected_changed"]:
            raise RuntimeError("Fresh serving process failed")
        print(json.dumps(dict(completed=destination.name, trials=len(result["trials"]))), flush=True)


if __name__ == "__main__":
    main()
