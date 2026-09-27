"""Fetch immutable source snapshots for an audit, never import or install them."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


EXTRA = {
    "sglang": ["python/sglang/srt/speculative/eagle_worker_v2.py",
        "python/sglang/srt/speculative/eagle_worker_common.py",
        "python/sglang/srt/speculative/eagle_info.py",
        "python/sglang/srt/distributed/device_communicators/custom_all_reduce.py",
        "python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py"],
    "vllm": ["vllm/v1/spec_decode/eagle.py",
        "vllm/distributed/device_communicators/custom_all_reduce.py"],
    "flashinfer": ["flashinfer/gemm/gemm_base.py"],
}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    old=json.loads(Path("benchmarks/eagle3-phase5_0/source-ledger.json").read_text())
    result=dict(access_utc=datetime.now(timezone.utc).isoformat(), repositories=[], files=[],
                note="Default remote HEAD at access time; selected source snapshots, not builds. Errors retained.")
    for repo in old["repositories"]:
        name=repo["project"]
        head=subprocess.check_output(["git","ls-remote",repo["repo"],"HEAD"],text=True,timeout=30).split()[0]
        result["repositories"].append(dict(project=name,repo=repo["repo"],commit=head,
            prior_commit=repo["commit"],changed=head!=repo["commit"]))
        paths={e["path"] for e in old["entries"] if e.get("project")==name and e.get("path")
               and e.get("commit")==repo["commit"]}
        paths.update(EXTRA.get(name,[]))
        paths.add("README.md")
        slug=repo["repo"].removeprefix("https://github.com/").removesuffix(".git")
        for path in sorted(paths):
            url=f"https://raw.githubusercontent.com/{slug}/{head}/{path}"
            row=dict(project=name,commit=head,path=path,url=f"https://github.com/{slug}/blob/{head}/{path}")
            try:
                content=urllib.request.urlopen(url,timeout=30).read()
                dest=a.output/name/path
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(content)
                row.update(local=str(dest),sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))
            except Exception as error:
                row["error"]=str(error)
            result["files"].append(row)
        a.ledger.parent.mkdir(parents=True,exist_ok=True)
        a.ledger.write_text(json.dumps(result,indent=2))
        print(name,head,len(paths),flush=True)


if __name__ == "__main__":
    main()
