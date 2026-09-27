"""Fresh-process NCCL capture feasibility probe, not a performance benchmark."""
import argparse
from datetime import timedelta
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, output, port):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    x = torch.empty((4, 5120), device="cuda", dtype=torch.bfloat16)
    gathered = [torch.empty_like(x) for _ in range(2)] if rank == 0 else None
    stream = torch.cuda.Stream()

    def work():
        dist.all_reduce(x)
        dist.gather(x, gathered, dst=0)

    rows = []
    x.fill_(rank + 1)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            x.fill_(rank + 1)
            work()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        work()
    for index in range(20):
        x.fill_(rank + 1)
        mode = "eager" if index % 4 == 2 else "graph"
        graph.replay() if mode == "graph" else work()
        torch.cuda.synchronize()
        good = bool(torch.equal(x, torch.full_like(x, 3)))
        if gathered is not None:
            good = good and all(torch.equal(item, x) for item in gathered)
        rows.append(dict(index=index, mode=mode, correct=good))
        if not good:
            raise RuntimeError("NCCL capture probe output mismatch")
    graph.reset()
    del graph, gathered, x
    torch.cuda.synchronize()
    dist.barrier()
    result = dict(rank=rank, gpu=torch.cuda.current_device(), torch=torch.__version__,
                  cuda=torch.version.cuda, nccl=dist.is_nccl_available(),
                  nccl_version=torch.cuda.nccl.version(), trials=rows, passed=True)
    Path(output, f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=2334)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mp.spawn(worker, args=(str(args.output), args.port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
