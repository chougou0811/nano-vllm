from .adapter import run_workload


def warmup_plan(factory, args):
    if args.warmup_mode == "legacy":
        if not args.warmup_requests:
            return []
        return [("legacy", factory.build(args.warmup_requests, args.prompt_length,
                min(8, args.output_length), 0, args.seed + 10000))]
    # Run the full requested decode span, including its block-table-width changes.
    return [(f"decode-batch-{batch}", factory.build(batch, args.prompt_length,
            max(2, args.output_length), 0, args.seed + 10000 + batch))
            for batch in range(1, args.max_num_seqs + 1)]


def execute_warmup(engine, specs, sampling, timeout_ns):
    result = run_workload(engine, specs, sampling, len(specs), timeout_ns)
    batches = sorted({st["batch_size"] for st in result.steps if not st["is_prefill"]})
    return result, dict(complete=result.complete, decode_batches=batches,
                        kv_released=not engine.scheduler.block_manager.used_block_ids,
                        prefix_cache_blocks=sum(r.prefix_cache_blocks for r in result.records))
