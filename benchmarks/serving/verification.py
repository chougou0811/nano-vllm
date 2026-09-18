import sys

from .adapter import run_workload
from .workload import RequestSpec


def verify_observer(engine, specs, sampling_params, reset_seed):
    """Frozen burst replay; baseline trace profiling is never used for measurement."""
    if not engine.is_finished():
        raise ValueError("Verification requires an idle engine")
    probes = [RequestSpec(i, list(s.prompt_token_ids[:8]), 4, 0)
              for i, s in enumerate(specs[:2])]
    if any(len(p.prompt_token_ids) + p.output_length >= engine.scheduler.block_manager.block_size
           for p in probes):
        raise ValueError("Verification probes must not create full prefix-cache blocks")
    # The original block size is >=256; these <=12-token probes cannot populate it.
    reset_seed()
    by_sequence = {}
    for probe in probes:
        engine.add_request(probe.prompt_token_ids, sampling_params(probe))
        by_sequence[engine.scheduler.waiting[-1].seq_id] = probe.request_id
    schedule_code = engine.scheduler.schedule.__func__.__code__
    trace = []

    def observe(frame, event, value):
        if frame.f_code is schedule_code and event == "return" and value is not None:
            seqs, prefill = value
            trace.append(dict(is_prefill=prefill,
                              request_ids=[by_sequence[s.seq_id] for s in seqs],
                              scheduled_tokens=[s.num_scheduled_tokens for s in seqs]))

    previous = sys.getprofile()
    outputs = {}
    sys.setprofile(observe)
    try:
        while not engine.is_finished():
            completed, _ = engine.step()
            outputs.update({by_sequence[sid]: ids for sid, ids in completed})
    finally:
        sys.setprofile(previous)
    reset_seed()
    observed = run_workload(engine, probes, sampling_params, len(probes), 300_000_000_000)
    if not observed.complete:
        raise RuntimeError(observed.error)
    observed_trace = [{key: step[key] for key in ["is_prefill", "request_ids", "scheduled_tokens"]}
                      for step in observed.steps]
    equal_tokens = outputs == {r.request_id: r.output_token_ids for r in observed.records}
    equal_schedule = trace == observed_trace
    released = not engine.scheduler.block_manager.used_block_ids
    verification = dict(requests=len(probes), schedule_equal=equal_schedule,
                        output_tokens_equal=equal_tokens, kv_released=released,
                        original_trace=trace, observed_trace=observed_trace,
                        original_outputs=outputs,
                        observer_bookkeeping_ns=observed.observer_bookkeeping_ns)
    if not (equal_tokens and equal_schedule and released):
        raise RuntimeError(f"Observer verification failed: {verification}")
    return verification
