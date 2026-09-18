from dataclasses import dataclass, field
from time import perf_counter_ns, sleep


@dataclass
class RequestRecord:
    request_id: int
    prompt_length: int
    requested_output_length: int
    arrival_ns: int
    sequence_id: int | None = None
    admitted_ns: int | None = None
    enqueued_ns: int | None = None
    first_scheduled_ns: int | None = None
    first_token_ns: int | None = None
    finish_ns: int | None = None
    token_times_ns: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    scheduled_times_ns: list[int] = field(default_factory=list)
    num_schedule_steps: int = 0
    num_prefill_steps: int = 0
    num_decode_steps: int = 0
    admission_step: int | None = None
    status: str = "pending"
    finish_reason: str | None = None
    error: str | None = None
    initial_prefix_cache_blocks: int = 0
    prefix_cache_blocks: int = 0
    num_kv_allocations: int = 0


@dataclass
class RunResult:
    records: list[RequestRecord]
    steps: list[dict] = field(default_factory=list)
    window_end_ns: int = 0
    complete: bool = False
    error: str | None = None
    observer_bookkeeping_ns: int = 0
    observer_calls: int = 0
    arrival_mode: str = "concurrency-gated"
    origin_ns: int = 0


class ObservedBlockManager:
    def __init__(self, original, records):
        self.original = original
        self.records = records

    def __getattr__(self, name):
        return getattr(self.original, name)

    def allocate(self, seq, num_cached_blocks):
        value = self.original.allocate(seq, num_cached_blocks)
        record = self.records[seq.seq_id]
        if record.num_kv_allocations == 0:
            record.initial_prefix_cache_blocks = num_cached_blocks
        record.prefix_cache_blocks += num_cached_blocks
        record.num_kv_allocations += 1
        return value


class ObservedScheduler:
    """Delegate original scheduler calls; inspect committed CPU Sequence state."""

    def __init__(self, scheduler, result, origin_ns, clock=perf_counter_ns):
        self.original = scheduler
        self.result = result
        self.origin_ns = origin_ns
        self.clock = clock
        self.current_request = None
        self.by_sequence = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _account(self, start, call_start, call_end):
        self.result.observer_bookkeeping_ns += (call_start - start) + (self.clock() - call_end)
        self.result.observer_calls += 1

    def add(self, seq):
        start = self.clock()
        record = self.current_request
        if record is None:
            raise RuntimeError("Request must be bound before add_request")
        call_start = self.clock()
        value = self.original.add(seq)
        call_end = self.clock()
        record.sequence_id = seq.seq_id
        record.enqueued_ns = call_end - self.origin_ns
        record.status = "admitted"
        self.by_sequence[seq.seq_id] = record
        self._account(start, call_start, call_end)
        return value

    def schedule(self):
        start = self.clock()
        queue_before = self.queue_state()
        call_start = self.clock()
        value = self.original.schedule()
        call_end = self.clock()
        seqs, is_prefill = value
        timestamp = call_end - self.origin_ns
        step = dict(step_id=len(self.result.steps), scheduled_ns=timestamp,
                    schedule_start_ns=call_start - self.origin_ns,
                    is_prefill=is_prefill, request_ids=[], scheduled_tokens=[],
                    queue_before=queue_before, queue_after_schedule=self.queue_state(),
                    batch_size=len(seqs), context_lengths=[], block_table_lengths=[])
        for seq in seqs:
            record = self.by_sequence[seq.seq_id]
            if record.first_scheduled_ns is None:
                record.first_scheduled_ns = timestamp
            record.scheduled_times_ns.append(timestamp)
            record.num_schedule_steps += 1
            record.num_prefill_steps += int(is_prefill)
            record.num_decode_steps += int(not is_prefill)
            step["request_ids"].append(record.request_id)
            step["scheduled_tokens"].append(seq.num_scheduled_tokens)
            step["context_lengths"].append(seq.num_prompt_tokens + seq.num_completion_tokens)
            step["block_table_lengths"].append(len(getattr(seq, "block_table", [])))
        self.result.steps.append(step)
        self._account(start, call_start, call_end)
        return value

    def queue_state(self):
        waiting = getattr(self.original, "waiting", [])
        running = getattr(self.original, "running", [])
        return dict(waiting=len(waiting), running=len(running),
                    waiting_tokens=sum(max(0, s.num_tokens - s.num_cached_tokens) for s in waiting))

    def postprocess(self, seqs, token_ids, is_prefill):
        start = self.clock()
        before = [seq.num_completion_tokens for seq in seqs]
        call_start = self.clock()
        value = self.original.postprocess(seqs, token_ids, is_prefill)
        call_end = self.clock()
        timestamp = call_end - self.origin_ns
        self.result.steps[-1]["committed_ns"] = timestamp
        self.result.steps[-1]["queue_after_commit"] = self.queue_state()
        self.result.steps[-1]["step_latency_ms"] = (timestamp - self.result.steps[-1]["schedule_start_ns"]) / 1e6
        for seq, count in zip(seqs, before):
            record = self.by_sequence[seq.seq_id]
            after = seq.num_completion_tokens
            if count != len(record.output_token_ids) or after < count:
                raise RuntimeError("Previously committed output changed")
            # Count only tokens appended to Sequence, never sampled/proposed IDs.
            committed = seq[seq.num_prompt_tokens + count:seq.num_prompt_tokens + after]
            if len(committed) != after - count:
                raise RuntimeError("Inconsistent committed Sequence length")
            record.output_token_ids.extend(committed)
            record.token_times_ns.extend([timestamp] * len(committed))
            if committed and record.first_token_ns is None:
                record.first_token_ns = timestamp
            if seq.is_finished:
                record.finish_ns = timestamp
                record.status = "completed"
                record.finish_reason = (
                    "eos" if not seq.ignore_eos and seq.last_token == self.original.eos
                    else "length"
                )
        self._account(start, call_start, call_end)
        return value


def run_workload(engine, specs, sampling_params, concurrency, timeout_ns,
                 clock=perf_counter_ns, sleeper=sleep, arrival_mode="concurrency-gated"):
    """Logical arrivals stay fixed even while inference or admission is blocked."""
    if not specs or concurrency < 1 or timeout_ns <= 0:
        raise ValueError("Workload, concurrency and timeout must be positive")
    if arrival_mode not in ["concurrency-gated", "open-loop"]:
        raise ValueError("Unknown arrival mode")
    if len({s.request_id for s in specs}) != len(specs):
        raise ValueError("Request IDs must be unique")
    if [s.arrival_ns for s in specs] != sorted(s.arrival_ns for s in specs):
        raise ValueError("Arrival offsets must be sorted")
    if any(s.arrival_ns < 0 for s in specs):
        raise ValueError("Arrival offsets cannot be negative")
    if not engine.is_finished():
        raise ValueError("Engine must be idle before starting a measurement")
    records = [RequestRecord(s.request_id, len(s.prompt_token_ids), s.output_length,
                             s.arrival_ns) for s in specs]
    result = RunResult(records, arrival_mode=arrival_mode)
    origin = clock()
    result.origin_ns = origin
    original = engine.scheduler
    observer = ObservedScheduler(original, result, origin, clock)
    block_manager = getattr(original, "block_manager", None)
    if block_manager is not None:
        original.block_manager = ObservedBlockManager(block_manager, observer.by_sequence)
    engine.scheduler = observer
    next_request = 0
    active = set()
    try:
        while next_request < len(specs) or active:
            now = clock() - origin
            if now >= timeout_ns:
                raise TimeoutError("Measurement deadline exceeded between engine steps")
            while (next_request < len(specs) and
                   (arrival_mode == "open-loop" or len(active) < concurrency) and
                   specs[next_request].arrival_ns <= now):
                spec, record = specs[next_request], records[next_request]
                record.admitted_ns = clock() - origin
                record.admission_step = len(result.steps)
                observer.current_request = record
                try:
                    engine.add_request(spec.prompt_token_ids, sampling_params(spec))
                finally:
                    observer.current_request = None
                active.add(record.request_id)
                next_request += 1
                now = clock() - origin
            if active:
                outputs, _ = engine.step()
                result.steps[-1]["step_return_ns"] = clock() - origin
                finished = {observer.by_sequence[seq_id].request_id for seq_id, _ in outputs}
                active.difference_update(finished)
            elif next_request < len(specs):
                due = min(specs[next_request].arrival_ns, timeout_ns)
                sleeper(max(0, due - (clock() - origin)) / 1e9)
        result.window_end_ns = max(r.finish_ns for r in records)
        result.complete = True
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
        result.window_end_ns = clock() - origin
        for record in records:
            if record.status == "completed":
                continue
            if record.arrival_ns > result.window_end_ns:
                record.status = "not_arrived"
            else:
                record.status = "timeout" if isinstance(error, TimeoutError) else "failed"
                record.finish_ns = result.window_end_ns
                record.error = result.error
    finally:
        engine.scheduler = original
        if block_manager is not None:
            original.block_manager = block_manager
    return result
