"""Phase 5.3 bounded experiment: real-state capture, then draft-only gate."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from types import SimpleNamespace

import torch
from transformers import AutoConfig

from benchmarks.serving.eagle3_phase42 import (
    TARGET, DRAFT, REFERENCE, Request, run_closed_loop)
from nanovllm.speculative.concurrent_engine import ConcurrentLLMEngine
from nanovllm.speculative.draft import ReferenceDraft
from nanovllm.speculative.draft_state import DraftState
from nanovllm.speculative.draft_batch import DraftBatchExecutor, DraftBatchInput


ROOT = Path('/root/autodl-tmp/eagle3-phase5.3-20260925')
GROUPS = ((255, 256), (255, 256, 257), (511, 512, 513, 1025))


def tree(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (tuple, list)):
        return tuple(tree(v, fn) for v in value)
    return value


def capture():
    out = ROOT/'capture'
    out.mkdir(exist_ok=False)
    data = dict(command=[sys.executable, *sys.argv], groups=GROUPS, trials=[], samples=[],
                gpu=subprocess.check_output(['nvidia-smi'], text=True), normal_exit=False)
    engine = ConcurrentLLMEngine(TARGET, draft_path=DRAFT, reference_path=REFERENCE,
        tensor_parallel_size=2, enforce_eager=True, max_model_len=2048,
        max_num_batched_tokens=2048, max_num_seqs=4, gpu_memory_utilization=.7,
        scheduler_policy='original', audit=True)
    original = DraftState.propose
    captured = {}
    try:
        for group, lengths in enumerate(GROUPS):
            counts = {}
            def propose(state, features, tokens, k, *, owner):
                iteration = counts.get(owner, 0)
                counts[owner] = iteration+1
                if iteration < 3:
                    captured.setdefault(owner, []).append(dict(
                        features=features.detach().cpu().clone(), tokens=tuple(tokens), k=k,
                        cursor=state.cursor, conditioned_tokens=state.conditioned_tokens,
                        past=tree(state.past, lambda t: t.detach().cpu().clone()),
                        generation=engine.speculative_coordinator.requests[owner].generation,
                        iteration=iteration, seq_id=owner))
                return original(state, features, tokens, k, owner=owner)
            DraftState.propose = propose
            phrase = engine.tokenizer.encode(
                'A scientist records measured values and explains each observation. '*400,
                add_special_tokens=False)
            requests = [Request(i, ([1000+group*7+i]+phrase)[:n], 16) for i,n in enumerate(lengths)]
            result = run_closed_loop(engine, 'speculative', requests, len(lengths))
            data['trials'].append(result)
            ids = [r.sequence_id for r in requests]
            samples = [[captured[owner][iteration] for owner in ids] for iteration in range(3)]
            path = out/f'group{group}.pt'
            torch.save(samples, path)
            data['samples'].append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('CAPTURE', group, lengths, flush=True)
            captured.clear()
        data['normal_exit'] = True
    finally:
        DraftState.propose = original
        engine.exit()
        (out/'manifest.json').write_text(json.dumps(data, indent=2))


def make_items(samples, draft, dtype=torch.bfloat16):
    items = []
    for sample in samples:
        state = DraftState(draft, sample['seq_id'])
        state.request_generation = sample['generation']
        state.cursor = sample['cursor']
        state.conditioned_tokens = sample['conditioned_tokens']
        state.past = tree(sample['past'], lambda t: t.to(device=draft.device, dtype=dtype))
        request = SimpleNamespace(seq_id=sample['seq_id'], generation=sample['generation'])
        items.append(DraftBatchInput(state, request, request.generation,
            sample['features'].to(device=draft.device, dtype=dtype), sample['tokens'], sample['k']))
    return items


@torch.inference_mode()
def serial_generation(prepared, observer=None, forced=None):
    begin = perf_counter_ns()
    result = []
    for row,p in enumerate(prepared):
        model, device = p.item.state.draft.model, p.hidden.device
        hidden, past = p.hidden, p.confirmed
        ids = []
        for step in range(p.item.k):
            logits = model.lm_head(model.norm(hidden))[:, -1]
            index = logits[0].argmax()
            token = int((index+model.d2t[index]).item())
            ids.append(token)
            if observer:
                observer(row, step, hidden, past, logits)
            if step+1 < p.item.k:
                feed = forced[row][step] if forced else token
                hidden,past = model(hidden, input_ids=torch.tensor([[feed]], device=device),
                                    past_key_values=past, use_cache=True)
        result.append(ids)
    return result, dict(generation_ns=perf_counter_ns()-begin,
                        feedback_forwards=sum(p.item.k-1 for p in prepared))


def stats(a, b):
    diff = (a.float()-b.float()).abs()
    return dict(max_abs=diff.max().item(), mean_abs=diff.mean().item(),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


@torch.inference_mode()
def compare(prepared, executor):
    serial_trace, batch_trace = {}, {}
    def serial_observer(row, step, hidden, past, logits):
        serial_trace[row, step] = (hidden.clone(), tree(past, torch.clone), logits.clone())
    def batch_observer(step, active, hidden, past, logits, positions, mask):
        for row,index in enumerate(active):
            valid = mask[row].nonzero().flatten()
            kv = tuple(tuple(t[row:row+1].index_select(2, valid) for t in layer) for layer in past)
            batch_trace[index, step] = (hidden[row:row+1].clone(), kv, logits[row:row+1].clone())
    serial, _ = serial_generation(prepared, serial_observer)
    executor.observer = batch_observer
    batch = executor.generate(prepared, publish=False)
    executor.observer = None
    rows = []
    for key, (h, kv, logits) in serial_trace.items():
        bh, bkv, blogits = batch_trace[key]
        rows.append(dict(row=key[0], step=key[1], hidden=stats(h,bh), logits=stats(logits,blogits),
            kv=[stats(a,b) for la,lb in zip(kv,bkv) for a,b in zip(la,lb)],
            serial_top1=int(logits.argmax()), batch_top1=int(blogits.argmax()),
            serial_margin=float(logits.float().topk(2).values.diff().abs().item())))
    return dict(serial=serial, batch=batch, proposals_equal=serial==batch, rows=rows)


def isolated():
    out = ROOT/'isolated'
    out.mkdir(exist_ok=False)
    torch.manual_seed(53)
    torch.set_num_threads(4)
    draft = ReferenceDraft(DRAFT, REFERENCE, TARGET,
        AutoConfig.from_pretrained(TARGET, local_files_only=True), torch.device('cuda:0'))
    executor = DraftBatchExecutor()
    data = dict(command=[sys.executable, *sys.argv], provenance=draft.provenance,
        versions=dict(torch=torch.__version__, cuda=torch.version.cuda), groups=[], normal_exit=False)
    for group in range(len(GROUPS)):
        for iteration, samples in enumerate(torch.load(ROOT/f'capture/group{group}.pt', weights_only=True)):
            items = make_items(samples, draft)
            prepared = executor.prepare(items)
            row = dict(group=group, iteration=iteration, lengths=[p.end for p in prepared],
                       cursors=[p.start for p in prepared], k=[p.item.k for p in prepared], trials=[])
            # Oracle is the untouched Phase2 method, from the same confirmed state.
            oracle_items = make_items(samples, draft)
            oracle = [i.state.propose(i.features,i.tokens,i.k,owner=i.request.seq_id) for i in oracle_items]
            check = compare(prepared, executor)
            check['serial_matches_frozen_oracle'] = check['serial'] == oracle
            check['confirmed_kv_exact'] = all(torch.equal(t,u)
                for p,i in zip(prepared,oracle_items) for la,lb in zip(p.confirmed,i.state.past)
                for t,u in zip(la,lb))
            row['correctness'] = check
            for _ in range(2):
                serial_generation(prepared)
                executor.generate(prepared, publish=False)
            for repeat in range(5):
                for mode in (('serial','batch') if repeat%2 == 0 else ('batch','serial')):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    if mode == 'serial':
                        tokens, metrics = serial_generation(prepared)
                    else:
                        tokens = executor.generate(prepared, publish=False)
                        metrics = dict(executor.last_metrics)
                    end.record()
                    torch.cuda.synchronize()
                    row['trials'].append(dict(mode=mode, repeat=repeat, metrics=metrics,
                        gpu_stream_ms=start.elapsed_time(end), tokens=tokens,
                        peak_allocated=torch.cuda.max_memory_allocated(),
                        peak_reserved=torch.cuda.max_memory_reserved()))
            data['groups'].append(row)
            (out/'manifest.json').write_text(json.dumps(data,indent=2))
            print('ISOLATED',group,iteration,check['proposals_equal'], flush=True)
            if iteration == 0:
                for mode in ('serial', 'batch'):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        if mode == 'serial':
                            serial_generation(prepared)
                        else:
                            executor.generate(prepared, publish=False)
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(out/f'group{group}-{mode}.trace.json'))
            del prepared, items, oracle_items
    # Same checkpoint, FP32 arithmetic control. No production dtype is changed.
    torch.backends.cuda.matmul.allow_tf32 = False
    draft.model.float()
    data['fp32_controls'] = []
    for group in range(len(GROUPS)):
        for iteration, samples in enumerate(torch.load(ROOT/f'capture/group{group}.pt', weights_only=True)):
            items = make_items(samples, draft, torch.float32)
            prepared = executor.prepare(items)
            data['fp32_controls'].append(dict(group=group, iteration=iteration,
                                              comparison=compare(prepared, executor)))
            del prepared, items
    data['normal_exit'] = True
    (out/'manifest.json').write_text(json.dumps(data,indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('capture','isolated'))
    args = parser.parse_args()
    (capture if args.stage=='capture' else isolated)()


if __name__ == '__main__':
    main()
