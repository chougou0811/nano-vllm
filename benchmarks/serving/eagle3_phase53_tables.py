"""Derive complete Phase5.3 serving tables without discarding any measured row."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from benchmarks.serving.eagle3_phase42 import distribution
from benchmarks.serving.eagle3_phase53 import ROOT
from benchmarks.serving.eagle3_phase53_analysis import OUT, geomean


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def pooled(trials):
    wall = sum(t['summary']['duration_s'] for t in trials)
    requests = [r for t in trials for r in t['requests']]
    steps = [s for t in trials for s in t['steps']]
    batches = [b for s in steps for b in s.get('draft_batches',[])]
    spec = {k:sum(t['summary']['speculation'][k] for t in trials) for k in
            ('proposed_tokens','accepted_tokens','verifications','target_forwards','draft_forwards',
             'committed_decode_tokens')}
    metrics = {k:sum(t['summary']['wall_clock'].get(k,0) for t in trials) for k in
               ('draft_ns','draft_catchup_ns','target_verify_ns','target_prefill_ns','schedule_ns')}
    sizes = [n for b in batches for n in b['forward_batch_sizes']]
    fallback_requests = sum(s.get('serial_fallback_count',0) for s in steps)
    fallback_forwards = spec['draft_forwards']-sum(b['draft_forwards'] for b in batches)
    fallback_feedback = fallback_forwards-fallback_requests
    groups = len(sizes)+fallback_feedback
    spec.update(acceptance_rate=spec['accepted_tokens']/spec['proposed_tokens'],
        accepted_per_verification=spec['accepted_tokens']/spec['verifications'],
        outputs_per_verification=spec['committed_decode_tokens']/spec['verifications'],
        batched_feedback_forwards=sum(n>1 for n in sizes),serial_request_calls=fallback_requests,
        generation_head_steps=sum(len(b['head_batch_sizes']) for b in batches)+fallback_forwards,
        effective_feedback_batch_size=(sum(sizes)+fallback_feedback)/groups if groups else None)
    return dict(trials=len(trials),wall_s=wall,requests=len(requests),
        output_tokens=sum(len(r['output_token_ids']) for r in requests),
        output_tokens_per_s=sum(len(r['output_token_ids']) for r in requests)/wall,
        requests_per_s=len(requests)/wall,
        latency={k:distribution(v for r in requests for v in
            (r['itl_ms'] if k=='itl_ms' else [r[k]])) for k in ('ttft_ms','e2e_ms','tpot_ms','itl_ms')},
        wall_ns=metrics,speculation=spec,
        peak_allocated_by_rank=[max(t['rank_memory'][rank]['peak_allocated'] for t in trials) for rank in (0,1)],
        peak_reserved_by_rank=[max(t['rank_memory'][rank]['peak_reserved'] for t in trials) for rank in (0,1)],
        min_posttrial_free_by_rank=[min(t['rank_memory'][rank]['free'] for t in trials) for rank in (0,1)],
        posttrial_allocated_range_by_rank=[
            [min(t['rank_memory'][rank]['allocated'] for t in trials),
             max(t['rank_memory'][rank]['allocated'] for t in trials)] for rank in (0,1)],
        max_scratch_bytes=max((b['scratch_bytes'] for b in batches),default=0),
        max_used_blocks=max(s['used_blocks_after'] for s in steps),
        cleanup_zero=all(not any(t['summary']['cleanup'].values()) and
                         not any(r['targets'] for r in t['rank_memory']) for t in trials),
        prefix_cache_hits=sum(t['summary']['prefix_cache_hit_requests'] for t in trials))


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    complete = []
    for path in sorted(ROOT.glob('serving-*/manifest.json')):
        if path.parent.name.startswith('serving-audit'):
            continue
        data = json.loads(path.read_text())
        if data['normal_exit']:
            complete.append((path.parent.name,data))
    pools, cells, throughput, latency, runtime = [], [], [], [], []
    for process,data in complete:
        grouped = defaultdict(list)
        by_cell = defaultdict(dict)
        for trial in data['trials']:
            grouped[trial['c'],trial['mode']].append(trial)
            by_cell[trial['family'],trial['c'],trial['repeat']][trial['mode']] = trial
        per_family = defaultdict(list)
        for (family,c,repeat),pair in by_cell.items():
            per_family[family,c].append(pair)
        for (family,c),pairs in per_family.items():
            ratios=[p['serial']['summary']['duration_s']/p['batch']['summary']['duration_s'] for p in pairs]
            p95={m:statistics.median(p[m]['summary']['latency']['itl_ms']['p95'] for p in pairs)
                 for m in ('serial','batch')}
            p99={m:statistics.median(p[m]['summary']['latency']['itl_ms']['p99'] for p in pairs)
                 for m in ('serial','batch')}
            row=dict(process=process,family=family,c=c,repeats=len(pairs),speedup=geomean(ratios),
                improving_repeats=sum(r>1 for r in ratios),itl_p95_median=p95,itl_p99_median=p99)
            cells.append(row)
            throughput.append([process,family,c,len(pairs),f'{row["speedup"]:.4f}',
                row['improving_repeats'],f'{p95["batch"]/p95["serial"]:.4f}',
                f'{p99["batch"]/p99["serial"]:.4f}'])
        for (c,mode),trials in grouped.items():
            p=pooled(trials)
            pools.append(dict(process=process,c=c,mode=mode,**p))
            percentiles=lambda k:'/'.join(f'{p["latency"][k][q]:.3f}' for q in ('p50','p95','p99'))
            latency.append([process,c,mode,f'{p["output_tokens_per_s"]:.3f}',f'{p["requests_per_s"]:.3f}',
                *[percentiles(k) for k in ('ttft_ms','e2e_ms','tpot_ms','itl_ms')]])
            w=p['wall_ns']
            runtime.append([process,c,mode,f'{w["draft_catchup_ns"]/1e9:.3f}',
                f'{(w["draft_ns"]-w["draft_catchup_ns"])/1e9:.3f}',
                f'{w["target_verify_ns"]/1e9:.3f}',p['speculation']['draft_forwards'],
                p['speculation']['batched_feedback_forwards'],f'{p["speculation"]["effective_feedback_batch_size"]:.3f}',
                f'{p["speculation"]["acceptance_rate"]:.4f}',
                f'{p["peak_allocated_by_rank"][0]/2**30:.3f}',f'{p["peak_reserved_by_rank"][0]/2**30:.3f}'])
    summary.update(completed_serving_pools=pools,completed_serving_cells=cells)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'throughput-table.md').write_text('# Serving Paired Results\n\nGeomean serial wall / batch wall. Tail ratios use the median of each repeat\'s percentile. All repeats included. Incomplete/error processes remain in summary.json but are not presented as completed five-repeat cells.\n\n'+table(
        ['process','family','c','repeats','speedup','improving','P95 ITL ratio','P99 ITL ratio'],throughput))
    (OUT/'latency-table.md').write_text('# Pooled Serving Latency\n\nAll request/token observations pooled within each process/c/mode. Latency columns are P50/P95/P99 in ms; throughput uses summed wall time, never summed request E2E. ITL timestamps refer to committed bursts, so intra-burst gaps can be zero.\n\n'+table(
        ['process','c','mode','output tok/s','requests/s','TTFT','E2E','TPOT','ITL'],latency))
    (OUT/'runtime-table.md').write_text('# Runtime and Memory\n\nWall columns are seconds summed over trials, not per step. Generation = total draft wall minus serial conditioning, including full serving publication/packing overhead. Allocated/reserved are rank0 peak GiB; both ranks and per-trial numbers remain in JSON.\n\n'+table(
        ['process','c','mode','catch-up s','generation s','verify s','draft forwards','batched feedback forwards',
         'effective feedback B','accepted/proposed','allocated GiB','reserved GiB'],runtime))
    print(json.dumps(dict(complete_processes=[p for p,_ in complete],cells=len(cells)),indent=2))


if __name__=='__main__':
    main()
