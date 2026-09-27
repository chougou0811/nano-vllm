"""Offline Phase5.3 summary, preserving incomplete/failed experiments."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmarks.serving.eagle3_phase53 import ROOT


OUT = Path('benchmarks/eagle3-phase5_3')


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def union_duration(events):
    intervals = sorted((e['ts'],e['ts']+e['dur']) for e in events)
    result, end = 0, -math.inf
    for start, stop in intervals:
        result += max(0,stop-max(start,end))
        end = max(end,stop)
    return result


def main():
    isolated = json.loads((ROOT/'isolated/manifest.json').read_text())
    groups = []
    for group in range(3):
        cells = [r for r in isolated['groups'] if r['group']==group]
        trials = [t for r in cells for t in r['trials']]
        wall = {mode:sum(t['metrics']['generation_ns'] for t in trials if t['mode']==mode)/1e6
                for mode in ('serial','batch')}
        batch = [t['metrics'] for t in trials if t['mode']=='batch']
        groups.append(dict(batch_size=group+2,paired_repeats=len(batch),wall_ms=wall,
            reduction=1-wall['batch']/wall['serial'],
            median_ms={m:statistics.median(t['metrics']['generation_ns']/1e6 for t in trials if t['mode']==m)
                       for m in ('serial','batch')},
            breakdown_mean_ms={k:statistics.mean(t[k]/1e6 for t in batch) for k in
                ('packing_ns','mask_position_ns','gather_ns','model_submit_ns','head_read_ns','publish_ns')},
            scratch_bytes=max(t['scratch_bytes'] for t in batch),
            baseline_feedback_forwards=(group+2)*2,batched_feedback_forwards=2,
            baseline_total_draft_forwards=(group+2)*3,batched_total_draft_forwards=(group+2)+2,
            peak_allocated={m:max(t['peak_allocated'] for t in trials if t['mode']==m) for m in ('serial','batch')}))
    profiles = []
    for path in sorted((ROOT/'isolated').glob('*.trace.json')):
        events = json.loads(path.read_text())['traceEvents']
        kernels = [e for e in events if e.get('cat')=='kernel']
        runtime = [e for e in events if e.get('cat')=='cuda_runtime']
        profiles.append(dict(path=str(path),kernels=len(kernels),gpu_kernel_union_ms=union_duration(kernels)/1000,
            gpu_kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,
            cuda_api_union_ms=union_duration(runtime)/1000))
    manifests = []
    aggregate = defaultdict(list)
    cells_out = []
    for path in sorted(ROOT.glob('serving-*/manifest.json')):
        if path.parent.name.startswith('serving-audit'):
            continue
        d = json.loads(path.read_text())
        manifests.append(dict(path=str(path),normal_exit=d['normal_exit'],error=d.get('error'),
            trials=len(d['trials']),checks=d['checks']))
        by_cell = defaultdict(dict)
        for t in d['trials']:
            by_cell[t['family'],t['c'],t['repeat']][t['mode']] = t
        for (family,c,repeat), pair in by_cell.items():
            if len(pair)!=2:
                continue
            a,b = pair['serial'],pair['batch']
            ratio = a['summary']['duration_s']/b['summary']['duration_s']
            aggregate[path.parent.name,c].append(ratio)
            row = dict(process=path.parent.name,family=family,c=c,repeat=repeat,speedup=ratio)
            for mode,t in pair.items():
                batches = [v for s in t['steps'] for v in s.get('draft_batches',[])]
                forwards = [n for v in batches for n in v['forward_batch_sizes']]
                row[mode] = dict(summary=t['summary'],rank_memory=t['rank_memory'],
                    batched_draft_forwards=sum(v['batched_draft_forwards'] for v in batches),
                    feedback_forward_batch_sizes=dict(Counter(forwards)),
                    effective_feedback_batch_size=statistics.mean(forwards) if forwards else None,
                    serial_request_calls=sum(s.get('serial_fallback_count',0) for s in t['steps']),
                    draft_runtime_ns={k:sum(v.get(k,0) for v in batches) for k in
                        ('generation_ns','conditioning_ns','packing_ns','mask_position_ns',
                         'model_submit_ns','head_read_ns','gather_ns','publish_ns')},
                    max_scratch_bytes=max((v['scratch_bytes'] for v in batches),default=0))
            cells_out.append(row)
    baseline=json.loads((ROOT/'baseline.json').read_text())
    changed=[p for p,h in baseline['hashes'].items() if not Path(p).exists() or
             hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    correctness = {p.parent.name:json.loads(p.read_text()) for p in ROOT.glob('correctness*/manifest.json')}
    edges = json.loads((ROOT/'edges/summary.json').read_text()) if (ROOT/'edges/summary.json').exists() else None
    summary = dict(phase='5.3',raw_root=str(ROOT),isolated=groups,profiles=profiles,
        isolated_correctness=[r['correctness'] for r in isolated['groups']],fp32=isolated.get('fp32_controls'),
        isolated_fixed_shape_deterministic=all(len({json.dumps(t['tokens']) for t in r['trials']
            if t['mode']==m})==1 for r in isolated['groups'] for m in ('serial','batch')),
        serving_manifests=manifests,serving_pairs=cells_out,
        serving_aggregate=[dict(process=p,c=c,pairs=len(v),speedup_geomean=geomean(v),
                                improving_pairs=sum(x>1 for x in v)) for (p,c),v in aggregate.items()],
        correctness=[dict(process=k,normal_exit=v['normal_exit'],error=v.get('error'),checks=v['checks'])
                     for k,v in correctness.items()],edges=edges,baseline_changed=changed,
        default_changed=False,mlp_replay=False,git_commit_created=False)
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(isolated=groups,serving=summary['serving_aggregate'],
                         manifests=manifests,changed=changed),indent=2))


if __name__=='__main__':
    main()
