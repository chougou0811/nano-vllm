"""Summarize all Phase 2 samples, keeping audit and timing data separate."""
import argparse
import json
from pathlib import Path
import statistics


def metrics(record):
    steps=record['steps']; ts=record['token_times_ns']
    ds=[s['draft_state'] for s in steps if s['draft_state']]
    return dict(proposal_ms=sum(s['proposal_latency_ns'] for s in steps)/1e6,
        conditioning_ms=sum(s['conditioning_ns'] for s in ds)/1e6,
        verification_ms=sum(s['verification_latency_ns'] for s in steps)/1e6,
        e2e_ms=(ts[-1]-record['arrival_ns'])/1e6,
        ttft_ms=(ts[0]-record['arrival_ns'])/1e6,
        tpot_ms=(ts[-1]-ts[0])/max(1,len(ts)-1)/1e6,
        output_tokens_s=len(ts)*1e9/(ts[-1]-record['arrival_ns']),
        processed=sum(s['draft_tokens_processed'] for s in ds),
        reused=sum(s['reused_draft_prefix_tokens'] for s in ds),
        rollback=sum(s['rollback_tokens'] for s in ds),
        accepted=sum(s['accepted'] for s in steps),proposed=sum(len(s['proposed_tokens']) for s in steps),
        accepted_per_verification=sum(s['accepted'] for s in steps)/len(steps),
        effective_outputs_per_verification=(len(ts)-1)/len(steps),
        draft_peak_kv_bytes=max((s['draft_kv_bytes'] for s in ds),default=0),
        draft_peak_kv_rows=max((s['draft_kv_rows'] for s in ds),default=0),
        target_forwards=record['target_forward_count'],draft_forwards=record['speculator_forward_count'],
        gpu_peak=record['gpu_peak'],gpu_allocated=record['gpu_allocated'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--raw',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); raw=Path(args.raw); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    initial=json.loads((raw/'initial/summary.json').read_text())
    follow=json.loads((raw/'followup/summary.json').read_text())
    diagnostics=json.loads((raw/'followup/draft-diagnostics.json').read_text())
    timing={}
    for length in (255,1024):
        for mode in ('full_rebuild','persistent'):
            samples=[metrics(json.loads(p.read_text())) for p in sorted((raw/'followup').glob(f'timing-{length}-*-{mode}.json'))]
            timing[f'{length}-{mode}']=dict(samples=samples,median={key:statistics.median(s[key] for s in samples) for key in samples[0]})
    summary=dict(checkpoint='087cc27',raw=str(raw),initial_checks=initial['comparisons'],followup_checks=follow,
        diagnostic_summary=dict(count=len(diagnostics),all_proposals_equal=all(d['proposal_equal'] for d in diagnostics),
            finite=all(d['finite'] for d in diagnostics),
            kv_max_abs=max(max(d['kv_max_abs']) for d in diagnostics),
            fc_bf16_max_abs=max(d['fc_shape_max_abs'] for d in diagnostics),
            fc_fp32_max_abs=max(d['fc_fp32_shape_max_abs'] for d in diagnostics)),timing=timing)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['diagnostic_summary'],indent=2))
    for label,group in timing.items(): print(label,json.dumps(group['median']))


if __name__=='__main__': main()
