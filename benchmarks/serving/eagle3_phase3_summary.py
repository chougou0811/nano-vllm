"""Retain all repeats and identify cross-shape disagreements for explicit audit."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--raw',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); raw=Path(args.raw); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(p.read_text()) for p in sorted((raw/'experiment').glob('trial-*.json'))]
    grouped=defaultdict(list); index={}
    for r in records:
        choices=[s['requested_k'] for s in r['steps'] if s['controller'] and s['controller']['reason']=='measured-rate']
        runs=[k for i,k in enumerate(choices) if i==0 or k!=choices[i-1]]
        r['stability_from_trace']=dict(exploitation_run_choices=runs,
            run_returns=sum(a==c for a,b,c in zip(runs,runs[1:],runs[2:])),
            exploitation_distribution=dict(Counter(choices)))
        p=r['plan']; grouped[(p['family'],p['length'],str(p['policy']))].append(r)
        index[p['family'],p['length'],p['repeat'],str(p['policy'])]=r
    groups=[]; disagreements=[]; fixed_determinism=[]
    for (family,length,policy),rows in grouped.items():
        metrics=[r['metrics'] for r in rows]
        by_k=defaultdict(list)
        for r in rows:
            for i,s in enumerate(r['steps']):
                by_k[s['actual_k']].append(dict(proposal_ms=s['proposal_latency_ns']/1e6,
                    verification_ms=s['verification_latency_ns']/1e6,accepted=s['accepted'],outputs=len(s['committed_tokens']),
                    context=s['context_length'],initial_prefix=i==0,terminal=i==len(r['steps'])-1,
                    clipped=s['actual_k']!=s['requested_k']))
        per_k={k:dict(steps=len(samples),proposal_median_ms=statistics.median(s['proposal_ms'] for s in samples),
            verification_median_ms=statistics.median(s['verification_ms'] for s in samples),
            accepted_per_verification=statistics.mean(s['accepted'] for s in samples),
            outputs_per_verification=statistics.mean(s['outputs'] for s in samples),
            initial_prefix_steps=sum(s['initial_prefix'] for s in samples),
            terminal_steps=sum(s['terminal'] for s in samples),clipped_steps=sum(s['clipped'] for s in samples),
            context_min=min(s['context'] for s in samples),context_max=max(s['context'] for s in samples)) for k,samples in by_k.items()}
        scalar=[k for k,v in metrics[0].items() if isinstance(v,(int,float))]
        chosen=Counter(); actual=Counter(); reasons=Counter()
        for m in metrics:
            chosen.update(m['chosen_k']); actual.update(m['actual_k']); reasons.update(m['reasons'])
        groups.append(dict(family=family,length=length,policy=policy,repeats=len(rows),
            median={k:statistics.median(m[k] for m in metrics) for k in scalar},
            min={k:min(m[k] for m in metrics) for k in scalar},max={k:max(m[k] for m in metrics) for k in scalar},
            chosen_k=dict(chosen),actual_k=dict(actual),reasons=dict(reasons),per_k_observations=per_k,
            samples=[dict(case=r['case'],metrics=r['metrics'],stability=r['stability_from_trace']) for r in rows]))
        if policy!='adaptive':
            fixed_determinism.append(dict(family=family,length=length,policy=policy,
                outputs_equal=all(r['token_ids']==rows[0]['token_ids'] for r in rows),
                proposals_equal=all([s['proposed_tokens'] for s in r['steps']]==[s['proposed_tokens'] for s in rows[0]['steps']] for r in rows)))
        for row in rows:
            base=index[family,length,row['plan']['repeat'],'3']
            if row['token_ids']!=base['token_ids']:
                i=next(i for i,(a,b) in enumerate(zip(row['token_ids'],base['token_ids'])) if a!=b)
                disagreements.append(dict(case=row['case'],reference=base['case'],first_output_index=i,
                    actual_token=row['token_ids'][i],reference_token=base['token_ids'][i],
                    status='requires_state_operator_high_precision_diagnostics'))
    summary=dict(checkpoint='0ce2c1e',trials=len(records),groups=groups,disagreements=disagreements,
                 fixed_determinism=fixed_determinism,
                 cleanup=all(r['kv_released'] and r['spec_state_released'] and r['draft_state_released'] for r in records))
    plan=json.loads((raw/'experiment/plan.json').read_text())['plans']
    key=lambda p:(p['family'],p['length'],p['repeat'],str(p['policy']))
    summary['plan_complete']=Counter(key(p) for p in plan)==Counter(key(r['plan']) for r in records)
    summary['correctness_checks']=json.loads((raw/'correctness-final/summary.json').read_text())['checks']
    summary['historical_coverage_gap']='correctness/ retained: all transaction checks passed, natural samples did not cover accept=6. Test-only causal oracle added in correctness-final/; no controller/state change.'
    summary['uniform_policy_totals']={policy:dict(
        requests=sum(len(g['samples']) for g in groups if g['policy']==policy),
        e2e_ms=sum(r['metrics']['e2e_ms'] for g in groups if g['policy']==policy for r in g['samples']))
        for policy in ('1','2','3','4','5','6','adaptive')}
    summary['maximum_verification_step_ms']=max(s['verification_latency_ns']/1e6 for r in records for s in r['steps'])
    summary['post_request_allocated_bytes_range']=[min(r['allocated_bytes'] for r in records),max(r['allocated_bytes'] for r in records)]
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Initial Phase 3 Trial Table','',
        '| Workload | Context | Policy | Acceptance | Accepted/verify | Output/verify | Proposal ms | Verify ms | TPOT ms | E2E ms | tok/s | Target/draft forwards |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for g in sorted(groups,key=lambda g:(g['family'],g['length'],g['policy'])):
        m=g['median']; lines.append(f"| {g['family']} | {g['length']} | {g['policy']} | {m['acceptance_rate']:.3f} | {m['accepted_per_verification']:.2f} | {m['effective_outputs_per_verification']:.2f} | {m['proposal_ms']:.2f} | {m['verification_ms']:.2f} | {m['tpot_ms']:.2f} | {m['e2e_ms']:.2f} | {m['output_tokens_s']:.2f} | {m['target_forwards']:g}/{m['draft_forwards']:g} |")
    (out/'trial-table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(trials=len(records),disagreements=len(disagreements),cleanup=summary['cleanup'],
                         fixed_determinism=all(r['outputs_equal'] and r['proposals_equal'] for r in fixed_determinism))))
    if not summary['plan_complete'] or not summary['cleanup'] or disagreements or not all(r['outputs_equal'] and r['proposals_equal'] for r in fixed_determinism):
        raise RuntimeError('Incomplete or disagreeing trials: diagnostics required; all records retained')


if __name__=='__main__': main()
