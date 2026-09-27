"""Summarize the complete preregistered grid without excluding samples."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); raw=Path(args.raw); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    read=lambda path:json.loads(path.read_text())
    records=[read(path) for path in sorted((raw/'heldout').glob('trial-*.json'))]
    planned=read(raw/'heldout/plan.json')['plans']
    key=lambda plan:(plan['family'],plan['context'],plan['output'],plan['repeat'],str(plan['policy']))
    index={key(r['plan']):r for r in records}
    grouped=defaultdict(list)
    for r in records:
        f,n,h,rep,k=key(r['plan']); grouped[f,n,h,k].append(r)
    groups=[]; comparisons=[]; disagreements=[]; deterministic=[]
    for (f,n,h,k),rows in sorted(grouped.items()):
        ms=[r['metrics'] for r in rows]
        scalars=[name for name,value in ms[0].items() if isinstance(value,(int,float))]
        chosen=Counter(); actual=Counter()
        for m in ms: chosen.update(m['chosen_k']); actual.update(m['actual_k'])
        groups.append(dict(family=f,context=n,output=h,policy=k,repeats=len(rows),
            median={name:statistics.median(m[name] for m in ms) for name in scalars},
            min={name:min(m[name] for m in ms) for name in scalars},max={name:max(m[name] for m in ms) for name in scalars},
            chosen_k=dict(chosen),actual_k=dict(actual),samples=[dict(case=r['case'],metrics=r['metrics']) for r in rows]))
        if k!='adaptive':
            deterministic.append(dict(cell=[f,n,h,k],equal=all(r['token_ids']==rows[0]['token_ids'] and
                [s['proposed_tokens'] for s in r['steps']]==[s['proposed_tokens'] for s in rows[0]['steps']] for r in rows)))
        for r in rows:
            base=index[f,n,h,r['plan']['repeat'],'3']
            if r['token_ids']!=base['token_ids']:
                first=next(i for i,(a,b) in enumerate(zip(r['token_ids'],base['token_ids'])) if a!=b)
                disagreements.append(dict(case=r['case'],reference=base['case'],first_output_index=first,
                    actual=r['token_ids'][first],expected=base['token_ids'][first],status='requires_diagnostics'))
    for f,n,h in sorted(set((g['family'],g['context'],g['output']) for g in groups)):
        cell=[g for g in groups if (g['family'],g['context'],g['output'])==(f,n,h)]
        best=min((g for g in cell if g['policy']!='adaptive'),key=lambda g:g['median']['e2e_ms'])
        adaptive=next(g for g in cell if g['policy']=='adaptive')
        base=next(g for g in cell if g['policy']=='3')
        comparisons.append(dict(family=f,context=n,output=h,best_fixed=best['policy'],
            best_fixed_e2e_ms=best['median']['e2e_ms'],adaptive_e2e_ms=adaptive['median']['e2e_ms'],
            relative_to_best=adaptive['median']['e2e_ms']/best['median']['e2e_ms']-1,
            relative_to_k3=adaptive['median']['e2e_ms']/base['median']['e2e_ms']-1))
    prior=read(raw/'calibration/prior.json'); frozen=read(raw/'heldout/frozen-prior.json')
    manifests={stage:read(raw/stage/'manifest.json') for stage in ('calibration','correctness','heldout')}
    frozen_equal=all(m['frozen_hashes']==manifests['calibration']['frozen_hashes'] for m in manifests.values())
    controller_path='nanovllm/speculative/prior_controller.py'
    controller_hashes={stage:read(raw/stage/'source.json')[controller_path]['sha256'] for stage in manifests}
    summary=dict(trials=len(records),plan_complete=Counter(map(key,planned))==Counter(key(r['plan']) for r in records),
        output_tokens=sum(len(r['token_ids']) for r in records),groups=groups,comparisons=comparisons,
        disagreements=disagreements,fixed_determinism=deterministic,prior_unchanged=prior==frozen,
        prior_hash=hashlib.sha256(json.dumps(prior,sort_keys=True).encode()).hexdigest(),
        offline_calibration_e2e_ms=prior['offline_calibration_e2e_ms'],
        cleanup=all(r['kv_released'] and r['spec_state_released'] and r['draft_state_released'] for r in records),
        allocated_range=[min(r['allocated_bytes'] for r in records),max(r['allocated_bytes'] for r in records)],
        max_verification_step_ms=max(s['verification_latency_ns']/1e6 for r in records for s in r['steps']),
        correctness=read(raw/'correctness/summary.json')['checks'])
    summary['frozen_hashes_equal']=frozen_equal
    summary['frozen_hashes']=manifests['heldout']['frozen_hashes']
    summary['controller_hashes']=controller_hashes
    summary['controller_frozen']=len(set(controller_hashes.values()))==1
    summary['correctness_pass']=all(v for check in summary['correctness'] for v in check.values() if isinstance(v,bool))
    summary['adaptive_decisions']=[dict(case=r['case'],epochs=[dict(context=s['context_length'],
        decision=s['controller']) for s in r['steps'] if s['controller'].get('reason')=='prior-posterior-epoch'])
        for r in records if str(r['plan']['policy'])=='adaptive']
    summary['horizon_totals']={str(h):{k:dict(
        e2e_ms=sum(r['metrics']['e2e_ms'] for r in records if r['plan']['output']==h and str(r['plan']['policy'])==k),
        api_e2e_ms=sum(r['metrics']['api_e2e_ms'] for r in records if r['plan']['output']==h and str(r['plan']['policy'])==k),
        output_tokens=sum(len(r['token_ids']) for r in records if r['plan']['output']==h and str(r['plan']['policy'])==k))
        for k in ('1','2','3','4','5','6','adaptive')} for h in (64,256,512)}
    summary['policy_totals']={k:dict(requests=sum(str(r['plan']['policy'])==k for r in records),
        e2e_ms=sum(r['metrics']['e2e_ms'] for r in records if str(r['plan']['policy'])==k),
        api_e2e_ms=sum(r['metrics']['api_e2e_ms'] for r in records if str(r['plan']['policy'])==k),
        output_tokens=sum(len(r['token_ids']) for r in records if str(r['plan']['policy'])==k)) for k in ('1','2','3','4','5','6','adaptive')}
    summary['largest_steps']={name:dict(zip(('latency_ms','case','step'),max(
        (s[name]/1e6,r['case'],i) for r in records for i,s in enumerate(r['steps']))))
        for name in ('proposal_latency_ns','verification_latency_ns','conditioning_latency_ns')}
    model_checks=defaultdict(list)
    for r in records:
        if str(r['plan']['policy'])!='adaptive':
            continue
        for step,next_step in zip(r['steps'][1:],r['steps'][2:]):
            k=step['actual_k']
            if k!=step['requested_k'] or not next_step['actual_k']:
                continue
            forecast=step['controller']['forecasts'][str(k)]
            predicted_ns=sum(forecast[name] for name in ('proposal_work_ns','verification_ns','next_catchup_ns'))
            observed_ns=step['proposal_work_latency_ns']+step['verification_latency_ns']+next_step['conditioning_latency_ns']
            model_checks[r['plan']['family'],r['plan']['context'],k].append((
                forecast['outputs'],len(step['committed_tokens']),predicted_ns,observed_ns))
    summary['selected_action_model_checks']=[dict(family=f,context=n,k=k,steps=len(rows),
        predicted_outputs=sum(row[0] for row in rows),observed_outputs=sum(row[1] for row in rows),
        predicted_cost_ms=sum(row[2] for row in rows)/1e6,observed_cost_ms=sum(row[3] for row in rows)/1e6)
        for (f,n,k),rows in sorted(model_checks.items())]
    adaptive=[r for r in records if str(r['plan']['policy'])=='adaptive']
    short=[r for r in adaptive if r['plan']['output']==64]
    signature=lambda r:[(s['proposed_tokens'],s['target_ids'],s['accepted'],s['committed_tokens']) for s in r['steps']]
    summary['short_k4_execution_parity']=all(signature(r)==signature(index[key(r['plan'])[:4]+('4',)]) for r in short)
    counts=Counter()
    for r in adaptive: counts.update(r['metrics']['chosen_k'])
    summary['adaptive_stability']=dict(chosen_k=dict(counts),
        switches=sum(r['metrics']['switches'] for r in adaptive),
        run_returns=sum(r['metrics']['run_returns'] for r in adaptive),
        max_switches=max(r['metrics']['switches'] for r in adaptive),
        selection_ms=sum(r['metrics']['controller_selection_ns'] for r in adaptive)/1e6,
        short_selection_ms=sum(r['metrics']['controller_selection_ns'] for r in short)/1e6,
        online_calibration_steps=sum(r['metrics']['online_calibration_steps'] for r in adaptive),
        online_probe_steps=sum(r['metrics']['online_probe_steps'] for r in adaptive))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'prior.json').write_text(json.dumps(prior,indent=2))
    lines=['# Held-Out Phase 3.1 Trials','',
        '| Family | Context | Output | Policy | Accept | Accepted/verify | TPOT ms | E2E ms | tok/s | Proposal ms | Verify ms | Catch-up ms | Switches | Returns |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        m=g['median']; lines.append(f"| {g['family']} | {g['context']} | {g['output']} | {g['policy']} | {m['acceptance_rate']:.3f} | {m['accepted_per_verification']:.2f} | {m['tpot_ms']:.2f} | {m['e2e_ms']:.2f} | {m['output_tokens_s']:.2f} | {m['proposal_ms']:.2f} | {m['verification_ms']:.2f} | {m['catchup_ms']:.2f} | {m['switches']:g} | {m['run_returns']:g} |")
    (out/'trial-table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('trials','output_tokens','plan_complete','prior_unchanged','cleanup')}))
    print('disagreements',len(disagreements))
    gates=('plan_complete','prior_unchanged','frozen_hashes_equal','controller_frozen','correctness_pass','cleanup')
    if not all(summary[g] for g in gates) or disagreements or not all(d['equal'] for d in deterministic):
        raise RuntimeError('Unresolved audit gate; all artifacts retained')


if __name__=='__main__': main()
