"""Independent calibration and preregistered held-out adaptive K finalization."""
import argparse
import atexit
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import traceback
from time import perf_counter_ns

import torch

from nanovllm import LLM
from nanovllm.speculative.session import generate
from benchmarks.serving.eagle3_phase3 import ReplayK, metric as old_metric


CALIBRATION = {
    'status': None,
    'recycling': 'Write a detailed 1500-word explanation of how municipal paper recycling works, including collection, sorting, pulping and quality control.',
    'digits': 'Produce 150 distinct random-looking 20-digit numeric identifiers, one per line. No explanation or headings.',
}
HELDOUT = {
    'orchard': None,
    'fermentation': 'Write at least 1800 words explaining how bread dough fermentation works. Cover yeast metabolism, bacterial activity, temperature, hydration, timing, and practical troubleshooting in six numbered sections.',
    'uuid': 'Produce 200 distinct random-looking UUID version 4 strings, one per line, using the standard 8-4-4-4-12 hexadecimal format. Do not include explanations or headings.',
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def build_prior(records):
    grouped=defaultdict(list)
    for r in records:
        for i,s in enumerate(r['steps']):
            if i and i<len(r['steps'])-1 and s['actual_k']==s['requested_k'] and s['actual_k']:
                grouped[s['context_length']//512].append(s)
    buckets={}
    for bucket,steps in grouped.items():
        successes=[0]*6; failures=[0]*6
        work=defaultdict(list); verify=defaultdict(list); catchup=defaultdict(list)
        for s in steps:
            a,k=s['accepted'],s['actual_k']
            for j in range(a): successes[j] += 1
            if a<k: failures[a] += 1
            work[k].append(s['proposal_work_latency_ns'])
            verify[k].append(s['verification_latency_ns'])
            catchup[s['conditioning_rows']].append(s['conditioning_latency_ns'])
        if set(work)!=set(range(1,7)) or not catchup:
            raise RuntimeError('Incomplete calibration coverage')
        row_sources={r:min(catchup,key=lambda n:(abs(n-r),n)) for r in range(1,8)}
        buckets[str(bucket)]=dict(acceptance_p=[(s+1)/(s+f+2) for s,f in zip(successes,failures)],
            successes=successes,failures=failures,
            proposal_work_ns=[statistics.mean(work[k]) for k in range(1,7)],
            verification_ns=[statistics.mean(verify[k]) for k in range(1,7)],
            catchup_ns=[statistics.mean(catchup[row_sources[r]]) for r in range(1,8)],
            catchup_row_sources=row_sources,cost_samples={k:len(v) for k,v in work.items()})
    return dict(schema=1,buckets=buckets,calibration_cases=[r['case'] for r in records],
        prior_strength=16,epoch_tokens=64,online_probe_steps=0,
        target_revision='40c069824f4251a91eefaf281ebe4c544efd3e18',draft_revision='3d13517724e81cb409ddf1d4650772ec52f1e18e')


def metrics(r):
    m=old_metric(r); steps=r['steps']; choices=[s['requested_k'] for s in steps]
    runs=[k for i,k in enumerate(choices) if i==0 or k!=choices[i-1]]
    m.update(run_choices=runs,run_returns=sum(a==c for a,b,c in zip(runs,runs[1:],runs[2:])),
        catchup_ms=sum(s['conditioning_latency_ns'] for s in steps[1:])/1e6,
        initial_conditioning_ms=steps[0]['conditioning_latency_ns']/1e6 if steps else 0,
        proposal_work_ms=sum(s['proposal_work_latency_ns'] for s in steps)/1e6,
        epochs=sum(bool(s['controller']) and s['controller']['reason']=='prior-posterior-epoch' for s in steps),
        online_calibration_steps=0,online_probe_steps=0)
    if 'api_duration_ns' in r:
        m['api_e2e_ms']=r['api_duration_ns']/1e6
    return m


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('calibration','correctness','heldout'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prior')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    def save(name,data): (out/name).write_text(json.dumps(data,indent=2,default=str))
    prior=json.loads(Path(args.prior).read_text()) if args.prior else None
    if args.stage!='calibration' and prior is None: raise ValueError('Frozen calibration prior required')
    config=dict(tensor_parallel_size=2,enforce_eager=True,max_model_len=2048,
        max_num_batched_tokens=2048,max_num_seqs=4,gpu_memory_utilization=.70)
    frozen_paths=['nanovllm/speculative/'+n+'.py' for n in ('draft_state','runtime','acceptance','draft','controller')]
    frozen_paths += ['nanovllm/engine/'+n+'.py' for n in ('scheduler','policy_scheduler','progress_scheduler','model_runner','block_manager')]
    source_files=list(Path('nanovllm').rglob('*.py'))+[Path(__file__)]
    save('source.json',{str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source=p.read_text()) for p in source_files})
    save('manifest.json',dict(config=config,arguments=vars(args),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi'],text=True),prior_hash=digest(prior) if prior else None,
        environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','NCCL_DEBUG','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')},
        frozen_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in frozen_paths},
        checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        calibration_questions=CALIBRATION,heldout_questions=HELDOUT))
    engine=LLM('/root/autodl-tmp/models/Qwen3-14B',**config)
    common=dict(draft_path='/root/autodl-tmp/models/Qwen3-14B_eagle3',reference_path='/root/autodl-tmp/references/eagle-pinned',
        k=3,max_tokens=96,ignore_eos=True,audit=args.stage=='correctness',draft_state_mode='persistent')
    records=[]; checks=[]
    cal_base=engine.tokenizer.encode('Date: Monday. System status: ready. The green indicator is on.\n'*260)
    test_base=engine.tokenizer.encode('The orchard gate opens at dawn, and the workers collect ripe pears.\n'*260)
    def prompt(family,length,calibration=False):
        questions,base=(CALIBRATION,cal_base) if calibration else (HELDOUT,test_base)
        question=questions[family]
        if question is None: return base[:length]
        chat=engine.tokenizer.apply_chat_template([dict(role='user',content=question)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False)
        assert len(chat)<length
        return base[:length-len(chat)]+chat
    def run(name,ids,**options):
        begin=perf_counter_ns()
        r=generate(engine,ids,**dict(common,**options))
        r.update(case=name,prompt_ids=ids,api_duration_ns=perf_counter_ns()-begin)
        r['metrics']=metrics(r); r['allocated_bytes']=torch.cuda.memory_allocated()
        for i,s in enumerate(r['steps']):
            s['catchup_caused_by_previous_k']=r['steps'][i-1]['actual_k'] if i else None
        save(name+'.json',r); records.append(r)
        assert r['kv_released'] and r['spec_state_released'] and r['draft_state_released']
        print(name,json.dumps(r['metrics']),flush=True)
        return r
    try:
        run('warmup',cal_base[:128],max_tokens=16)
        if args.stage=='calibration':
            plans=[dict(family=f,context=n,k=k) for f in CALIBRATION for n in (255,768,1024) for k in range(1,7)]
            save('plan.json',dict(calibration=plans,heldout_questions=HELDOUT,
                heldout_contexts=[255,1024],heldout_outputs=[64,256,512],repeats=3))
            for k in range(1,7): run('shape-warmup-'+str(k),cal_base[:255],k=k,max_tokens=24)
            start=len(records)
            for p in plans: run(f"cal-{p['family']}-{p['context']}-{p['k']}",prompt(p['family'],p['context'],True),k=p['k'])
            result=build_prior(records[start:]); result['source_manifest_sha256']=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
            result['offline_calibration_e2e_ms']=sum(r['metrics']['e2e_ms'] for r in records[start:])
            save('prior.json',result)
        elif args.stage=='correctness':
            for n in (255,256,257,1088):
                a=run(f'prior-{n}',test_base[:n],adaptive_k=True,adaptive_prior=prior,max_tokens=80)
                b=run(f'replay-{n}',test_base[:n],max_tokens=80,k_controller=ReplayK([s['requested_k'] for s in a['steps']]),
                    proposal_override=lambda ids,i:a['steps'][i]['proposed_tokens'])
                signature=lambda r:[(s['actual_k'],s['accepted'],s['committed_tokens'],s['target_ids'],s['ranks']) for s in r['steps']]
                checks.append(dict(case=str(n),exact=signature(a)==signature(b),outputs=a['token_ids']==b['token_ids']))
            for maximum in (1,2,7):
                r=run('max-'+str(maximum),test_base[:255],adaptive_k=True,adaptive_prior=prior,max_tokens=maximum)
                checks.append(dict(case='max-'+str(maximum),valid=len(r['token_ids'])==maximum))
            r=run('forced-reject',test_base[:255],adaptive_k=True,adaptive_prior=prior,max_tokens=80,
                proposal_override=lambda ids,i:[0]*len(ids))
            checks.append(dict(case='forced-reject',zero=all(s['accepted']==0 for s in r['steps'])))
            first=r['token_ids'][0]
            eos=run('eos',test_base[:255],adaptive_k=True,adaptive_prior=prior,ignore_eos=False,eos_override=first)
            checks.append(dict(case='eos',valid=eos['token_ids']==[first]))
            long=run('long-512',test_base[:1024],adaptive_k=True,adaptive_prior=prior,max_tokens=512)
            checks.append(dict(case='long-512',epochs=long['metrics']['epochs']<=8))
        else:
            policies=[1,2,3,4,5,6,'adaptive']
            plans=[]
            for repeat in range(args.repeats):
                for fi,family in enumerate(HELDOUT):
                    for ni,n in enumerate((255,1024)):
                        for hi,horizon in enumerate((64,256,512)):
                            shift=(repeat*2+fi+ni+hi)%7
                            order=policies[shift:]+policies[:shift]
                            plans.extend(dict(family=family,context=n,output=horizon,repeat=repeat,policy=k) for k in order)
            save('plan.json',dict(plans=plans,prior_hash=digest(prior),
                prompts={f'{f}-{n}':prompt(f,n) for f in HELDOUT for n in (255,1024)}))
            save('frozen-prior.json',prior)
            for n in (255,1024):
                for k in range(1,7): run(f'shape-warmup-{n}-{k}',cal_base[:n],k=k,max_tokens=24)
            for p in plans:
                f,n,h,rep,k=(p[key] for key in ('family','context','output','repeat','policy'))
                options=dict(k=3,adaptive_k=True,adaptive_prior=prior) if k=='adaptive' else dict(k=k)
                r=run(f'trial-{f}-{n}-{h}-{rep}-{k}',prompt(f,n),max_tokens=h,**options)
                r['plan']=p; save(r['case']+'.json',r)
        save('summary.json',dict(checks=checks,records=[dict(case=r['case'],metrics=r['metrics']) for r in records],
            cleanup=all(r['kv_released'] and r['spec_state_released'] and r['draft_state_released'] for r in records)))
        if not all(value for c in checks for key,value in c.items() if key!='case'):
            raise RuntimeError('Correctness check failed; all evidence retained')
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        atexit.unregister(engine.exit); engine.exit()


if __name__=='__main__': main()
