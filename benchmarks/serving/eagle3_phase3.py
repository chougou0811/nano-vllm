"""Adaptive K pilot, transaction replay, and balanced repeated single-request trials."""
import argparse
import atexit
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import torch

from nanovllm import LLM
from nanovllm.speculative.session import generate


QUESTIONS = {
    'repetition': None,
    'sky': 'Explain why the sky looks blue in two sentences.',
    'arithmetic': 'What is 17 times 23? Show the calculation briefly.',
    'hex': 'Write twenty distinct random-looking 16-digit hexadecimal identifiers, one per line. No explanation.',
    'code': 'Write a Python function that merges two sorted lists. Include a short example.',
    'translation': 'Translate to Chinese: The old lighthouse keeper carefully recorded the changing tides and distant ships every evening.',
}


class ReplayK:
    def __init__(self, values):
        self.values, self.index, self.last_decision = values, 0, {}
    def select(self, context, remaining):
        k=self.values[self.index % len(self.values)]
        self.index += 1
        self.last_decision=dict(reason='scripted-replay',requested_k=k)
        return k
    def observe(self, **kwargs): pass


def metric(record):
    steps=record['steps']; proposed=sum(s['actual_k'] for s in steps); accepted=sum(s['accepted'] for s in steps)
    ts=record['token_times_ns']; n=len(steps)
    choices=[s['requested_k'] for s in steps]
    exploit=[s['requested_k'] for s in steps if s['controller'] and s['controller']['reason']=='measured-rate']
    return dict(acceptance_rate=accepted/proposed if proposed else None,accepted_per_verification=accepted/n if n else None,
        effective_outputs_per_verification=(len(ts)-1)/n if n else None,
        proposed=proposed,accepted=accepted,verification_count=n,
        proposal_ms=sum(s['proposal_latency_ns'] for s in steps)/1e6,
        verification_ms=sum(s['verification_latency_ns'] for s in steps)/1e6,
        tpot_ms=(ts[-1]-ts[0])/max(1,len(ts)-1)/1e6,e2e_ms=(ts[-1]-record['arrival_ns'])/1e6,
        ttft_ms=(ts[0]-record['arrival_ns'])/1e6,output_tokens_s=len(ts)*1e9/(ts[-1]-record['arrival_ns']),
        progress_tokens_s=(len(ts)-1)*1e9/sum(s['proposal_latency_ns']+s['verification_latency_ns'] for s in steps) if n else None,
        target_forwards=record['target_forward_count'],draft_forwards=record['speculator_forward_count'],
        chosen_k=dict(Counter(choices)),actual_k=dict(Counter(s['actual_k'] for s in steps)),
        switches=sum(a!=b for a,b in zip(choices,choices[1:])),
        exploit_switches=sum(a!=b for a,b in zip(exploit,exploit[1:])),
        exploit_reversals=sum(a==c and a!=b for a,b,c in zip(exploit,exploit[1:],exploit[2:])),
        reasons=dict(Counter(s['controller']['reason'] for s in steps if s['controller'])),
        controller_selection_ns=sum(s['controller_selection_ns'] for s in steps))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=('pilot','correctness','experiment'),required=True)
    parser.add_argument('--families',nargs='+',default=['repetition','sky','hex'])
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    def save(name,obj): (out/name).write_text(json.dumps(obj,indent=2,default=str))
    config=dict(tensor_parallel_size=2,enforce_eager=True,max_model_len=2048,max_num_batched_tokens=2048,
                max_num_seqs=4,gpu_memory_utilization=0.70)
    save('manifest.json',dict(arguments=vars(args),config=config,checkpoint='0ce2c1e',
        torch=torch.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi'],text=True),
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NCCL_DEBUG','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
            'TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')},
        target_revision='40c069824f4251a91eefaf281ebe4c544efd3e18',draft_revision='3d13517724e81cb409ddf1d4650772ec52f1e18e',
        controller=dict(candidates=list(range(1,7)),alpha=.25,probe_interval=12,switch_margin=.10,context_bucket_width=512)))
    files=list(Path('nanovllm').rglob('*.py'))+[Path(__file__)]
    save('sources.json',{str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source=p.read_text()) for p in files})
    engine=LLM('/root/autodl-tmp/models/Qwen3-14B',**config)
    common=dict(draft_path='/root/autodl-tmp/models/Qwen3-14B_eagle3',reference_path='/root/autodl-tmp/references/eagle-pinned',
                max_tokens=64,k=3,ignore_eos=True,audit=True,draft_state_mode='persistent')
    records=[]; checks=[]
    base=engine.tokenizer.encode('A scientist writes careful notes about an experiment. '*220)
    def prompt(family,length):
        if QUESTIONS[family] is None: return base[:length]
        chat=engine.tokenizer.apply_chat_template([dict(role='user',content=QUESTIONS[family])],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        assert len(chat)<length
        return base[:length-len(chat)]+chat
    def run(name,ids,**options):
        r=generate(engine,ids,**dict(common,**options))
        r['case']=name; r['metrics']=metric(r)
        r['prompt_ids']=ids
        r['allocated_bytes']=torch.cuda.memory_allocated()
        save(name+'.json',r); records.append(r)
        assert r['kv_released'] and r['spec_state_released'] and r['draft_state_released']
        print(name,json.dumps(r['metrics']),flush=True)
        return r
    try:
        run('warmup',base[:128],max_tokens=16)
        if args.stage=='pilot':
            for family in QUESTIONS:
                for length in (255,1024): run(f'{family}-{length}',prompt(family,length))
        elif args.stage=='correctness':
            oracle_calls=0
            def oracle(ids,iteration):
                nonlocal oracle_calls
                # Test-only causal fixed point: no commit until proposals match
                # this exact parallel target shape. Never used in performance runs.
                for _ in range(len(ids)+1):
                    result=engine.model_runner.call('eagle3','verify',dict(proposals=ids))
                    oracle_calls += 1
                    expected=result['target_ids'][:-1]
                    if expected==ids: return ids
                    ids=expected
                raise RuntimeError('Causal proposal oracle failed to converge')
            full=run('forced-full-accept',base[:255],max_tokens=32,
                k_controller=ReplayK([1,6,1]),proposal_override=oracle)
            save('oracle.json',dict(extra_verification_forwards=oracle_calls,performance_sample=False))
            checks.append(dict(case='forced-full-accept',full_six=any(s['actual_k']==6 and s['accepted']==6 for s in full['steps']),
                all_accepted=all(s['accepted']==s['actual_k'] for s in full['steps'])))
            def signature(r):
                return [(s['actual_k'],s['accepted'],s['committed_tokens'],s['target_ids'],s['ranks']) for s in r['steps']]
            for length in (255,256,257,1088):
                traces=[]
                def probe(draft,state,proposals,i):
                    kv=engine.model_runner.kv_cache[:,:,state['blocks']].flatten(2,3)[:,:,:state['cached']]
                    def digest(t): return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                    traces.append(dict(cursor=state['cached'],tokens=list(state['tokens']),kv=digest(kv),features=digest(state['features'])))
                old=run(f'dynamic-{length}',base[:length],max_tokens=32,k_controller=ReplayK([1,6,1]),state_probe=probe)
                original_traces=traces[:]; traces.clear()
                replay=run(f'replay-{length}',base[:length],max_tokens=32,
                    draft_state_mode='full_rebuild',k_controller=ReplayK([s['requested_k'] for s in old['steps']]),
                    proposal_override=lambda ids,i:old['steps'][i]['proposed_tokens'],state_probe=probe)
                checks.append(dict(case=f'dynamic-{length}',outputs=old['token_ids']==replay['token_ids'],
                    transactions=signature(old)==signature(replay),state=original_traces==traces))
                save(f'state-{length}.json',dict(persistent=original_traces,full_rebuild=traces))
                rejected=run(f'reject-{length}',base[:length],max_tokens=16,k_controller=ReplayK([1,6,1]),
                    proposal_override=lambda ids,i:[0]*len(ids))
                checks.append(dict(case=f'reject-{length}',all_rejected=all(s['accepted']==0 for s in rejected['steps'])))
            for maximum in (1,2,7):
                result=run(f'max-{maximum}',base[:255],adaptive_k=True,max_tokens=maximum)
                checks.append(dict(case=f'max-{maximum}',length=len(result['token_ids'])==maximum))
            reference=run('eos-reference',base[:255],k=3,max_tokens=8)
            for label,token,override in (
                ('initial',reference['token_ids'][0],None),('accepted',reference['token_ids'][1],None),
                ('fallback',reference['token_ids'][1],lambda ids,i:[0]*len(ids))):
                r=run('eos-'+label,base[:255],adaptive_k=True,max_tokens=16,ignore_eos=False,
                    eos_override=token,proposal_override=override)
                checks.append(dict(case='eos-'+label,terminated=r['token_ids']==reference['token_ids'][:1 if label=='initial' else 2]))
            for mode in (False,True):
                old=run('adaptive-replay-source-'+str(mode),prompt('sky',255),adaptive_k=mode,max_tokens=48)
                replay=run('adaptive-replay-'+str(mode),prompt('sky',255),max_tokens=48,
                    k_controller=ReplayK([s['requested_k'] for s in old['steps']]),
                    proposal_override=lambda ids,i:old['steps'][i]['proposed_tokens'])
                checks.append(dict(case='adaptive-replay-'+str(mode),exact=signature(old)==signature(replay)))
            accepts=Counter(s['accepted'] for r in records for s in r['steps'])
            checks.append(dict(case='acceptance-coverage',zero=accepts[0]>0,one=accepts[1]>0,
                full_six=accepts[6]>0,partial=any(0<s['accepted']<s['actual_k'] for r in records for s in r['steps'])))
        else:
            policies=[1,2,3,4,5,6,'adaptive']
            plans=[]
            for repeat in range(args.repeats):
                for family in args.families:
                    for length in (255,1024):
                        order=policies[repeat:]+policies[:repeat]
                        plans.extend(dict(family=family,length=length,repeat=repeat,policy=k) for k in order)
            save('plan.json',dict(plans=plans,prompts={f'{f}-{n}':prompt(f,n) for f in args.families for n in (255,1024)}))
            for length in (255,1024):
                for k in policies:
                    run(f'warmup-{length}-{k}',base[:length],k=3 if k=='adaptive' else k,adaptive_k=k=='adaptive',max_tokens=24)
            for plan in plans:
                f,n,r,k=(plan[x] for x in ('family','length','repeat','policy'))
                result=run(f'trial-{f}-{n}-{r}-{k}',prompt(f,n),k=3 if k=='adaptive' else k,adaptive_k=k=='adaptive')
                result['plan']=plan
                save(result['case']+'.json',result)
        save('summary.json',dict(checks=checks,records=[dict(case=r['case'],metrics=r['metrics']) for r in records],
            cleanup=all(r['kv_released'] and r['spec_state_released'] and r['draft_state_released'] for r in records)))
        if not all(value for row in checks for key,value in row.items() if key!='case'):
            raise RuntimeError('Phase 3 correctness failed; raw results retained')
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        atexit.unregister(engine.exit); engine.exit()


if __name__=='__main__': main()
