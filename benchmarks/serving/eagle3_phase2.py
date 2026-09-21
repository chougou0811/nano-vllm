"""Persistent draft audit: fixed-proposal replay and unmodified natural proposals."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np
import torch

from nanovllm import LLM
from nanovllm.speculative.session import generate


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (out/name).write_text(json.dumps(value,indent=2,default=str))
    config = dict(tensor_parallel_size=2,enforce_eager=True,max_model_len=2048,
                  max_num_batched_tokens=2048,max_num_seqs=4,gpu_memory_utilization=0.70)
    save('manifest.json',dict(config=config,arguments=vars(args),torch=torch.__version__,cuda=torch.version.cuda,
        git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gpu=subprocess.check_output(['nvidia-smi'],text=True),
        environment={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','NCCL_DEBUG',
            'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')},
        target_revision='40c069824f4251a91eefaf281ebe4c544efd3e18',
        draft_revision='3d13517724e81cb409ddf1d4650772ec52f1e18e'))
    save('sources.json',{str(p):p.read_text() for folder in ('nanovllm/speculative','benchmarks/serving')
                         for p in Path(folder).glob('*.py') if folder.endswith('speculative') or p.name=='eagle3_phase2.py'})
    engine = LLM('/root/autodl-tmp/models/Qwen3-14B',**config)
    common = dict(draft_path='/root/autodl-tmp/models/Qwen3-14B_eagle3',
                  reference_path='/root/autodl-tmp/references/eagle-pinned',
                  max_tokens=16,k=3,ignore_eos=True,audit=True)
    records, comparisons = [], []
    try:
        base = engine.tokenizer.encode('A scientist writes careful notes about an experiment. '*180)
        def run(name,prompt,mode,override=None,**extra):
            traces=[]
            def probe(draft,state,proposals,step):
                cache = engine.model_runner.kv_cache[:,:,state['blocks']].flatten(2,3)
                cache = cache[:,:,:state['cached']]
                traces.append(dict(step=step,cursor=state['cached'],tokens=list(state['tokens']),
                    target_kv=digest(cache),features=digest(state['features']),
                    draft_cursor=draft.cursor,proposals=list(proposals),
                    draft_kv=[digest(t) for layer in draft.past for t in layer] if draft.past else []))
            torch.cuda.reset_peak_memory_stats()
            result=generate(engine,prompt,**dict(common,**extra),draft_state_mode=mode,
                            proposal_override=override,state_probe=probe)
            result.update(case=name,mode=mode,traces=traces,
                gpu_allocated=torch.cuda.memory_allocated(),gpu_peak=torch.cuda.max_memory_allocated(),
                gpu_reserved=torch.cuda.memory_reserved(),leased_blocks=(len(prompt)+common['max_tokens']+255)//256)
            save(name+'.json',result)
            records.append(result)
            return result
        # Initial compilation/load effects are retained separately and not used for timing comparisons.
        run('warmup-full',base[:32],'full_rebuild',max_tokens=8)
        run('warmup-persistent',base[:32],'persistent',max_tokens=8)
        for length in (255,256,257,1024):
            old=run(f'boundary-{length}-full',base[:length],'full_rebuild')
            replay=run(f'boundary-{length}-replay',base[:length],'persistent',
                       lambda ids,i:old['steps'][i]['proposed_tokens'])
            natural=run(f'boundary-{length}-natural',base[:length],'persistent')
            fields=('proposed_tokens','target_ids','accepted','committed_tokens','ranks')
            tracefields=('cursor','tokens','target_kv','features')
            comparisons.append(dict(case=f'boundary-{length}',
                replay_outputs=old['token_ids']==replay['token_ids'],
                replay_steps=len(old['steps'])==len(replay['steps']) and all(
                    all(a[f]==b[f] for f in fields) for a,b in zip(old['steps'],replay['steps'])),
                replay_state=all(all(a[f]==b[f] for f in tracefields) for a,b in zip(old['traces'],replay['traces'])),
                natural_outputs=old['token_ids']==natural['token_ids'],
                natural_proposals=[s['proposed_tokens'] for s in old['steps']]==[s['proposed_tokens'] for s in natural['steps']]))
        for mode in ('full_rebuild','persistent'):
            for maximum in (1,2,5):
                run(f'max-{maximum}-{mode}',base[:33],mode,max_tokens=maximum)
            rejection=run(f'repeated-reject-{mode}',base[:255],mode,lambda ids,i:[0]*len(ids))
            comparisons.append(dict(case=f'repeated-reject-{mode}',all_rejected=all(s['accepted']==0 for s in rejection['steps'])))
            first=rejection['token_ids'][0]
            eos=run(f'eos-{mode}',base[:255],mode,ignore_eos=False,eos_override=first)
            assert eos['token_ids']==[first]
            captured=[]
            def fail(state,target,ids,step):
                captured.append(state)
                raise RuntimeError('intentional phase2 exception')
            try:
                generate(engine,base[:255],**common,draft_state_mode=mode,state_probe=fail)
            except RuntimeError as exc:
                assert str(exc)=='intentional phase2 exception'
            assert captured and captured[0].closed and captured[0].past is None
            assert not engine.scheduler.block_manager.used_block_ids and engine.model_runner._eagle_state is None
        metrics=[]
        for r in records:
            steps=r['steps']; ts=r['token_times_ns']; ds=[s['draft_state'] for s in steps if s['draft_state']]
            metrics.append(dict(case=r['case'],mode=r['mode'],outputs=len(ts),
                ttft_ms=(ts[0]-r['arrival_ns'])/1e6,
                tpot_ms=float(np.diff(ts).mean()/1e6) if len(ts)>1 else None,
                itl_p50_p95_p99_ms=(np.percentile(np.diff(ts),[50,95,99])/1e6).tolist() if len(ts)>1 else [],
                output_tokens_s=len(ts)*1e9/(ts[-1]-r['arrival_ns']),
                proposal_ms=sum(s['proposal_latency_ns'] for s in steps)/1e6,
                verification_ms=sum(s['verification_latency_ns'] for s in steps)/1e6,
                conditioning_ms=sum(s['conditioning_ns'] for s in ds)/1e6,
                draft_tokens_processed=sum(s['draft_tokens_processed'] for s in ds),
                reused_prefix_tokens=sum(s['reused_draft_prefix_tokens'] for s in ds),
                rollback_tokens=sum(s['rollback_tokens'] for s in ds),
                target_forwards=r['target_forward_count'],draft_forwards=r['speculator_forward_count'],
                accepted=sum(s['accepted'] for s in steps),proposed=sum(len(s['proposed_tokens']) for s in steps),
                effective_outputs_per_verification=(len(ts)-1)/len(steps) if steps else None,
                draft_peak_kv_bytes=max([s['draft_kv_bytes'] for s in ds],default=0),
                gpu_peak=r['gpu_peak'],gpu_allocated=r['gpu_allocated']))
        save('summary.json',dict(comparisons=comparisons,metrics=metrics,
            cleanup=all(r['kv_released'] and r['spec_state_released'] for r in records),
            natural_disagreement_requires_diagnostics=any(not c.get('natural_proposals',True) for c in comparisons)))
        if not all(value for c in comparisons for key,value in c.items() if key != 'case'):
            raise RuntimeError('Phase 2 comparison failed; all raw records retained')
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


if __name__=='__main__': main()
