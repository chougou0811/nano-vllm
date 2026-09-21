"""Uninstrumented-host-copy timings plus focused state and draft operator controls."""
import argparse
import atexit
import json
from pathlib import Path
import subprocess

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.speculative.session import generate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    def save(name,value): (out/name).write_text(json.dumps(value,indent=2,default=str))
    config=dict(tensor_parallel_size=2,enforce_eager=True,max_model_len=2048,
                max_num_batched_tokens=2048,max_num_seqs=4,gpu_memory_utilization=0.70)
    save('manifest.json',dict(config=config,checkpoint='087cc27',torch=torch.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi'],text=True),
        target_revision='40c069824f4251a91eefaf281ebe4c544efd3e18',
        draft_revision='3d13517724e81cb409ddf1d4650772ec52f1e18e'))
    save('source.json',{str(p):p.read_text() for p in Path('nanovllm/speculative').glob('*.py')})
    engine=LLM('/root/autodl-tmp/models/Qwen3-14B',**config)
    common=dict(draft_path='/root/autodl-tmp/models/Qwen3-14B_eagle3',
                reference_path='/root/autodl-tmp/references/eagle-pinned',
                max_tokens=16,k=3,ignore_eos=True,audit=True)
    checks={}; records=[]; diagnostics=[]
    def run(name,prompt,mode,**extra):
        torch.cuda.reset_peak_memory_stats()
        r=generate(engine,prompt,**dict(common,**extra),draft_state_mode=mode)
        r.update(case=name,gpu_peak=torch.cuda.max_memory_allocated(),gpu_allocated=torch.cuda.memory_allocated())
        save(name+'.json',r); records.append(r)
        return r
    try:
        base=engine.tokenizer.encode('A scientist writes careful notes about an experiment. '*180)
        torch.manual_seed(2026)
        before=engine.generate([base[:31]],SamplingParams(temperature=0.8,max_tokens=8,ignore_eos=True),use_tqdm=False)
        old=run('reference',base[:255],'full_rebuild')
        def one(ids,i):
            if i==0: ids[1]=(ids[1]+1)%engine.model_runner.config.hf_config.vocab_size
            return ids
        one_result=run('accept-one',base[:255],'persistent',proposal_override=one)
        checks['accept_one']=one_result['steps'][0]['accepted']==1
        for label,kwargs,expected in (
            ('accepted-eos',dict(eos_override=old['token_ids'][1]),old['token_ids'][:2]),
            ('fallback-eos',dict(eos_override=old['token_ids'][1],proposal_override=lambda ids,i:[0]*len(ids)),old['token_ids'][:2])):
            r=run(label,base[:255],'persistent',ignore_eos=False,**kwargs)
            checks[label]=r['token_ids']==expected

        def probe(draft,state,proposals,iteration):
            # On identical target features, compare the retained draft cache to
            # the unchanged Phase 1 full rebuild, and locate projection rounding.
            model=draft.draft.model
            with torch.inference_mode():
                model.reset(); model.reset_kv()
                ids=torch.tensor([state['tokens'][1:]],device=draft.draft.device)
                features=state['features'][None]
                _,full=model(features,input_ids=ids,use_cache=True)
                errors=[float((a.float()-b.float()).abs().max()) for a,b in zip(draft.past[0],full[0])]
                reference=draft.draft.propose(state['features'],state['tokens'],len(proposals))
                start=draft.last_metrics['cursor_before']
                whole=model.fc(features); chunk=model.fc(features[:,start:])
                fc_error=float((whole[:,start:].float()-chunk.float()).abs().max())
                # Local FP32 uses identical real features/weights, not BF16 outputs.
                weights=model.fc.weight[:64].float()
                hp_full=torch.nn.functional.linear(features.float(),weights)[:,start:]
                hp_chunk=torch.nn.functional.linear(features[:,start:].float(),weights)
                hp_error=float((hp_full-hp_chunk).abs().max())
                diagnostics.append(dict(iteration=iteration,cursor=state['cached'],start=start,
                    proposal_equal=reference==proposals,proposals=proposals,reference=reference,
                    kv_max_abs=errors,fc_shape_max_abs=fc_error,fc_fp32_shape_max_abs=hp_error,
                    finite=all(bool(torch.isfinite(t).all()) for layer in draft.past for t in layer)))
        run('draft-shadow-control',base[:257],'persistent',state_probe=probe,max_tokens=24)
        save('draft-diagnostics.json',diagnostics)
        # Repeat in the same execution mode; no diagnostic tensor copies in timing runs.
        for length in (255,1024):
            run(f'warmup-{length}-full',base[:length],'full_rebuild')
            run(f'warmup-{length}-persistent',base[:length],'persistent')
            for repeat in range(3):
                modes=('full_rebuild','persistent') if repeat%2==0 else ('persistent','full_rebuild')
                for mode in modes:
                    run(f'timing-{length}-{repeat}-{mode}',base[:length],mode)
        for label,text in [('sky','Explain why the sky looks blue in two sentences.'),
                           ('arithmetic','What is 17 times 23? Show the calculation briefly.')]:
            prompt=engine.tokenizer.apply_chat_template([dict(role='user',content=text)],
                     tokenize=True,add_generation_prompt=True,enable_thinking=False)
            old=run(label+'-full',prompt,'full_rebuild',max_tokens=32)
            new=run(label+'-persistent',prompt,'persistent',max_tokens=32,state_probe=probe)
            repeated=run(label+'-repeat',prompt,'persistent',max_tokens=32)
            checks[label+'-deterministic']=(
                new['token_ids']==repeated['token_ids'] and
                [s['proposed_tokens'] for s in new['steps']]==[s['proposed_tokens'] for s in repeated['steps']])
            checks[label+'-outputs']=old['token_ids']==new['token_ids']
        torch.manual_seed(2026)
        after=engine.generate([base[:31]],SamplingParams(temperature=0.8,max_tokens=8,ignore_eos=True),use_tqdm=False)
        checks['off_parity']=before==after
        checks['cleanup']=all(r['kv_released'] and r['spec_state_released'] and r['draft_state_released'] for r in records)
        save('draft-diagnostics.json',diagnostics)
        save('summary.json',dict(checks=checks,diagnostic_proposals_equal=all(d['proposal_equal'] for d in diagnostics),
            records=[r['case'] for r in records]))
        if not all(checks.values()) or not all(d['proposal_equal'] and d['finite'] for d in diagnostics):
            raise RuntimeError('Phase 2 followup needs investigation; raw failures retained')
    finally:
        atexit.unregister(engine.exit); engine.exit()


if __name__=='__main__': main()
