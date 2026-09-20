"""Small reproducible, greedy EAGLE integration audit (not a speed benchmark)."""
import argparse
import atexit
import json
import os
from pathlib import Path
import subprocess
import traceback
import hashlib
from time import perf_counter_ns

import numpy as np
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.speculative.session import generate
from nanovllm.engine.block_manager import BlockManager


def require_parity(summary):
    if not all(summary[key] for key in ['all_equal','off_parity','forced_rejection_equal']):
        raise RuntimeError('Correctness parity failed; raw results and summary are retained')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-probe', action='store_true')
    parser.add_argument('--natural-only', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (out/name).write_text(json.dumps(obj, indent=2, default=str))
    sources = list(Path('nanovllm').rglob('*.py')) + list(Path('benchmarks/serving').glob('eagle3*.py'))
    save('source-snapshot.json',{str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                           source=p.read_text()) for p in sources})
    (out/'tracked.diff').write_bytes(subprocess.check_output(['git','diff','--','nanovllm']))
    config = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=2048,
                  max_num_batched_tokens=2048, max_num_seqs=4, gpu_memory_utilization=0.75)
    save('manifest.json', dict(config=config, target_revision='40c069824f4251a91eefaf281ebe4c544efd3e18',
        draft_revision='3d13517724e81cb409ddf1d4650772ec52f1e18e',
        git=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        environment={key:os.environ.get(key) for key in ['CUDA_VISIBLE_DEVICES','NCCL_DEBUG',
            'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR']},
        arguments=vars(args),torch=torch.__version__, cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi'], text=True)))
    engine = LLM('/root/autodl-tmp/models/Qwen3-14B', **config)
    common = dict(draft_path='/root/autodl-tmp/models/Qwen3-14B_eagle3',
                  reference_path='/root/autodl-tmp/references/eagle-pinned',
                  max_tokens=32 if args.natural_only else 12, ignore_eos=True, audit=True)
    records = []
    native_records = []
    original_sampler = engine.model_runner.sampler.forward
    def native(prompt):
        bm = engine.scheduler.block_manager
        assert engine.is_finished() and not bm.used_block_ids
        engine.scheduler.block_manager = BlockManager(len(bm.blocks),bm.block_size)
        engine.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(dim=-1)
        times = []
        start = perf_counter_ns()
        engine.add_request(prompt, SamplingParams(max_tokens=common['max_tokens'], ignore_eos=True))
        output = None
        while not engine.is_finished():
            finished, _ = engine.step()
            times.append(perf_counter_ns())
            if finished:
                output = finished[0][1]
        engine.model_runner.sampler.forward = original_sampler
        return dict(token_ids=output, arrival_ns=start, token_times_ns=times,
                    prefix_cache='fresh metadata; no hit')
    try:
        prompt = engine.tokenizer.encode('The capital of France is')
        if args.reference_probe:
            checks = []
            def probe(draft, state, value):
                if draft is not None:
                    model = draft.model
                    model.top_k, model.total_tokens, model.depth = 1, len(value), len(value)-1
                    model.init_tree()
                    model.reset_kv()
                    ids = torch.tensor([state['tokens']],device=draft.device)
                    with torch.inference_mode():
                        reference = model.topK_genrate(state['features'].unsqueeze(0),ids,None,None)[0]
                    expected = [state['tokens'][-1]]+value
                    actual = reference.flatten().tolist()
                    checks.append(dict(proposal_equal=actual==expected,actual=actual,expected=expected))
                    if len(checks) == 1:
                        torch.save(dict(tokens=state['tokens'][:-1],pending=state['tokens'][-1],
                                        features=state['features'].cpu()),out/'target-features.pt')
                    model.reset()
                    model.reset_kv()
                    assert actual == expected
                else:
                    from eagle.model.utils import evaluate_posterior
                    pending, proposals, accepted = value
                    candidate = torch.tensor([[pending]+proposals],device=state['logits'].device)
                    _, length, logits = evaluate_posterior(state['logits'].unsqueeze(0),candidate,None)
                    checks.append(dict(acceptance_equal=int(length)==accepted.matched,
                        fallback_equal=accepted.fallback is None or int(logits.argmax())==accepted.fallback))
                    assert checks[-1]['acceptance_equal'] and checks[-1]['fallback_equal']
            diagnostic = generate(engine,prompt,k=3,reference_probe=probe,**common)
            save('reference-checks.json',checks)
            save('reference-diagnostic.json',diagnostic)
        torch.manual_seed(2026)
        before = engine.generate([prompt], SamplingParams(temperature=0.8,max_tokens=8,ignore_eos=True),use_tqdm=False)
        prompts = [('short',prompt)]
        base = engine.tokenizer.encode('A scientist writes careful notes about an experiment. ' * 100)
        prompts += [(f'boundary-{n}',base[:n]) for n in [255,256,257,511,512]]
        if args.natural_only:
            questions = ['Explain why the sky looks blue in two sentences.',
                         'What is 17 multiplied by 23? Show the calculation briefly.',
                         'Write a Python function that returns the larger of two integers.']
            prompts = [(f'chat-{i}',engine.tokenizer.apply_chat_template(
                [dict(role='user',content=q)],tokenize=True,add_generation_prompt=True,enable_thinking=False))
                for i,q in enumerate(questions)]
        for name, ids in prompts:
            control = native(ids)
            native_records.append(dict(case=name,**control))
            save(name+'-original.json',control)
            for k in [0,3]:
                result = generate(engine,ids,k=k,**common)
                result['original_equal'] = result['token_ids'] == control['token_ids']
                result['case'] = name
                records.append(result)
                save(name+f'-k{k}.json',result)
                print(name,k,'equal',result['original_equal'],'accepted',sum(s['accepted'] for s in result['steps']),flush=True)
        forced = generate(engine,prompt,k=3,proposal_override=lambda ids,step:[0]*len(ids),**common)
        forced['original_equal'] = forced['token_ids'] == native(prompt)['token_ids']
        save('forced-rejection.json',forced)
        boundary_control = native(base[:255])['token_ids']
        boundary_reject = generate(engine,base[:255],k=3,
            proposal_override=lambda ids,step:[0]*len(ids),**common)
        assert boundary_reject['token_ids'] == boundary_control
        save('boundary-forced-rejection.json',boundary_reject)
        for limit in [1,2,5]:
            result = generate(engine,prompt,k=3,**dict(common,max_tokens=limit))
            assert len(result['token_ids']) == limit and result['kv_released']
            save(f'max-tokens-{limit}.json',result)
        first_token = native(prompt)['token_ids'][0]
        eos = generate(engine,prompt,k=3,eos_override=first_token,**dict(common,ignore_eos=False))
        assert eos['token_ids'] == [first_token]
        save('eos.json',eos)
        accepted_eos = generate(engine,base[:255],k=3,eos_override=boundary_control[1],
                               **dict(common,ignore_eos=False))
        assert accepted_eos['token_ids'] == boundary_control[:2]
        assert accepted_eos['steps'][0]['accepted'] == 1
        save('accepted-eos.json',accepted_eos)
        short_control = native(prompt)['token_ids']
        fallback_eos = generate(engine,prompt,k=3,eos_override=short_control[1],
            proposal_override=lambda ids,step:[0]*len(ids),**dict(common,ignore_eos=False))
        assert fallback_eos['token_ids'] == short_control[:2]
        assert fallback_eos['steps'][0]['accepted'] == 0
        save('fallback-eos.json',fallback_eos)
        def fail(ids,step):
            raise RuntimeError('intentional host failure')
        try:
            generate(engine,prompt,k=3,proposal_override=fail,**common)
        except RuntimeError as exc:
            assert str(exc) == 'intentional host failure'
        assert not engine.scheduler.block_manager.used_block_ids
        assert engine.model_runner._eagle_state is None
        torch.manual_seed(2026)
        after = engine.generate([prompt], SamplingParams(temperature=0.8,max_tokens=8,ignore_eos=True),use_tqdm=False)
        save('off-parity.json',dict(before=before,after=after,equal=before==after))
        save('draft-provenance.json',engine._eagle_draft.provenance)
        save('gpu-final.json',dict(gpu=subprocess.check_output(['nvidia-smi'],text=True),
            cache_blocks=len(engine.scheduler.block_manager.blocks),
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))
        metrics = []
        for r in records:
            ts = r['token_times_ns']
            itl = np.diff(ts)/1e6
            steps = r['steps']
            proposed = sum(len(s['proposed_tokens']) for s in steps)
            accepted = sum(s['accepted'] for s in steps)
            metrics.append(dict(case=r['case'],k=r['k'],equal=r['original_equal'],
                ttft_ms=(ts[0]-r['arrival_ns'])/1e6,tpot_ms=float(itl.mean()),
                itl_p50_p95_p99_ms=np.percentile(itl,[50,95,99]).tolist(),
                output_tokens_s=len(ts)/((ts[-1]-r['arrival_ns'])/1e9),
                proposed=proposed,accepted=accepted,acceptance_rate=accepted/proposed if proposed else None,
                accepted_per_verification=accepted/len(steps),
                effective_tokens_per_verification=sum(len(s['committed_tokens']) for s in steps)/len(steps),
                target_forwards=r['target_forward_count'],speculator_forwards=r['speculator_forward_count'],
                rejections=sum(s['rejection'] for s in steps),discarded=sum(s['discarded'] for s in steps),
                proposal_ms=sum(s['proposal_latency_ns'] for s in steps)/1e6,
                verification_ms=sum(s['verification_latency_ns'] for s in steps)/1e6))
        summary = dict(metrics=metrics,off_parity=before==after,
             forced_rejection_equal=forced['original_equal'],all_equal=all(r['original_equal'] for r in records),
             original_metrics=[dict(case=r['case'],
                 ttft_ms=(r['token_times_ns'][0]-r['arrival_ns'])/1e6,
                 tpot_ms=float(np.diff(r['token_times_ns']).mean()/1e6),
                 itl_p50_p95_p99_ms=(np.percentile(np.diff(r['token_times_ns']),[50,95,99])/1e6).tolist(),
                 output_tokens_s=len(r['token_ids'])/((r['token_times_ns'][-1]-r['arrival_ns'])/1e9))
                 for r in native_records])
        save('summary.json',summary)
        require_parity(summary)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        engine.model_runner.sampler.forward = original_sampler
        atexit.unregister(engine.exit)
        engine.exit()


if __name__ == '__main__':
    main()
