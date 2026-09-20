"""Teacher-forced serial/parallel verification on identical physical prefix KV."""
import argparse
import atexit
import json
from pathlib import Path

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.speculative.session import generate
from nanovllm.utils.context import reset_context


@torch.inference_mode()
def serial_probe(runner,payload):
    blocks = payload['blocks']
    saved = runner.kv_cache[:,:,blocks].clone()
    rows = []
    try:
        for i in range(len(payload['proposals'])+1):
            seq = Sequence(payload['tokens']+payload['proposals'][:i])
            seq.block_table = blocks[:seq.num_blocks]
            ids, positions = runner.prepare_decode([seq])
            logits = runner.run_model(ids,positions,False)
            if runner.rank == 0:
                rows.append(logits[0].float().cpu())
            reset_context()
    finally:
        runner.kv_cache[:,:,blocks] = saved
        reset_context()
    return torch.stack(rows) if rows else None


# Test-only method is installed in both spawned interpreters. No production edit.
ModelRunner.eagle_serial_probe = serial_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    engine = LLM('/root/autodl-tmp/models/Qwen3-14B',tensor_parallel_size=2,
                 enforce_eager=True,max_model_len=2048,max_num_batched_tokens=2048,
                 max_num_seqs=4,gpu_memory_utilization=0.75)
    original = engine.model_runner.sampler.forward
    questions = ['Explain why the sky looks blue in two sentences.',
                 'What is 17 multiplied by 23? Show the calculation briefly.']
    summaries = []
    try:
        for index,question in enumerate(questions):
            prompt = engine.tokenizer.apply_chat_template([dict(role='user',content=question)],
                tokenize=True,add_generation_prompt=True,enable_thinking=False)
            native_logits = []
            def sample(logits,temperatures):
                native_logits.append(logits[0].float().cpu())
                return logits.argmax(-1)
            engine.model_runner.sampler.forward = sample
            control = engine.generate([prompt],SamplingParams(max_tokens=32,ignore_eos=True),use_tqdm=False)[0]['token_ids']
            engine.model_runner.sampler.forward = original
            records, tensors = [],[]
            pending = {}
            def probe(draft,state,value):
                if draft is not None:
                    pending['tokens'] = list(state['tokens'])
                    pending['serial'] = engine.model_runner.call('eagle_serial_probe',dict(
                        tokens=state['tokens'],proposals=value,blocks=state['blocks']))
                else:
                    _, proposals, accepted = value
                    if not proposals:
                        return
                    parallel = state['logits'].float().cpu()
                    serial = pending['serial']
                    for row in range(accepted.accepted+1):
                        prefix = pending['tokens']+proposals[:row]
                        position = len(prefix)-len(prompt)
                        same_prefix = prefix == prompt+control[:position]
                        p,s = parallel[row],serial[row]
                        ref = native_logits[position] if position < len(native_logits) and same_prefix else None
                        ids = p.topk(5).indices.tolist()
                        record = dict(position=position,same_native_prefix=same_prefix,
                            parallel_top1=int(p.argmax()),serial_same_kv_top1=int(s.argmax()),
                            native_top1=int(ref.argmax()) if ref is not None else None,
                            parallel_top5=list(zip(ids,p[ids].tolist())),
                            serial_top5=list(zip(s.topk(5).indices.tolist(),s.topk(5).values.tolist())),
                            parallel_serial_max_abs=float((p-s).abs().max()),
                            parallel_serial_mean_abs=float((p-s).abs().mean()),
                            parallel_serial_cosine=float(torch.nn.functional.cosine_similarity(p,s,dim=0)))
                        if ref is not None:
                            record['native_top5'] = list(zip(ref.topk(5).indices.tolist(),ref.topk(5).values.tolist()))
                            record['native_parallel_max_abs'] = float((p-ref).abs().max())
                        records.append(record)
                    tensors.append(dict(prefix=pending['tokens'],proposals=proposals,serial=serial,parallel=parallel))
            common = dict(draft_path='/root/autodl-tmp/models/Qwen3-14B_eagle3',
                          reference_path='/root/autodl-tmp/references/eagle-pinned',
                          max_tokens=32,k=3,ignore_eos=True,audit=True)
            result = generate(engine,prompt,reference_probe=probe,**common)
            repeat = generate(engine,prompt,**common)
            prior = json.loads((Path('/root/autodl-tmp/benchmarks/eagle3-phase1/trial-04')/f'chat-{index}-k3.json').read_text())
            summary = dict(case=index,native_equal=result['token_ids']==control,
                repeat_equal=result['token_ids']==repeat['token_ids'],
                prior_run_equal=result['token_ids']==prior['token_ids'],
                first_difference=next((i for i,(a,b) in enumerate(zip(control,result['token_ids'])) if a!=b),None),
                control=control,speculative=result['token_ids'],records=records)
            (out/f'chat-{index}.json').write_text(json.dumps(summary,indent=2))
            torch.save(dict(native_logits=native_logits,verifications=tensors),out/f'chat-{index}-logits.pt')
            summaries.append({k:v for k,v in summary.items() if k not in ['records','control','speculative']})
            print(summaries[-1],flush=True)
        (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    finally:
        engine.model_runner.sampler.forward = original
        atexit.unregister(engine.exit)
        engine.exit()


if __name__ == '__main__':
    main()
