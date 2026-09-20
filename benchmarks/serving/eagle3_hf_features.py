"""Independent HF check of the reference feature taps, outside target residency."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifact')
    parser.add_argument('--numerical-audit')
    args = parser.parse_args()
    path = Path(args.artifact)
    record = torch.load(path, map_location='cpu', weights_only=True)
    model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/models/Qwen3-14B',
        torch_dtype=torch.bfloat16, device_map='auto', max_memory={0:'22GiB',1:'22GiB'},
        attn_implementation='sdpa', local_files_only=True).eval()
    ids = torch.tensor([record['tokens']],device=model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        result = model(ids,output_hidden_states=True,use_cache=False)
    layers = [2,model.config.num_hidden_layers//2,model.config.num_hidden_layers-3]
    target = record['features'].float()
    reference = torch.cat([result.hidden_states[i][0].cpu() for i in layers],dim=-1).float()
    difference = (target-reference).abs()
    summary = dict(feature_layers=layers,finite=bool(torch.isfinite(reference).all()),
        feature_max_abs=float(difference.max()),feature_mean_abs=float(difference.mean()),
        feature_cosine_mean=float(torch.nn.functional.cosine_similarity(target,reference,dim=-1).mean()),
        nano_top1=record['pending'],hf_top1=int(result.logits[0,-1].argmax()),
        top1_equal=record['pending']==int(result.logits[0,-1].argmax()),
        hf_top5=result.logits[0,-1].topk(5).indices.tolist(),
        note='BF16 implementation comparison, not a raw max-abs pass threshold.')
    destination = (Path(args.numerical_audit)/'hf-initial-feature.json' if args.numerical_audit
                   else path.with_name('hf-feature-reference.json'))
    destination.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if args.numerical_audit:
        root = Path(args.numerical_audit)
        cases = []
        for index in range(2):
            audit = json.loads((root/f'chat-{index}.json').read_text())
            tensors = torch.load(root/f'chat-{index}-logits.pt',map_location='cpu',weights_only=True)
            position = audit['first_difference']
            prompt = tensors['verifications'][0]['prefix'][:-1]
            prefix = prompt+audit['control'][:position]
            ids = torch.tensor([prefix],device=model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                logits = model(ids,use_cache=False).logits[0,-1].float().cpu()
            native = tensors['native_logits'][position]
            cases.append(dict(case=index,position=position,hf_top1=int(logits.argmax()),
                native_top1=int(native.argmax()),hf_top5=list(zip(logits.topk(5).indices.tolist(),logits.topk(5).values.tolist())),
                native_hf_max_abs=float((native-logits).abs().max()),finite=bool(torch.isfinite(logits).all())))
        (root/'hf-teacher-forced.json').write_text(json.dumps(cases,indent=2))
        print(json.dumps(cases,indent=2))


if __name__ == '__main__':
    main()
