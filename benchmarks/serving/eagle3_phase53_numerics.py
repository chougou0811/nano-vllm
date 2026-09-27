"""Teacher-forced draft shape audit; no production changes or tolerance gate."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from benchmarks.serving.eagle3_phase53 import (
    ROOT, TARGET, DRAFT, REFERENCE, make_items, serial_generation, stats, compare)
from nanovllm.speculative.draft import ReferenceDraft
from nanovllm.speculative.draft_batch import DraftBatchExecutor


@torch.inference_mode()
def audit(prepared):
    model = prepared[0].item.state.draft.model
    device = prepared[0].hidden.device
    forced, _ = serial_generation(prepared)
    b, width = len(prepared), max(p.end for p in prepared)
    traces, context = {}, {}
    handles = []
    for name, module in model.named_modules():
        if not name or 'rotary' in name or name.endswith('embed_tokens'):
            continue
        def hook(module, args, output, name=name):
            out = output[0] if isinstance(output, tuple) else output
            if not isinstance(out, torch.Tensor) or out.ndim < 2:
                return
            rows = context['rows']
            if out.shape[0] != len(rows):
                return
            for r,owner in enumerate(rows):
                key = (context['path'], owner, context['step'], name)
                traces[key] = out[r:r+1].detach().clone()
        handles.append(module.register_forward_hook(hook))
    try:
        for path in ('native', 'padded', 'batch'):
            groups = [list(range(b))] if path == 'batch' else [[i] for i in range(b)]
            for rows in groups:
                local_width = prepared[rows[0]].end if path == 'native' else width
                hidden = torch.cat([prepared[r].hidden for r in rows])
                past = tuple(tuple(torch.cat([F.pad(prepared[r].confirmed[l][v],
                    (0,0,0,local_width-prepared[r].end)) for r in rows]) for v in range(2))
                    for l in range(len(prepared[0].confirmed)))
                lengths = torch.tensor([prepared[r].end for r in rows], device=device)
                mask = torch.arange(local_width,device=device)[None,:] < lengths[:,None]
                for step in range(max(prepared[r].item.k for r in rows)):
                    context.update(path=path,rows=rows,step=step)
                    model.lm_head(model.norm(hidden))
                    keep = [j for j,r in enumerate(rows) if step+1 < prepared[r].item.k]
                    if not keep:
                        break
                    if len(keep) != len(rows):
                        selection = torch.tensor(keep,device=device,dtype=torch.long)
                        rows = [rows[j] for j in keep]
                        hidden = hidden.index_select(0,selection)
                        past = tuple(tuple(t.index_select(0,selection) for t in layer) for layer in past)
                        mask = mask.index_select(0,selection)
                        lengths = lengths.index_select(0,selection)
                        context['rows'] = rows
                    mask = torch.cat((mask,torch.ones((len(rows),1),dtype=torch.bool,device=device)),1)
                    ids = torch.tensor([[forced[r][step]] for r in rows],device=device)
                    hidden,past = model(hidden,input_ids=ids,past_key_values=past,
                        attention_mask=mask,position_ids=(lengths+step)[:,None],use_cache=True)
        result = []
        for key,value in traces.items():
            path,row,step,name = key
            if path != 'native':
                continue
            p = traces['padded',row,step,name]
            q = traces['batch',row,step,name]
            result.append(dict(row=row,step=step,operator=name,native_vs_padded=stats(value,p),
                padded_vs_batch=stats(p,q),native_vs_batch=stats(value,q)))
        return dict(forced_proposals=forced,operators=result)
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--disagreement', action='store_true')
    parser.add_argument('--input-name',default='disagreement')
    parser.add_argument('--output-name')
    args = parser.parse_args()
    out = ROOT/(args.output_name or ('disagreement-numerics' if args.disagreement else 'numerics'))
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    draft = ReferenceDraft(DRAFT,REFERENCE,TARGET,AutoConfig.from_pretrained(TARGET),torch.device('cuda:0'))
    data = []
    for dtype in (torch.bfloat16,torch.float32):
        draft.model.to(dtype=dtype)
        paths = sorted((ROOT/args.input_name).glob('*.pt')) if args.disagreement else [ROOT/f'capture/group{g}.pt' for g in range(3)]
        for group,path in enumerate(paths):
            raw = torch.load(path,weights_only=True)
            samples = raw if args.disagreement else raw[0]
            prepared = DraftBatchExecutor().prepare(make_items(samples,draft,dtype))
            data.append(dict(dtype=str(dtype),group=group,path=str(path),diagnostic=audit(prepared),
                             comparison=compare(prepared,DraftBatchExecutor())))
            del prepared
            (out/'summary.json').write_text(json.dumps(data,indent=2))
    print('NUMERICS_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
