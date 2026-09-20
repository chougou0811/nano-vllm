"""Offline local FP64 dot controls and independent small-model FP32 HF checks."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from benchmarks.serving.eagle3_equivalence import stats
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


def metadata_control(record):
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = Sequence.block_size
    blocks = record['attention_isolation']['00']['blocks'][0]
    seq = Sequence(record['tokens']+record['proposals'])
    seq.block_table = blocks
    seq.num_cached_tokens = record['cached']
    seq.num_scheduled_tokens = len(record['proposals'])+1
    _,positions = runner.prepare_prefill([seq])
    ctx = get_context()
    parallel = dict(positions=positions.tolist(),slots=ctx.slot_mapping.tolist(),
                    block_table=ctx.block_tables.tolist(),cu_q=ctx.cu_seqlens_q.tolist(),
                    cu_k=ctx.cu_seqlens_k.tolist(),max_q=ctx.max_seqlen_q,max_k=ctx.max_seqlen_k)
    serial = []
    for i in range(seq.num_scheduled_tokens):
        item = Sequence(record['tokens']+record['proposals'][:i])
        item.block_table = blocks[:item.num_blocks]
        _,pos = runner.prepare_decode([item])
        ctx = get_context()
        row = dict(position=int(pos[0]),slot=int(ctx.slot_mapping[0]),
                   context_len=int(ctx.context_lens[0]),block_table=ctx.block_tables.tolist())
        assert row['position'] == parallel['positions'][i]
        assert row['slot'] == parallel['slots'][i]
        assert row['context_len'] == record['cached']+i+1
        serial.append(row)
    reset_context()
    return dict(parallel=parallel,serial=serial,positions_slots_lengths_match=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bf16',required=True)
    parser.add_argument('--fp32',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path('/root/autodl-tmp/models/Qwen3-14B')
    config = AutoConfig.from_pretrained(root,local_files_only=True)
    index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    controls = []
    metadata = []
    for path in sorted(Path(args.bf16).glob('case-*/iteration-*-rank-0.json')):
        metadata.append(dict(file=str(path),**metadata_control(json.loads(path.read_text()))))
    (out/'metadata-controls.json').write_text(json.dumps(metadata,indent=2))
    for rank in range(2):
        weights = []
        for name,heads in [('q',config.num_attention_heads),('k',config.num_key_value_heads),('v',config.num_key_value_heads)]:
            key = f'model.layers.0.self_attn.{name}_proj.weight'
            with safe_open(root/index[key],framework='pt',device='cpu') as file:
                n = heads*config.head_dim//2
                weights.append(file.get_tensor(key)[rank*n:(rank+1)*n])
        w = torch.cat(weights).cuda()
        for case,iteration in [(0,0),(0,12),(1,0),(1,5)]:
            tensors = torch.load(Path(args.bf16)/f'case-{case}/iteration-{iteration:02d}-rank-{rank}.pt',weights_only=True)
            x = tensors['parallel_s']['00.attention_input'].cuda()
            with torch.inference_mode():
                ref = F.linear(x.double(),w.double())
                old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                modes = {}
                try:
                    for flag in [True,False]:
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
                        serial = torch.cat([F.linear(row[None],w) for row in x])
                        parallel = F.linear(x,w)
                        unequal = (serial != parallel).nonzero()
                        modes[str(flag)] = dict(serial_parallel=stats(serial,parallel),
                            serial_fp64=stats(serial,ref),parallel_fp64=stats(parallel,ref),
                            serial_rounded_fp64=stats(serial,ref.to(serial.dtype)),
                            parallel_rounded_fp64=stats(parallel,ref.to(parallel.dtype)),
                            examples=[dict(index=ij.tolist(),serial=float(serial[tuple(ij)]),
                                parallel=float(parallel[tuple(ij)]),fp64=float(ref[tuple(ij)])) for ij in unequal[:5]])
                finally:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old
                fp32s = torch.cat([F.linear(row[None].float(),w.float()) for row in x])
                fp32p = F.linear(x.float(),w.float())
            controls.append(dict(case=case,iteration=iteration,rank=rank,modes=modes,
                fp32_serial_parallel=stats(fp32s,fp32p),fp32_fp64=stats(fp32p,ref),
                identical_input=torch.equal(tensors['parallel_s']['00.attention_input'],tensors['serial_s']['00.attention_input'])))
    (out/'local-dot-controls.json').write_text(json.dumps(controls,indent=2))
    model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/models/Qwen3-0.6B',
        dtype=torch.float32,attn_implementation='eager',local_files_only=True).cuda().eval()
    checks = []
    for case,iteration in [(0,0),(0,12),(1,0),(1,5)]:
        root = Path(args.fp32)/f'case-{case}'
        record = json.loads((root/f'iteration-{iteration:02d}-rank-0.json').read_text())
        tensors = torch.load(root/f'iteration-{iteration:02d}-rank-0.pt',weights_only=True)
        for row in range(len(record['proposals'])+1):
            ids = torch.tensor([record['tokens']+record['proposals'][:row]],device='cuda')
            with torch.inference_mode():
                logits = model(ids,use_cache=False).logits[0,-1].cpu()
            nano = tensors['serial_n']['logits'][row]
            checks.append(dict(case=case,iteration=iteration,row=row,nano_top1=int(nano.argmax()),
                hf_top1=int(logits.argmax()),top1_equal=bool(nano.argmax()==logits.argmax()),
                error=stats(nano,logits)))
    (out/'hf-small-fp32.json').write_text(json.dumps(checks,indent=2))
    print('HF small FP32 top1',sum(c['top1_equal'] for c in checks),'/',len(checks),flush=True)
    if not all(c['top1_equal'] and c['error']['finite'] for c in checks):
        raise RuntimeError('Independent HF FP32 control failed; artifacts retained')


if __name__ == '__main__':
    main()
