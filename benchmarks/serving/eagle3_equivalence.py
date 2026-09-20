"""Test-only layer and KV replay audit. Does not patch production source files."""
import argparse
import atexit
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm import LLM
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.speculative.runtime import _forward, _zero
from nanovllm.utils.context import get_context, reset_context


def stats(a,b):
    a,b = a.float().reshape(-1),b.float().reshape(-1)
    d = a-b
    return dict(max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),
                rms=float(d.square().mean().sqrt()),relative_l2=float(d.norm()/a.norm().clamp_min(1e-30)),
                unequal=int(torch.count_nonzero(d)),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


def logical(cache,blocks,length):
    return cache[:,:,blocks].flatten(2,3)[:,:,:length]


def dense_attention(q,k,v,scale):
    # Explicit right-aligned causal mask; no SDPA implicit q!=k mask convention.
    repeats = q.shape[1]//k.shape[1]
    keys = k.repeat_interleave(repeats,dim=1).transpose(0,1)
    values = v.repeat_interleave(repeats,dim=1).transpose(0,1)
    scores = q.transpose(0,1) @ keys.transpose(-1,-2) * scale
    mask = torch.arange(k.shape[0],device=q.device)[None,:] <= (
        k.shape[0]-q.shape[0]+torch.arange(q.shape[0],device=q.device)[:,None])
    return (scores.masked_fill(~mask,float('-inf')).softmax(-1) @ values).transpose(0,1)


def math_attention(module,q,k,v):
    ctx = get_context()
    if module.k_cache.numel():
        store_kvcache(k,v,module.k_cache,module.v_cache,ctx.slot_mapping)
    if ctx.block_tables is not None:
        length = int(ctx.cu_seqlens_k[-1]) if ctx.is_prefill else int(ctx.context_lens[0])
        blocks = ctx.block_tables[0]
        k = module.k_cache[blocks].flatten(0,1)[:length]
        v = module.v_cache[blocks].flatten(0,1)[:length]
    return dense_attention(q,k,v,module.scale)


def math_norm(module,x,residual=None):
    # .float()/.to(float32) alias FP32 storage: avoid production in-place BF16
    # shortcuts in this explicitly high-precision, test-only execution control.
    base = x if residual is None else x+residual
    normalized = base*torch.rsqrt(base.square().mean(-1,keepdim=True)+module.eps)*module.weight
    return normalized if residual is None else (normalized,base)


if os.environ.get('EAGLE_AUDIT_FP32') == '1':
    Attention.forward = math_attention
    RMSNorm.forward = math_norm
    original_init = ModelRunner.__init__
    def float_init(self,config,rank,event):
        config.hf_config.dtype = torch.float32
        original_init(self,config,rank,event)
    ModelRunner.__init__ = float_init


class Capture:
    def __init__(self,runner,isolate=False):
        self.data,self.handles,self.isolation,self.linear = {},[],{},{}
        self.runner,self.isolate = runner,isolate
        for index,layer in enumerate(runner.model.model.layers):
            key = f'{index:02d}'
            self.handles.append(layer.register_forward_pre_hook(
                lambda m,a,k=key:self.layer_input(k,a)))
            self.handles.append(layer.register_forward_hook(
                lambda m,a,o,k=key:self.put(k+'.layer_output',(o[0].float()+o[1].float()).to(o[0].dtype))))
            for name,module in [('attention_input',layer.self_attn.qkv_proj),('mlp_input',layer.mlp)]:
                self.handles.append(module.register_forward_pre_hook(lambda m,a,k=key+'.'+name:self.put(k,a[0])))
            for name,module in [('qkv',layer.self_attn.qkv_proj),('q_norm',layer.self_attn.q_norm),
                                ('k_norm',layer.self_attn.k_norm),('attention_output',layer.self_attn),
                                ('gate_up',layer.mlp.gate_up_proj),('mlp_output',layer.mlp)]:
                self.handles.append(module.register_forward_hook(lambda m,a,o,k=key+'.'+name:self.put(k,o)))
            self.handles.append(layer.self_attn.rotary_emb.register_forward_hook(
                lambda m,a,o,k=key:self.rope(k,a,o)))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(
                lambda m,a,o,k=key:self.put(k+'.post_attention_residual',o[1])))
            self.handles.append(layer.self_attn.attn.register_forward_hook(
                lambda m,a,o,k=key:self.attention(k,m,a,o)))
            if isolate and index == 0:
                self.handles.append(layer.self_attn.qkv_proj.register_forward_hook(self.linear_control))
        self.handles.append(runner.model.model.norm.register_forward_hook(lambda m,a,o:self.put('final_norm',o[0])))
        self.handles.append(runner.model.lm_head.register_forward_pre_hook(lambda m,a:self.put('lm_head_input',a[0])))

    def put(self,key,value):
        self.data.setdefault(key,[]).append(value.detach().cpu())

    def layer_input(self,key,args):
        self.active_layer = key
        self.put(key+'.input_residual',args[1] if args[2] is None else (args[1].float()+args[2].float()).to(args[1].dtype))

    def rope(self,key,args,output):
        # get_rope is cached: all layers share this module, so gate its hooks.
        if self.active_layer == key:
            self.put(key+'.rope_q',output[0])
            self.put(key+'.rope_k',output[1])
            self.put(key+'.positions',args[0].reshape(-1,1))

    def linear_control(self,module,args,output):
        x,w = args[0].float(),module.weight.float()
        parallel = F.linear(x,w)
        serial = torch.cat([F.linear(row[None],w) for row in x])
        low_serial = torch.cat([F.linear(row[None],module.weight) for row in args[0]])
        self.linear = dict(fp32_serial_parallel=stats(serial,parallel),
                           native_dtype=str(output.dtype),
                           same_input_low_precision_serial_parallel=stats(low_serial,output),
                           native_vs_fp32=stats(output,parallel),
                           native_vs_rounded_fp32=stats(output,parallel.to(output.dtype)))
        if not getattr(self.runner,'_linear_profiled',False):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                F.linear(args[0][:1],module.weight)
                F.linear(args[0],module.weight)
                torch.cuda.synchronize()
            self.linear['profiled_cuda_kernels'] = sorted({e.name for e in prof.events() if 'CUDA' in str(e.device_type)})
            self.runner._linear_profiled = True

    def attention(self,key,module,args,output):
        self.put(key+'.attention_core',output.reshape(args[0].shape))
        if not self.isolate or args[0].dtype != torch.bfloat16:
            return
        ctx = get_context()
        q = args[0]
        count = q.shape[0]
        length = int(ctx.cu_seqlens_k[-1])
        table = ctx.block_tables
        kcache,vcache = module.k_cache,module.v_cache
        blocks = table[0]
        k,v = kcache[blocks].flatten(0,1)[:length],vcache[blocks].flatten(0,1)[:length]
        serial, varlen1 = [],[]
        for i in range(count):
            n = length-count+i+1
            lens = torch.tensor([n],device=q.device,dtype=torch.int32)
            serial.append(flash_attn_with_kvcache(q[i:i+1].unsqueeze(0),kcache,vcache,
                cache_seqlens=lens,block_table=table,softmax_scale=module.scale,causal=True).reshape(1,*q.shape[1:]))
            varlen1.append(flash_attn_varlen_func(q[i:i+1],kcache,vcache,
                cu_seqlens_q=torch.tensor([0,1],device=q.device,dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0,n],device=q.device,dtype=torch.int32),
                max_seqlen_q=1,max_seqlen_k=n,block_table=table,softmax_scale=module.scale,causal=True))
        cache_multi = flash_attn_with_kvcache(q.unsqueeze(0),kcache,vcache,
            cache_seqlens=torch.tensor([length],device=q.device,dtype=torch.int32),
            block_table=table,softmax_scale=module.scale,causal=True).reshape_as(q)
        reference = dense_attention(q.double(),k.double(),v.double(),module.scale)
        self.isolation[key] = dict(serial_cache_vs_parallel_varlen=stats(torch.cat(serial),output),
            serial_varlen_vs_parallel_varlen=stats(torch.cat(varlen1),output),
            parallel_cache_vs_parallel_varlen=stats(cache_multi,output),
            serial_cache_vs_fp64=stats(torch.cat(serial),reference),
            parallel_varlen_vs_fp64=stats(output,reference),query_len=count,context_len=length,
            causal=True,dtype=str(q.dtype),positions=list(range(length-count,length)),
            slots=ctx.slot_mapping.cpu().tolist(),blocks=table.cpu().tolist())
        if key == '00' and not getattr(self.runner,'_audit_profiled',False):
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                flash_attn_with_kvcache(q[:1].unsqueeze(0),kcache,vcache,
                    cache_seqlens=torch.tensor([length-count+1],device=q.device,dtype=torch.int32),
                    block_table=table,softmax_scale=module.scale,causal=True)
                flash_attn_varlen_func(q,kcache,vcache,cu_seqlens_q=ctx.cu_seqlens_q,
                    cu_seqlens_k=ctx.cu_seqlens_k,max_seqlen_q=count,max_seqlen_k=length,
                    block_table=table,softmax_scale=module.scale,causal=True)
                torch.cuda.synchronize()
            self.isolation[key]['profiled_cuda_kernels'] = sorted({e.name for e in prof.events()
                                                                   if 'CUDA' in str(e.device_type)})
            self.runner._audit_profiled = True

    def finish(self):
        for handle in self.handles:
            handle.remove()
        return {k:torch.cat(v,dim=0) for k,v in self.data.items()}


def execute(runner,state,proposals,mode,isolate=False):
    capture = Capture(runner,isolate)
    outputs = []
    try:
        if mode == 'parallel':
            logits,_,_,_ = _forward(runner,state,state['tokens']+proposals,state['cached'])
            if runner.rank == 0:
                outputs.append(logits.float().cpu())
        else:
            for i in range(len(proposals)+1):
                tokens = state['tokens']+proposals[:i]
                if mode == 'varlen1':
                    logits,_,_,_ = _forward(runner,state,tokens,len(tokens)-1)
                else:
                    seq = Sequence(tokens)
                    seq.block_table = state['blocks'][:seq.num_blocks]
                    ids,pos = runner.prepare_decode([seq])
                    logits = runner.run_model(ids,pos,False)
                if runner.rank == 0:
                    outputs.append(logits.float().cpu())
                reset_context()
    finally:
        data = capture.finish()
        reset_context()
    if outputs:
        data['logits'] = torch.cat(outputs)
    return data,capture.isolation,capture.linear


@torch.inference_mode()
def audit_rpc(runner,operation,payload):
    torch.backends.cuda.matmul.allow_tf32 = False
    state = runner._eagle_state
    blocks = state['blocks']
    out = Path(payload['out'])
    out.mkdir(parents=True,exist_ok=True)
    if operation == 'initialize':
        runner._serial_reference = runner.kv_cache[:,:,blocks].clone()
        runner._serial_features = state['features'].clone()
        return
    if operation == 'compare':
        saved = runner.kv_cache[:,:,blocks].clone()
        cached = state['cached']
        prefix = saved.flatten(2,3)[:,:,:cached]
        ref = runner._serial_reference.flatten(2,3)[:,:,:cached]
        kv = {str(i):stats(prefix[:,i],ref[:,i]) for i in range(prefix.shape[1])}
        captures,isolation,linear = {},{},{}
        new_reference = None
        try:
            for name,mode,base in [('serial_s','serial',saved),('varlen1_s','varlen1',saved),
                                   ('parallel_s','parallel',saved),('serial_n','serial',runner._serial_reference),
                                   ('parallel_n','parallel',runner._serial_reference)]:
                runner.kv_cache[:,:,blocks] = base
                data,iso,lin = execute(runner,state,payload['proposals'],mode,name=='parallel_s')
                captures[name] = data
                if name == 'parallel_s':
                    runner._expected_kv = runner.kv_cache[:,:,blocks].clone()
                    isolation,linear = iso,lin
                if name == 'serial_n':
                    new_reference = runner.kv_cache[:,:,blocks].clone()
        finally:
            runner.kv_cache[:,:,blocks] = saved
        pairs = {}
        for a,b in [('serial_s','parallel_s'),('serial_s','varlen1_s'),
                    ('varlen1_s','parallel_s'),('serial_n','parallel_n'),('parallel_n','parallel_s')]:
            pairs[a+'__'+b] = {key:dict(stats(captures[a][key],captures[b][key]),
                per_query_max_abs=(captures[a][key].float()-captures[b][key].float()).reshape(
                    captures[a][key].shape[0],-1).abs().amax(dim=1).tolist()) for key in captures[a]}
        # Serialize first differing operator and largest absolute-error operator.
        nonzero = [key for key in captures['serial_s'] if pairs['serial_s__parallel_s'][key]['unequal']]
        chosen = nonzero[:1]
        if nonzero:
            chosen.append(max(nonzero,key=lambda key:pairs['serial_s__parallel_s'][key]['max_abs']))
        chosen += ['logits'] if runner.rank == 0 else []
        chosen += ['00.attention_input','00.qkv','00.q_norm','00.k_norm','00.rope_q','00.rope_k','00.attention_core']
        torch.save({name:{key:data[key] for key in set(chosen)} for name,data in captures.items()},
                   out/f"iteration-{payload['iteration']:02d}-rank-{runner.rank}.pt")
        indices = [2,len(runner.model.model.layers)//2,len(runner.model.model.layers)-3]
        runner._expected_features = torch.cat([captures['parallel_s'][f'{i:02d}.input_residual'] for i in indices],dim=-1)
        runner._next_serial_features = torch.cat([captures['serial_n'][f'{i:02d}.input_residual'] for i in indices],dim=-1)
        runner._next_reference = new_reference
        runner._before_kv = saved
        runner._before_features = state['features'].clone()
        runner._before_cached = cached
        summary = dict(iteration=payload['iteration'],cached=cached,tokens=state['tokens'],
            proposals=payload['proposals'],prefix_kv=kv,pairs=pairs,attention_isolation=isolation,
            linear_fp32=linear,first_nonzero=nonzero[:1],saved_operators=chosen,
            feature_prefix=stats(state['features'].cpu(),runner._serial_features.cpu()))
        if runner.rank == 0:
            summary['top1'] = {name:data['logits'].argmax(-1).tolist() for name,data in captures.items()}
        (out/f"iteration-{payload['iteration']:02d}-rank-{runner.rank}.json").write_text(json.dumps(summary,indent=2))
        return summary.get('top1')
    if operation == 'commit_check':
        keep = 1+payload['accepted']
        old,new = runner._before_cached,state['cached']
        current = runner.kv_cache[:,:,blocks].flatten(2,3)
        expect = runner._expected_kv.flatten(2,3)
        before = runner._before_kv.flatten(2,3)
        expected_features = torch.cat([runner._before_features.cpu(),runner._expected_features[:keep]],dim=0)
        end = old+runner._expected_features.shape[0]
        checks = dict(cursor_correct=new==old+keep,retained_kv_exact=torch.equal(current[:,:,:new],expect[:,:,:new]),
            old_prefix_unchanged=torch.equal(current[:,:,:old],before[:,:,:old]),
            suffix_zero=not bool(torch.count_nonzero(current[:,:,new:end])),
            feature_retention_exact=torch.equal(state['features'].cpu(),expected_features),
            pending_token_count=len(state['tokens'])==new+1)
        runner._serial_reference = runner._next_reference
        runner._serial_reference.flatten(2,3)[:,:,new:] = 0
        runner._serial_features = torch.cat([runner._serial_features.cpu(),runner._next_serial_features[:keep]],dim=0)
        (out/f"commit-{payload['iteration']:02d}-rank-{runner.rank}.json").write_text(json.dumps(checks,indent=2))
        if not all(checks.values()):
            raise RuntimeError(checks)
        return checks


ModelRunner.equivalence_audit = audit_rpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--small',action='store_true')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    frozen = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('nanovllm').rglob('*.py')}
    (out/'frozen-source.json').write_text(json.dumps(frozen,indent=2))
    (out/'probe-source.py').write_text(Path(__file__).read_text())
    model = '/root/autodl-tmp/models/Qwen3-'+('0.6B' if args.small else '14B')
    engine = LLM(model,tensor_parallel_size=2,enforce_eager=True,max_model_len=256,
                 max_num_batched_tokens=256,max_num_seqs=1,gpu_memory_utilization=0.2 if args.small else 0.75)
    runner,bm = engine.model_runner,engine.scheduler.block_manager
    (out/'manifest.json').write_text(json.dumps(dict(arguments=vars(args),model=model,
        config=runner.config.hf_config.to_dict(),torch=torch.__version__,cuda=torch.version.cuda,
        fp32_math_control=os.environ.get('EAGLE_AUDIT_FP32')=='1',tp=2,eager=True,
        gpu=subprocess.check_output(['nvidia-smi'],text=True)),indent=2,default=str))
    questions = ['Explain why the sky looks blue in two sentences.',
                 'What is 17 multiplied by 23? Show the calculation briefly.']
    try:
        for case,question in enumerate(questions):
            history = json.loads(Path(f'/root/autodl-tmp/benchmarks/eagle3-phase1/trial-04/chat-{case}-k3.json').read_text())
            prompt = engine.tokenizer.apply_chat_template([dict(role='user',content=question)],tokenize=True,
                add_generation_prompt=True,enable_thinking=False)
            lease = Sequence([0]*(len(prompt)+32))
            bm.allocate(lease,0)
            case_out = out/f'case-{case}'
            try:
                runner.call('eagle3','begin',dict(tokens=prompt,blocks=lease.block_table,audit=True))
                first = runner.call('eagle3','prefill',{})
                seed = history['token_ids'][0]
                if not args.small:
                    assert seed == first['token']
                runner.call('eagle3','seed',dict(token=seed))
                runner.call('equivalence_audit','initialize',dict(out=str(case_out)))
                count = 1
                for iteration,step in enumerate(history['steps']):
                    payload = dict(out=str(case_out),iteration=iteration,proposals=step['proposed_tokens'])
                    top1 = runner.call('equivalence_audit','compare',payload)
                    verification = runner.call('eagle3','verify',dict(proposals=step['proposed_tokens']))
                    assert verification['target_ids'] == top1['parallel_s']
                    if not args.small:
                        assert verification['target_ids'] == step['target_ids']
                    runner.call('eagle3','commit',dict(accepted=step['accepted'],tokens=step['committed_tokens']))
                    runner.call('equivalence_audit','commit_check',dict(payload,accepted=step['accepted']))
                    print(case,iteration,'output',count,'first_op',top1,flush=True)
                    count += len(step['committed_tokens'])
                    if count > (20 if case == 0 else 10):
                        break
            finally:
                runner.call('eagle3','close',{})
                bm.deallocate(lease)
        assert frozen == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('nanovllm').rglob('*.py')}
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


if __name__ == '__main__':
    main()
