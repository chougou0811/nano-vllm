"""GPU termination and failure-path regression for the optional draft executor."""
from dataclasses import asdict
import json

from nanovllm import SamplingParams
from nanovllm.speculative.draft_batch import DraftBatchExecutor
import nanovllm.speculative.concurrent_engine as engine_module
from benchmarks.serving.eagle3_phase53_serving import ServingRunner
from benchmarks.serving.eagle3_phase53 import ROOT, TARGET, DRAFT, REFERENCE
from benchmarks.serving.eagle3_phase42 import cleanup_state


def main():
    out = ROOT/'edges'
    out.mkdir(exist_ok=False)
    engine_module.ConcurrentModelRunner = ServingRunner
    engine = engine_module.ConcurrentLLMEngine(TARGET,draft_path=DRAFT,reference_path=REFERENCE,
        tensor_parallel_size=2,enforce_eager=True,max_model_len=2048,max_num_batched_tokens=2048,
        max_num_seqs=4,gpu_memory_utilization=.7,scheduler_policy='original',audit=True)
    data = dict(limits=[],normal_exit=False)
    executor = DraftBatchExecutor()
    stream = engine.tokenizer.encode('A scientist describes recorded observations. '*400,add_special_tokens=False)
    def generate(mode, limit):
        engine.speculative_coordinator.draft_batch_executor = executor if mode=='batch' else None
        seqs, commits = [], []
        for i,n in enumerate((255,256,257,1025)):
            engine.add_request(([2500+i]+stream)[:n], SamplingParams(temperature=1e-9,max_tokens=limit,ignore_eos=False))
            seqs.append(engine.scheduler.waiting[-1])
        while not engine.is_finished():
            engine.step()
            last = engine.speculative_coordinator.last_step
            if last['kind'] == 'decode':
                commits.append([{k:v for k,v in asdict(c).items() if k!='seq_id'} for c in last['entries']])
        clean = cleanup_state(engine,'speculative')
        ranks = engine.model_runner.call('phase53_control','snapshot')
        assert not any(clean.values()) and not any(r['targets'] for r in ranks)
        return dict(outputs=[s.completion_token_ids for s in seqs],commits=commits,cleanup=clean)
    try:
        for limit in (1,2,3,4,5,9):
            a,b = generate('serial',limit),generate('batch',limit)
            data['limits'].append(dict(limit=limit,serial=a,batch=b,exact=a==b))
            assert a==b
        probe = generate('serial',9)
        original_eos = engine.model_runner.config.eos
        engine.model_runner.config.eos = probe['outputs'][0][2]
        try:
            a,b = generate('serial',16),generate('batch',16)
            data['eos'] = dict(serial=a,batch=b,exact=a==b,injected_eos=engine.model_runner.config.eos)
            assert a==b and len(a['outputs'][0])<16
        finally:
            engine.model_runner.config.eos = original_eos
        engine.speculative_coordinator.draft_batch_executor = executor
        for i in range(2):
            engine.add_request(([3500+i]+stream)[:255],SamplingParams(temperature=1e-9,max_tokens=12))
        while engine.scheduler.waiting:
            engine.step()
        def fail(*args):
            raise RuntimeError('Phase53 intentional batched-draft exception')
        executor.observer = fail
        try:
            engine.step()
            raise AssertionError('Expected injected exception')
        except RuntimeError as error:
            if 'Phase53 intentional' not in str(error):
                raise
        clean = cleanup_state(engine,'speculative')
        ranks = engine.model_runner.call('phase53_control','snapshot')
        assert not any(clean.values()) and not any(r['targets'] for r in ranks)
        data['exception'] = dict(cleanup=clean,ranks=ranks,faulted=engine._faulted)
        data['normal_exit'] = True
    finally:
        engine.exit()
        (out/'summary.json').write_text(json.dumps(data,indent=2))
    print('EDGES_COMPLETE',flush=True)


if __name__=='__main__':
    main()
