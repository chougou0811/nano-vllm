"""Out-of-window same-state oracle for a newly observed serving proposal change."""
import argparse
import copy
import json

import torch

import benchmarks.serving.eagle3_phase42 as base
from benchmarks.serving.eagle3_phase53 import ROOT, tree
from benchmarks.serving.eagle3_phase53_serving import ServingRunner, Collector, requests
from nanovllm.speculative.draft_state import DraftState
from nanovllm.speculative.draft_batch import DraftBatchExecutor
import nanovllm.speculative.concurrent_engine as engine_module


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--name',default='disagreement')
    parser.add_argument('--manifest')
    args = parser.parse_args()
    out = ROOT/args.name
    out.mkdir(exist_ok=False)
    engine_module.ConcurrentModelRunner = ServingRunner
    base.TimingCollector = Collector
    engine = engine_module.ConcurrentLLMEngine(base.TARGET,draft_path=base.DRAFT,reference_path=base.REFERENCE,
        tensor_parallel_size=2,enforce_eager=True,max_model_len=2048,max_num_batched_tokens=2048,
        max_num_seqs=4,gpu_memory_utilization=.7,scheduler_policy='original',audit=True)
    engine.phase53_audit = True
    executor = DraftBatchExecutor()
    engine.speculative_coordinator.draft_batch_executor = executor
    original = executor.propose
    frozen = DraftState.propose
    mismatches, calls, results = [], [], []
    current = {}
    def propose(items):
        clones = [copy.copy(i.state) for i in items]
        expected = [frozen(s,i.features,i.tokens,i.k,owner=i.request.seq_id) for s,i in zip(clones,items)]
        # Recover pre-call state metadata from the immutable source inputs.
        before = [(i.state.cursor,i.state.conditioned_tokens,i.state.past) for i in items]
        actual = original(items)
        call = dict(index=len(calls),expected=expected,actual=actual,
                    lengths=[i.features.shape[0] for i in items],equal=expected==actual,cell=dict(current))
        calls.append(call)
        if expected != actual:
            samples = [dict(features=i.features.detach().cpu().clone(),tokens=i.tokens,k=i.k,
                cursor=old[0],conditioned_tokens=old[1],past=tree(old[2],lambda t:t.cpu().clone()),
                generation=i.generation,seq_id=i.request.seq_id) for i,old in zip(items,before)]
            path = out/f'call{call["index"]}.pt'
            torch.save(samples,path)
            mismatches.append(dict(**call,path=str(path)))
        return actual
    executor.propose = propose
    try:
        cells = ([r for r in json.loads(open(args.manifest).read())['checks'] if not r['exact_signature']]
                 if args.manifest else [dict(family='short-short',c=2,repeat=0)])
        for cell in cells:
            current.clear()
            current.update({k:cell[k] for k in ('family','c','repeat')})
            result = base.run_closed_loop(engine,'speculative',requests(engine,**dict(
                family=cell['family'],c=cell['c'],repeat=cell['repeat'])),cell['c'])
            results.append(dict(cell=dict(current),result=result))
            data = dict(calls=calls,mismatches=mismatches,results=results,
                        ranks=engine.model_runner.call('phase53_control','snapshot'))
            (out/'manifest.json').write_text(json.dumps(data,indent=2))
            print('AUDIT_CELL',current,'mismatches',len(mismatches),flush=True)
    finally:
        engine.exit()
    print('DISAGREEMENTS',len(mismatches),flush=True)


if __name__=='__main__':
    main()
