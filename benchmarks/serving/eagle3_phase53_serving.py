"""Phase5.3 additive serving collector; frozen target and scheduler algorithms."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
import traceback

import torch
import torch.distributed as dist

import benchmarks.serving.eagle3_phase42 as base
from benchmarks.serving.eagle3_phase51 import heldout
from benchmarks.serving.eagle3_phase53 import ROOT, GROUPS
import nanovllm.speculative.concurrent_engine as engine_module
from nanovllm.speculative.draft_batch import DraftBatchExecutor
from nanovllm.speculative.acceptance import accept_greedy


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class ServingRunner(engine_module.ConcurrentModelRunner):
    def phase53_control(self, operation):
        if operation == 'reset':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        row = dict(rank=self.rank, targets=len(getattr(self, '_eagle_batch_states', {})),
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            free=torch.cuda.mem_get_info()[0], kv_blocks=self.config.num_kvcache_blocks)
        if operation == 'audit':
            row['states'] = []
            for state in getattr(self, '_eagle_batch_states', {}).values():
                positions = torch.arange(state['cursor'], device=self.kv_cache.device)
                blocks = torch.tensor(state['blocks'], device=positions.device, dtype=torch.long)
                kv = self.kv_cache[:, :, blocks[positions//self.block_size], positions%self.block_size]
                row['states'].append(dict(cursor=state['cursor'], tokens=state['tokens'],
                    kv=digest(kv), features=digest(state['features']), tentative=state['tentative'] is not None))
        ranks = [None]*self.world_size
        dist.all_gather_object(ranks, row)
        return ranks if self.rank == 0 else None


class Collector(base.TimingCollector):
    def _install(self):
        super()._install()
        coordinator = self.engine.speculative_coordinator
        executor = getattr(coordinator, 'draft_batch_executor', None)
        if executor:
            original = self._replace(executor, 'propose', None)
            def propose(items, _original=original):
                start = perf_counter_ns()
                result = _original(items)
                self._add('draft_ns', perf_counter_ns()-start)
                m = dict(executor.last_metrics)
                self._add('draft_catchup_ns', m['conditioning_ns'])
                self._add('draft_forward_count', m['draft_forwards'])
                self._add('draft_tokens_processed', sum(i.state.last_metrics['draft_tokens_processed'] for i in items))
                self.current.setdefault('draft_batches', []).append(m)
                return result
            executor.propose = propose
        if getattr(self.engine, 'phase53_audit', False):
            original = self._replace(self.runner, 'call', None)
            def call(method, *args, _original=original):
                result = _original(method, *args)
                if method == 'eagle3_batch' and args[0] == 'commit':
                    ranks = _original('phase53_control', 'audit')
                    drafts = []
                    for state in coordinator.drafts.values():
                        drafts.append(dict(cursor=state.cursor, tokens=state.conditioned_tokens,
                            kv=[digest(t) for layer in (state.past or ()) for t in layer]))
                    self.current['state_audit'] = dict(target=[r['states'] for r in ranks], draft=drafts)
                return result
            self.runner.call = call

    def finish_step(self, start, end):
        last = self.engine.speculative_coordinator.last_step
        if last.get('kind') == 'decode':
            self.current['commits'] = [{k:v for k,v in asdict(c).items() if k not in ('seq_id',)}
                                      for c in last['entries']]
            self.current['serial_fallback_count'] = (0 if self.current.get('draft_batches') else
                                                     sum(c.actual_k > 0 for c in last['entries']))
        return super().finish_step(start, end)


def requests(engine, family, c, repeat, warmup=False):
    if family in base.WORKLOADS:
        return base.build_requests(engine.tokenizer, family, c, repeat, warmup)
    if family == 'heldout-mixed':
        rows = heldout(engine.tokenizer, c, repeat)
    elif family.startswith('boundary-'):
        lengths = GROUPS[int(family.split('-')[-1])]
        stream = engine.tokenizer.encode('A scientist describes recorded observations. '*400, add_special_tokens=False)
        rows = [base.Request(i, ([2100+i]+stream)[:n], (9,13,17,7)[i]) for i,n in enumerate(lengths)]
    else:
        plan = json.loads(Path('benchmarks/eagle3-phase5_3/workload-plan.json').read_text())
        cell = next(r for r in plan['new_families'] if r['name'] == family)
        rows = []
        for i in range(max(4,2*c)):
            length = cell['contexts'][i%2]
            tail = engine.tokenizer.encode('\nTask: '+cell['prompt']+'\nAnswer:', add_special_tokens=False)
            filler = engine.tokenizer.encode('Reference record: the instrument is checked before use. '*300, add_special_tokens=False)
            prompt = [12000+repeat*19+i]+filler[:length-len(tail)-1]+tail
            assert len(prompt) == length
            rows.append(base.Request(i,prompt,cell['output_limits'][i%2]))
    if warmup:
        for r in rows:
            r.output_limit = min(r.output_limit, 6)
    return rows


def signature(run):
    return dict(outputs=[r['output_token_ids'] for r in run['requests']],
        commits=[s.get('commits') for s in run['steps'] if not s['is_prefill']],
        states=[s['state_audit'] for s in run['steps'] if 'state_audit' in s])


def acceptance_check(run, eos):
    limits = {r['sequence_id']:r['output_limit'] for r in run['requests']}
    emitted = {seq_id:1 for seq_id in limits}
    for step in run['steps']:
        for seq_id,entry in zip(step['selected_sequence_ids'],step.get('commits',[])):
            expected = accept_greedy(entry['proposed_token_ids'],entry['target_token_ids'],
                                    limits[seq_id]-emitted[seq_id],eos)
            assert expected.accepted == entry['accepted_length']
            assert expected.tokens == list(entry['committed_token_ids'])
            assert expected.fallback == entry['fallback_token'] and expected.finished == entry['finished']
            assert entry['new_cursor'] == entry['old_cursor']+1+expected.accepted
            emitted[seq_id] += len(expected.tokens)
    return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage', choices=('correctness', 'serving'), required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--families', nargs='+')
    parser.add_argument('--concurrencies', nargs='+', type=int, default=[1,2,4])
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--record-draft-variation', action='store_true',
                        help='Keep proposal differences as pending diagnostics; output mismatch still stops')
    args = parser.parse_args()
    out = ROOT/args.name
    out.mkdir(exist_ok=False)
    base.TimingCollector = Collector
    engine_module.ConcurrentModelRunner = ServingRunner
    config = dict(tensor_parallel_size=2, enforce_eager=True, max_model_len=2048,
        max_num_batched_tokens=2048, max_num_seqs=4, gpu_memory_utilization=.7,
        scheduler_policy='original')
    data = dict(command=[sys.executable,*sys.argv], environment={k:os.environ.get(k) for k in
        ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM','NCCL_DEBUG','TORCH_DISABLE_ADDR2LINE')},
        config=config, gpu=subprocess.check_output(['nvidia-smi'], text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'], text=True),
        versions=dict(torch=torch.__version__,cuda=torch.version.cuda),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('nanovllm','benchmarks/serving','tests') for p in Path(folder).rglob('*.py')},
        target=base.TARGET, draft=base.DRAFT, reference=base.REFERENCE,
        target_revision='40c069824f4251a91eefaf281ebe4c544efd3e18',
        draft_revision='3d13517724e81cb409ddf1d4650772ec52f1e18e',
        git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        warmups=[], trials=[], checks=[], normal_exit=False)
    def save():
        (out/'manifest.json').write_text(json.dumps(data, indent=2))
    save()
    engine = None
    try:
        engine = engine_module.ConcurrentLLMEngine(base.TARGET, draft_path=base.DRAFT,
            reference_path=base.REFERENCE, audit=args.stage=='correctness', **config)
        engine.phase53_audit = args.stage == 'correctness'
        coordinator = engine.speculative_coordinator
        executor = DraftBatchExecutor()
        cells = ([(f'boundary-{i}',len(lengths)) for i,lengths in enumerate(GROUPS)]
                 if args.stage=='correctness' else
                 [(f,c) for f in (args.families or [*base.WORKLOADS,'heldout-mixed',
                    'copy-pattern','observatory-prose','routing-table']) for c in args.concurrencies])
        for index,(family,c) in enumerate(cells):
            repeats = 1 if args.stage == 'correctness' else args.repeats
            for mode in ('serial','batch'):
                coordinator.draft_batch_executor = executor if mode=='batch' else None
                warm = base.run_closed_loop(engine,'speculative',requests(engine,family,c,100,True),c)
                data['warmups'].append(dict(family=family,c=c,mode=mode,result=warm))
            for repeat in range(repeats):
                pair = []
                order = ('serial','batch') if (index+repeat+int(args.reverse))%2 == 0 else ('batch','serial')
                for mode in order:
                    coordinator.draft_batch_executor = executor if mode=='batch' else None
                    engine.model_runner.call('phase53_control','reset')
                    result = base.run_closed_loop(engine,'speculative',requests(engine,family,c,repeat),c)
                    result.update(family=family,c=c,mode=mode,repeat=repeat,
                        rank_memory=engine.model_runner.call('phase53_control','snapshot'))
                    data['trials'].append(result)
                    pair.append(result)
                    save()
                    print('TRIAL',family,c,repeat,mode,result['summary']['duration_s'],flush=True)
                equal = signature(pair[0]) == signature(pair[1])
                outputs_equal = signature(pair[0])['outputs'] == signature(pair[1])['outputs']
                semantics = all(acceptance_check(r,engine.model_runner.config.eos) for r in pair)
                data['checks'].append(dict(family=family,c=c,repeat=repeat,exact_signature=equal,
                    outputs_equal=outputs_equal,acceptance_semantics=semantics,
                    proposal_diagnostics_pending=not equal))
                save()
                if not outputs_equal or (not equal and not args.record_draft_variation):
                    raise RuntimeError('New serial/batch disagreement: retain both trials and diagnose before proceeding')
                if any(any(r['summary']['cleanup'].values()) or any(m['targets'] for m in r['rank_memory']) for r in pair):
                    raise RuntimeError('Serving state leak')
        data['normal_exit'] = True
    except Exception:
        data['error'] = traceback.format_exc()
        raise
    finally:
        if engine is not None:
            engine.exit()
        save()


if __name__ == '__main__':
    main()
