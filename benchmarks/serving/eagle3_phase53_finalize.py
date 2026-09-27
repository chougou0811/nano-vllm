"""Freeze Phase5.3 evidence and verify protected files; never commits Git."""
import difflib
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

from benchmarks.serving.eagle3_phase53 import ROOT, TARGET
from benchmarks.serving.eagle3_phase53_analysis import OUT
from benchmarks.serving.eagle3_phase53_serving import signature, acceptance_check


def main():
    summary = json.loads((OUT/'summary.json').read_text())
    baseline = json.loads((ROOT/'baseline.json').read_text())
    allowed = {'nanovllm/speculative/coordinator.py','nanovllm/speculative/concurrent_engine.py'}
    unexpected = [p for p,h in baseline['hashes'].items()
        if 'phase5_3' not in p and (not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h)
        and p not in allowed]
    if unexpected:
        raise RuntimeError(f'Protected source changed: {unexpected}')
    test = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-q'],text=True,capture_output=True)
    (ROOT/'post-cpu-tests.log').write_text(test.stdout+test.stderr)
    if test.returncode:
        raise RuntimeError(test.stderr)
    count = int(re.search(r'Ran (\d+) tests',test.stderr)[1])
    sources = sorted([Path(p) for p in allowed]+[Path('nanovllm/speculative/draft_batch.py')]+
        list(Path('tests').glob('test_eagle3_draft_batch.py'))+list(Path('tests').glob('test_eagle3_phase53*.py'))+
        list(Path('benchmarks/serving').glob('eagle3_phase53*.py'))+[Path('docs/EAGLE3_PHASE5_3.md')])
    diff = []
    with tarfile.open(ROOT/'baseline-source.tar.gz','r:gz') as archive:
        for p in sources:
            try:
                old = archive.extractfile(str(p)).read().decode()
            except KeyError:
                old = ''
            diff.extend(difflib.unified_diff(old.splitlines(True),p.read_text().splitlines(True),
                        fromfile=f'baseline/{p}',tofile=f'phase5.3/{p}'))
    (OUT/'phase5.3-only.diff').write_text(''.join(diff))
    with tarfile.open(ROOT/'phase53-source-final.tar.gz','w:gz') as archive:
        for p in sources:
            archive.add(p)
        archive.add(OUT/'workload-plan.json')
    main_run = json.loads((ROOT/'serving-main-r3/manifest.json').read_text())
    fresh = json.loads((ROOT/'serving-fresh-r3/manifest.json').read_text())
    index = {(r['family'],r['c'],r['repeat'],r['mode']):r for r in main_run['trials']}
    cross = []
    for r in fresh['trials']:
        key = r['family'],r['c'],r['repeat'],r['mode']
        a,b = signature(index[key]),signature(r)
        cross.append(dict(family=key[0],c=key[1],repeat=key[2],mode=key[3],
                          outputs_equal=a['outputs']==b['outputs'],full_signature_equal=a==b))
    diagnostics = {}
    for name in ('serving-audit','serving-audit-numerics'):
        p = ROOT/name/('manifest.json' if name=='serving-audit' else 'summary.json')
        if not p.exists():
            diagnostics[name] = dict(complete=False)
            continue
        data = json.loads(p.read_text())
        if name=='serving-audit':
            expected = [r for r in main_run['checks'] if not r['exact_signature']]
            eos = json.loads((Path(TARGET)/'config.json').read_text())['eos_token_id']
            replay_checks = []
            for row in data['results']:
                cell, result = row['cell'], row['result']
                reference = index[cell['family'],cell['c'],cell['repeat'],'batch']
                replay_checks.append(dict(**cell,
                    outputs_equal=signature(reference)['outputs']==signature(result)['outputs'],
                    acceptance_semantics=acceptance_check(result,eos)))
            diagnostics[name] = dict(complete=len(data['results'])==len(expected),
                expected_cells=len(expected),replayed_cells=len(data['results']),
                checked_groups=len(data['calls']),local_disagreements=len(data['mismatches']),
                cleanup_zero=all(not any(r['result']['summary']['cleanup'].values()) for r in data['results']),
                cells_with_local_disagreement=len({json.dumps(r['cell'],sort_keys=True) for r in data['mismatches']}),
                replay_checks=replay_checks,
                raw_manifest=str(p))
        else:
            hp = [r for r in data if r['dtype']=='torch.float32']
            bf = [r for r in data if r['dtype']=='torch.bfloat16']
            expected = len(list((ROOT/'serving-audit').glob('*.pt')))
            diagnostics[name] = dict(complete=len(hp)==len(bf)==expected,expected_groups=expected,
                bf16_groups=len(bf),fp32_groups=len(hp),
                fp32_proposals_equal=sum(r['comparison']['proposals_equal'] for r in hp),
                bf16_proposals_equal=sum(r['comparison']['proposals_equal'] for r in bf),
                finite=all(v['finite'] for r in data for row in r['comparison']['rows']
                           for v in [row['hidden'],row['logits'],*row['kv']]),
                teacher_forced={dtype:dict(
                    padding_nonzero_groups=sum(any(op['native_vs_padded']['max_abs']>0
                        for op in r['diagnostic']['operators']) for r in rows),
                    first_observed_batch_operator=dict(Counter(next((op['operator']
                        for op in r['diagnostic']['operators'] if op['padded_vs_batch']['max_abs']>0),
                        'none') for r in rows)),
                    padding_max_abs=max(op['native_vs_padded']['max_abs'] for r in rows
                                        for op in r['diagnostic']['operators']),
                    batch_max_abs=max(op['padded_vs_batch']['max_abs'] for r in rows
                                      for op in r['diagnostic']['operators']),
                    finite=all(op[path]['finite'] for r in rows for op in r['diagnostic']['operators']
                               for path in ('native_vs_padded','padded_vs_batch')))
                    for dtype,rows in (('bf16',bf),('fp32',hp))},
                raw_summary=str(p))
    summary.update(validation=dict(cpu_tests=count,cpu_passed=True,protected_unexpected_changes=unexpected,
        historical_tests_unchanged=True,git_commit_created=False),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        workload_plan_sha256=hashlib.sha256((OUT/'workload-plan.json').read_bytes()).hexdigest(),
        cross_process_checks=cross,numerical_diagnostics=diagnostics,
        software={n:importlib.metadata.version(n) for n in ('torch','transformers','triton','flash-attn')})
    summary['decision'] = dict(status='Keep Experimental',default_draft_batching=False,
        numerical_contract='Strict state and actual greedy acceptance; no universal BF16 cross-shape proposal parity',
        qualification_limits=['Unresolved cross-process proposal-trajectory variation, including serial baseline',
                              'Main c1 family-level latency regressions retained',
                              'Target verification wall reduction not causally isolated'],
        correctness_state_tests_passed=True,universal_proposal_parity=False,
        adoption_all_gates_passed=False,secondary_implemented=False)
    summary['measurement_notes'] = dict(
        serving_warmup='One short-output run per system/cell; not exhaustive shape warmup',
        isolated_warmup='Two generation calls per system/cell',
        throughput='Committed output tokens divided by measurement wall time',
        headline_speedup='Geometric mean of paired serial wall / batch wall',
        itl='Committed burst timestamps; intra-burst gaps can be zero',
        timing_decomposition='Host submission spans overlap device execution; not GPU service times',
        incomplete_runs='Retained separately, not pooled into completed five-repeat cells',
        c1_tail_check='Post-hoc targeted diagnostic; original negative observations retained')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'commands.md').write_text('# Executed Commands\n\nWorking directory: `/root/autodl-tmp/projects/nano-vllm`.\nRaw root: `'+str(ROOT)+'`. Never overwrite a run directory.\n\n'+
        '\n\n'.join('```text\n'+' '.join(json.loads(p.read_text())['command'])+'\n```'
                    for p in sorted(ROOT.glob('*/manifest.json')) if 'command' in json.loads(p.read_text()))+
        '\n\n## Additional Diagnostics and Reporting\n\n'+
        'GPU commands used HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false '+
        'NCCL_DEBUG=WARN TORCH_DISABLE_ADDR2LINE=1. Full source/environment hashes are in manifests.\n\n'+
        '\n\n'.join('```text\n'+sys.executable+' -m '+command+'\n```' for command in (
            'benchmarks.serving.eagle3_phase53_numerics',
            'benchmarks.serving.eagle3_phase53_edges',
            'benchmarks.serving.eagle3_phase53_disagreement',
            'benchmarks.serving.eagle3_phase53_numerics --disagreement',
            'benchmarks.serving.eagle3_phase53_disagreement --name serving-audit --manifest '+str(ROOT/'serving-main-r3/manifest.json'),
            'benchmarks.serving.eagle3_phase53_numerics --disagreement --input-name serving-audit --output-name serving-audit-numerics',
            'benchmarks.serving.eagle3_phase53_analysis',
            'benchmarks.serving.eagle3_phase53_tables',
            'benchmarks.serving.eagle3_phase53_finalize',
            'unittest discover -s tests -q'))+'\n')
    print(json.dumps(dict(validation=summary['validation'],diagnostics=diagnostics,
                         cross_process_output_equal=all(r['outputs_equal'] for r in cross),
                         cross_process_signature_changes=sum(not r['full_signature_equal'] for r in cross)),indent=2))


if __name__=='__main__':
    main()
