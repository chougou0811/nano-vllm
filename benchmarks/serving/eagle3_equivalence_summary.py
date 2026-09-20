"""Aggregate audit evidence without thresholding or removing failed controls."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--unit-log',default='/root/autodl-tmp/eagle11-tests-final.log')
    args = parser.parse_args()
    root,out = Path(args.root),Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    def read(path):
        return json.loads(path.read_text())
    groups = {}
    operator_rows = []
    for name in ['bf16-final','fp32-small-final']:
        records = [read(p) for p in sorted((root/name).glob('case-*/iteration-*-rank-*.json'))]
        commits = [read(p) for p in (root/name).glob('case-*/commit-*.json')]
        top = [d for d in records if 'top1' in d]
        groups[name] = dict(rank_iterations=len(records),commit_checks=len(commits),
            commits_all_pass=all(all(c.values()) for c in commits),
            first_operators=sorted({k for d in records for k in d['first_nonzero']}),
            query_count=sum(len(d['top1']['serial_s']) for d in top),
            all_five_paths_top1_equal=all(len({tuple(v) for v in d['top1'].values()})==1 for d in top),
            serial_parallel_logit_max=max(d['pairs']['serial_s__parallel_s']['logits']['max_abs'] for d in top),
            all_operator_values_finite=all(v['finite'] for d in records for pair in d['pairs'].values() for v in pair.values()))
        for case in [0,1]:
            for path in sorted((root/name/f'case-{case}').glob('iteration-*-rank-0.json')):
                record = read(path)
                for key,value in record['pairs']['serial_s__parallel_s'].items():
                    operator_rows.append(dict(run=name,case=case,iteration=record['iteration'],
                        operator=key,max_abs=value['max_abs'],rms=value['rms'],relative_l2=value['relative_l2']))
        if name == 'bf16-final':
            isolated = [v for d in records for v in d['attention_isolation'].values()]
            groups[name]['fixed_qkv_attention_comparisons'] = len(isolated)
            groups[name]['fixed_qkv_all_three_paths_exact'] = all(
                v[key]['unequal']==0 for v in isolated for key in ['serial_cache_vs_parallel_varlen',
                'serial_varlen_vs_parallel_varlen','parallel_cache_vs_parallel_varlen'])
            groups[name]['q1_decode_varlen_exact'] = all(v['unequal']==0 for d in records
                for v in d['pairs']['serial_s__varlen1_s'].values())
    hf = read(root/'precision-controls-final/hf-small-fp32.json')
    dots = read(root/'precision-controls-final/local-dot-controls.json')
    metadata = read(root/'precision-controls-final/metadata-controls.json')
    frozen = read(root/'bf16-final/frozen-source.json')
    current = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in frozen}
    unit_log = Path(args.unit_log).read_text()
    match = re.search(r'Ran (\d+) tests',unit_log)
    valid = (current==frozen and all(g['commits_all_pass'] and g['all_operator_values_finite'] for g in groups.values())
             and groups['fp32-small-final']['all_five_paths_top1_equal']
             and groups['bf16-final']['fixed_qkv_all_three_paths_exact']
             and groups['bf16-final']['first_operators']==['00.qkv']
             and all(c['top1_equal'] and c['error']['finite'] for c in hf)
             and all(c['positions_slots_lengths_match'] for c in metadata))
    summary = dict(outcome=2 if valid else 3,classification='BF16 shape-dependent GEMV/GEMM rounding and accumulated prefix numerical drift' if valid else 'Audit gate incomplete',
        production_source_unchanged=current==frozen,
        unit_tests_passed=int(match.group(1)) if match and unit_log.rstrip().endswith('OK') else None,runs=groups,
        hf_fp32=dict(passed=sum(r['top1_equal'] for r in hf),total=len(hf),
                     max_abs=max(r['error']['max_abs'] for r in hf)),
        local_dot_controls=dict(count=len(dots),all_inputs_identical=all(d['identical_input'] for d in dots),
            fp32_serial_parallel_max=max(d['fp32_serial_parallel']['max_abs'] for d in dots),
            fp32_fp64_max=max(d['fp32_fp64']['max_abs'] for d in dots),
            disabling_bf16_reduced_reduction_did_not_remove_difference=all(
                d['modes']['False']['serial_parallel']['unequal']>0 for d in dots)),
        metadata_checks=dict(passed=sum(d['positions_slots_lengths_match'] for d in metadata),total=len(metadata)),
        invalid_controls_retained={
            'bf16':'Initial shared-RoPE hook mislabeled repeated captures; superseded by layer-gated bf16-final.',
            'fp32-small':'Unmodified BF16-oriented in-place RMSNorm aliases FP32 residual; HF 0/16. Not valid evidence.',
            'precision-controls':'Initial local dot controls valid; associated HF comparison used invalid FP32 norm.'},
        strict_original_token_parity_restored=False,
        numerical_contract='User accepted docs/EAGLE3_NUMERICAL_CONTRACT.md on 2026-09-20; historical strict-parity failures remain failures.',
        phase1_gate=('numerical semantics accepted under the documented BF16 cross-shape contract; strict serial parity not guaranteed.' if valid else 'Audit gate incomplete'),
        raw_root=str(root))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'layer-operator-summary.csv').open('w') as file:
        writer = csv.DictWriter(file,fieldnames=list(operator_rows[0]))
        writer.writeheader()
        writer.writerows(operator_rows)
    (out/'frozen-source.json').write_text(json.dumps(frozen,indent=2))
    for name in ['local-dot-controls.json','hf-small-fp32.json','metadata-controls.json']:
        (out/name).write_bytes((root/'precision-controls-final'/name).read_bytes())
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
