"""Read saved evaluations for post-hoc paper analysis; never optimize or select."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from microgrid.risk import weighted_var_cvar


def cvar_rows(values: np.ndarray, beta: float = 0.9) -> np.ndarray:
    """Equal-weight empirical CVaR, including fractional boundary mass."""
    ordered = np.sort(np.atleast_2d(values), axis=1)[:, ::-1]
    mass = ordered.shape[1] * (1.0 - beta)
    whole = int(np.floor(mass))
    remainder = mass - whole
    result = ordered[:, :whole].sum(axis=1)
    if whole < ordered.shape[1]:
        result += remainder * ordered[:, whole]
    return result / mass


def analyze(root: Path, repetitions: int = 3000) -> dict:
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    root = root.resolve()
    records, summaries, hashes = {}, {}, {}
    models = ['M0', 'M1', 'M2', 'M3', 'M3-R', 'M3-S', 'M3-RS']
    for model in models:
        folder = root / 'evaluation' / model
        if not (folder / 'complete.json').exists():
            raise ValueError(f'{model} is incomplete')
        files = sorted(folder.glob('????-??-??.json'))
        records[model] = [json.loads(p.read_text(encoding='utf-8')) for p in files]
        summaries[model] = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        hashes[model] = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
        costs = np.array([r['emergency_cost'] for r in records[model]])
        expected = summaries[model]['emergency_cvar']
        assert np.isclose(cvar_rows(costs)[0], expected, rtol=0, atol=1e-7)
        assert np.isclose(weighted_var_cvar(costs, np.ones(len(costs))/len(costs), .9)[1], expected)
        assert np.isclose(sum(r['cash_cost'] for r in records[model]), summaries[model]['cash_cost'])

    dates = [r['date'] for r in records['M2']]
    assert len(dates) == 334 and dates[0] == '2025-02-01' and dates[-1] == '2025-12-31'
    assert all([r['date'] for r in records[m]] == dates for m in models)
    metrics = ['cash_cost', 'emergency_cost', 'H']
    arrays = {m: {k: np.array([r[k] for r in records[m]]) for k in metrics} for m in ['M2', 'M3']}
    monthly = []
    for month in sorted({d[:7] for d in dates}):
        mask = np.array([d.startswith(month) for d in dates])
        row = {'month': month, 'days': int(mask.sum())}
        for model in ['M2', 'M3']:
            row[model] = {k: float(arrays[model][k][mask].sum()) for k in metrics}
        row['cost_premium_pct'] = 100*(row['M3']['cash_cost']/row['M2']['cash_cost']-1)
        base_h = row['M2']['H']
        row['emergency_kwh_change_pct'] = 100*(row['M3']['H']/base_h-1) if base_h else None
        monthly.append(row)

    paired = []
    n = len(dates)
    for block in [3, 7, 14]:
        rng = np.random.default_rng(20260912 + block)
        starts = rng.integers(0, n-block+1, size=(repetitions, int(np.ceil(n/block))))
        indices = (starts[..., None] + np.arange(block)).reshape(repetitions, -1)[:, :n]
        cvar2 = cvar_rows(arrays['M2']['emergency_cost'][indices])
        cvar3 = cvar_rows(arrays['M3']['emergency_cost'][indices])
        cash2 = arrays['M2']['cash_cost'][indices].sum(axis=1)
        cash3 = arrays['M3']['cash_cost'][indices].sum(axis=1)
        paired.append({
            'block_days': block, 'repetitions': repetitions,
            'seed': 20260912 + block,
            'cvar_difference_yuan_percentile_95': np.quantile(cvar3-cvar2, [.025,.975]).tolist(),
            'cost_premium_pct_percentile_95': np.quantile(100*(cash3/cash2-1), [.025,.975]).tolist(),
        })

    comparisons = {}
    for before, after in [('M0','M1'), ('M1','M2'), ('M2','M3')]:
        comparisons[f'{before}_to_{after}'] = {
            k: 100*(summaries[after][k]/summaries[before][k]-1)
            for k in ['cash_cost','H','emergency_cvar']
        }
    stress_file = root / 'stress_summary.json'
    stress = json.loads(stress_file.read_text(encoding='utf-8'))
    stress_maps = {
        m: {(r['anchor'], r['shock']['name']): r for r in stress['models'][m]['results']}
        for m in ['M2', 'M3']
    }
    assert stress_maps['M2'].keys() == stress_maps['M3'].keys()
    stress_pairs = {}
    for metric in ['emergency_kwh', 'additional_cash_cost']:
        deltas = []
        for key in sorted(stress_maps['M2']):
            base = stress_maps['M2'][key][metric]
            selected = stress_maps['M3'][key][metric]
            deltas.append({'anchor': key[0], 'shock': key[1], 'M2': base, 'M3': selected,
                           'difference': selected-base})
        stress_pairs[metric] = {
            'improved': sum(d['difference'] < -1e-8 for d in deltas),
            'worsened': sum(d['difference'] > 1e-8 for d in deltas),
            'tied': sum(abs(d['difference']) <= 1e-8 for d in deltas),
            'cases': deltas,
        }
    return {
        'created': datetime.now().isoformat(), 'source': str(root),
        'post_hoc_analysis': True, 'used_for_selection': False, 'optimization_rerun': False,
        'source_daily_json_concatenation_sha256': hashes,
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'stress_summary_sha256': hashlib.sha256(stress_file.read_bytes()).hexdigest(),
        'days': n, 'beta': .9, 'summaries': summaries, 'ablation_changes_pct': comparisons,
        'monthly_M2_M3': monthly, 'paired_block_bootstrap': paired,
        'paired_stress_cases': stress_pairs,
        'bootstrap_interpretation': '两模型使用相同日期块索引，重采样已经保存的日级结果；不重新模拟被拼接的储能轨迹。此为事后、有限样本、对块长敏感的描述性区间。季节非平稳与更长状态依赖未必被短块充分保留，不能当作未来费用或风险保证，也不用于重新选择模型。',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'结果'/'statistical_risk_v1')
    parser.add_argument('--repetitions', type=int, default=3000)
    args = parser.parse_args()
    result = analyze(args.root, args.repetitions)
    output = args.root / 'report' / '论文补充分析.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['ablation_changes_pct','monthly_M2_M3','paired_block_bootstrap']}, ensure_ascii=False, indent=2))
    print(output)
