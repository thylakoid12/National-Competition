"""Restart-safe annual M3 archived-plan sensitivity: extra 5/10/15/20 sweeps.
Optimization fixes original daily inputs and E0. Settlement replays each snapshot
chronologically with continuous storage; this is not online reoptimization.
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from q2_config import HERE, load_config
from q2_controller import rollout
from q2_cost_diagnosis import scalar_replay
from q2_data import load_data
from q2_forecast import combine
from q2_optimizer import Evaluator
from q2_stress import read_issue

MILESTONES = (5, 10, 15, 20)
ROOT = HERE / 'result/formal_v2'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def signature(paths):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def sweep(evaluator, state):
    """Mutate only after completing all candidates, so interrupted sweeps repeat."""
    G = np.array(state['G'])
    value, scale = state['objective'], state['scale']
    step = evaluator.cfg.steps[scale] * evaluator.cfg.B
    best_G, best_value = G, value
    for t in range(len(G)):
        for sign in (1, -1):
            trial = G.copy()
            trial[t] = max(0., trial[t] + sign * step)
            score = evaluator.evaluate(trial, 'M3')[0]
            if score < best_value - evaluator.cfg.objective_tolerance:
                best_G, best_value = trial, score
    improved = best_value < value - evaluator.cfg.objective_tolerance
    row = dict(round=len(state['rounds']) + 1, step_kwh=step,
               attempts=2 * len(G), accepted=improved, objective=best_value)
    state.update(G=best_G.tolist(), objective=best_value,
                 scale=scale if improved else scale + 1)
    state['rounds'].append(row)
    return row


def init_worker():
    global CFG, DATA
    CFG = load_config(HERE / 'config_formal_v2.json')
    DATA = load_data()


def run_date(date, output):
    cfg, data = CFG, DATA
    old = read_json(ROOT / f'runs/star_M3/{date}.json')
    paths = [HERE / 'config_formal_v2.json', ROOT / f'locked_plans/star_M3/{date}.npz',
             ROOT / f'systems/star/{date}.json', ROOT / f'scenarios/star/{date}.npz',
             ROOT / f'scenarios/star/{date}.json', ROOT / f'runs/star_M3/{date}.npz',
             ROOT / f'runs/star_M3/{date}.json']
    hashes = signature(paths)
    forecast = combine(read_issue(ROOT, date), read_json(paths[2])['composition'], 'star')
    with np.load(paths[3]) as z:
        scenes = dict(read_json(paths[4]), **{k: z[k].copy() for k in z.files})
    with np.load(paths[5]) as z:
        E0 = float(z['E'][0])
    with np.load(paths[1]) as z:
        initial_G = z['G'].copy()
    actual = data.values[data.dates.index(date)]
    for name, array in [('forecast', forecast['path']), ('price', data.price), ('actual', actual)]:
        hashes[name] = hashlib.sha256(array.tobytes()).hexdigest()
    checkpoint = output / 'checkpoints' / f'{date}.json'
    if checkpoint.exists():
        state = read_json(checkpoint)
        assert state['source_hashes'] == hashes, 'Source inputs changed; use a new output directory'
        assert state['target_extra_rounds'] == 20 and state['date'] == date
        if state['status'] == 'complete':
            return date
    evaluator = Evaluator(forecast, scenes, E0, data.price, cfg)
    initial_value = evaluator.evaluate(initial_G, 'M3')[0]
    assert abs(initial_value - old['plan']['best_objective']) < 1e-5
    if checkpoint.exists():
        assert abs(evaluator.evaluate(np.array(state['G']), 'M3')[0] - state['objective']) < 1e-5
    else:
        selected = next(s for s in old['plan']['starts'] if s['start'] == old['plan']['start'])
        scale = sum(not r['accepted'] and r['complete_sweep'] for r in selected['rounds'])
        state = dict(date=date, strategy='star_M3', source_hashes=hashes, target_extra_rounds=20,
                     initial_objective=initial_value, initial_cash=old['settlement']['cash_cost'],
                     G=initial_G.tolist(), objective=initial_value, scale=scale, E0=E0,
                     rounds=[], milestones=[], status='running',
                     interpretation='fixed archived daily inputs and E0; annual snapshots settled separately with continuous E')
        atomic_json(checkpoint, state)
    while len(state['rounds']) < 20:
        if state['scale'] < len(cfg.steps):
            row = sweep(evaluator, state)
        else:
            row = dict(round=len(state['rounds']) + 1, step_kwh=0., attempts=0,
                       accepted=False, objective=state['objective'], carried_after_convergence=True)
            state['rounds'].append(row)
        if row['round'] in MILESTONES:
            G = np.array(state['G'])
            value, item, q, Q = evaluator.evaluate(G, 'M3')
            assert abs(value - state['objective']) < 1e-5
            execution = rollout(G, item['reference'], E0, iter(actual), data.price, cfg)
            independent = scalar_replay(G, item['reference'], E0, actual, data.price, cfg)
            assert abs(independent['cash_cost'] - execution['cash_cost']) < 1e-5
            record = dict(extra_round=row['round'],
                          executed_sweeps=sum(r['attempts'] > 0 for r in state['rounds']),
                          converged=state['scale'] == len(cfg.steps),
                          objective=value, objective_change=value - initial_value,
                          fixed_E0_cash_cost=execution['cash_cost'],
                          plan_cost=execution['plan_cost'], fixed_E0_emergency_cost=execution['emergency_cost'],
                          G=G.tolist(), reference=item['reference'].tolist(),
                          E=execution['E'].tolist(), H=execution['H'].tolist(),
                          worst_q=q.tolist(), scenario_costs=Q.tolist(), residuals=execution['residuals'])
            state['milestones'].append(record)
        state['status'] = 'complete' if len(state['rounds']) == 20 else 'running'
        atomic_json(checkpoint, state)
        if row['round'] in MILESTONES:
            print(json.dumps(dict(date=date, **row), ensure_ascii=False), flush=True)
        current_key = evaluator.key(np.array(state['G']))
        evaluator.cache = {current_key: evaluator.cache[current_key]}
    assert signature(paths) == {str(p.resolve()): hashes[str(p.resolve())] for p in paths}
    return date


def export_annual(output, dates, cfg, data):
    states = [read_json(output / 'checkpoints' / f'{d}.json') for d in dates]
    assert all(s['status'] == 'complete' and len(s['milestones']) == 4 for s in states)
    baseline_cash = sum(read_json(ROOT / f'runs/star_M3/{d}.json')['settlement']['cash_cost'] for d in dates)
    summaries, daily = [], []
    for extra in (0,) + MILESTONES:
        E0 = states[0]['E0']
        rows = []
        trajectories = {k: [] for k in ('G', 'reference', 'E', 'H')}
        for date, state in zip(dates, states):
            if extra == 0:
                with np.load(ROOT / f'locked_plans/star_M3/{date}.npz') as z:
                    G, reference = z['G'].copy(), z['reference'].copy()
                objective = state['initial_objective']
                fixed_cash = state['initial_cash']
            else:
                m = next(m for m in state['milestones'] if m['extra_round'] == extra)
                G, reference = np.array(m['G']), np.array(m['reference'])
                objective, fixed_cash = m['objective'], m['fixed_E0_cash_cost']
            actual = data.values[data.dates.index(date)]
            execution = rollout(G, reference, E0, iter(actual), data.price, cfg)
            independent = scalar_replay(G, reference, E0, actual, data.price, cfg)
            assert abs(independent['cash_cost'] - execution['cash_cost']) < 1e-5
            assert np.max(np.abs(independent['E'] - execution['E'])) < 1e-6
            if extra == 0:
                assert abs(execution['cash_cost'] - state['initial_cash']) < 1e-5
                assert abs(E0 - state['E0']) < 1e-6
            row = dict(date=date, extra_round=extra, objective_at_archived_E0=objective,
                       fixed_E0_cash_cost=fixed_cash, cash_cost=execution['cash_cost'],
                       plan_cost=execution['plan_cost'], emergency_cost=execution['emergency_cost'],
                       emergency_kwh=float(execution['H'].sum()), E0=E0, Eend=float(execution['E'][-1]),
                       max_residual=max(execution['residuals'].values()))
            rows.append(row)
            for k in trajectories:
                trajectories[k].append(reference if k == 'reference' else execution[k])
            E0 = float(execution['E'][-1])
        daily.extend(rows)
        total = dict(extra_round=extra, days=len(dates),
                     **{k: sum(r[k] for r in rows) for k in
                        ('objective_at_archived_E0', 'fixed_E0_cash_cost', 'cash_cost', 'plan_cost', 'emergency_cost', 'emergency_kwh')},
                     initial_energy=rows[0]['E0'], final_energy=E0,
                     max_residual=max(r['max_residual'] for r in rows))
        total['cash_change_vs_original'] = total['cash_cost'] - baseline_cash
        total['cash_reduction_pct'] = (baseline_cash - total['cash_cost']) / baseline_cash * 100
        summaries.append(total)
        np.savez_compressed(output / f'annual_stage_{extra:02d}.npz', dates=np.array(dates),
                            **{k: np.stack(v) for k, v in trajectories.items()})
    objectives = [s['objective_at_archived_E0'] for s in summaries]
    assert all(b <= a + 1e-5 for a, b in zip(objectives, objectives[1:]))
    atomic_json(output / 'annual_summary.json', summaries)
    atomic_json(output / 'daily_stage_metrics.json', daily)
    lines = ['# M3 全年断点续跑结果', '',
             f'期间：{dates[0]} 至 {dates[-1]}，共 {len(dates)} 天。第 5/10/15/20 轮均指从正式原计划开始新增的轮次。',
             '每轮扫描 144 个坐标的正负方向，共 288 次候选评价；沿用原计划选中起点的步长进度。若所有步长已无改进，后续阶段沿用收敛解，并记录实际执行轮数。',
             '搜索固定原每日预测、场景和日初电量。优化目标含风险及终端项，不能等同于实际费用；四组全年实际费用按各自跨日连续电量逐日结算，未逐日重置电量。',
             '此实验衡量原计划延长搜索的效果，不是依据新日初电量重新优化的在线全年策略。1 月预热结果保持原值，正式比较期间为 2—12 月。金额单位：元。', '',
             '| 新增轮数 | 原日初状态下目标值合计 | 全年实际购电费 | 计划购电费 | 紧急购电费 | 比原结果降低 |',
             '|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['extra_round']} | {s['objective_at_archived_E0']:.6f} | {s['cash_cost']:.6f} | {s['plan_cost']:.6f} | {s['emergency_cost']:.6f} | {s['cash_reduction_pct']:.4f}% |")
    lines += ['', '逐日四阶段目标值及固定日初状态结算值在 checkpoints 中；连续跨日的逐日费用在 daily_stage_metrics.json 中；完整计划和电量轨迹在 annual_stage_XX.npz 中。',
              '断点文件每完成一轮原子替换并刷新到磁盘。断电/中断后重复运行同一命令，已完成日期自动跳过，未完成的一轮重算。系统关机期间不会计算，重启后需重新启动命令。']
    (output / 'M3全年续跑结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_json(output / 'verification.json', dict(days=len(dates), milestones=list(MILESTONES),
                checkpoints_complete=True, objective_monotonic=True, baseline_cash_reproduced=True,
                continuous_storage=True, independent_scalar_settlement_passed=True,
                max_physical_residual=max(s['max_residual'] for s in summaries)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'result/m3_annual_plus20_20260911')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit-days', type=int, default=0, help='Validation only: first N consecutive dates')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (args.output / 'run.lock').open('a+b')
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as e:
        raise RuntimeError('Another process owns this output directory') from e
    cfg, data = load_config(HERE / 'config_formal_v2.json'), load_data()
    dates = sorted(p.stem for p in (ROOT / 'locked_plans/star_M3').glob('*.npz'))
    assert dates == list(data.dates[31:]) and len(dates) == 334
    if args.limit_days:
        dates = dates[:args.limit_days]
    source_paths = list(HERE.glob('q2_*.py')) + [HERE / 'config_formal_v2.json',
                   ROOT / 'checkpoint.pkl', HERE / 'result/result2.xlsx']
    manifest = dict(dates=dates, milestone_rounds=list(MILESTONES), source_hashes=signature(source_paths))
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        assert read_json(manifest_path) == manifest, 'Experiment source changed; choose a new output directory'
    else:
        atomic_json(manifest_path, manifest)
    progress = dict(status='running', total_days=len(dates), completed_days=0)
    atomic_json(args.output / 'progress.json', progress)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
            futures = [pool.submit(run_date, date, args.output) for date in dates]
            for future in as_completed(futures):
                date = future.result()
                progress.update(completed_days=progress['completed_days'] + 1, last_completed_date=date)
                atomic_json(args.output / 'progress.json', progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
        export_annual(args.output, dates, cfg, data)
        assert signature(source_paths) == manifest['source_hashes'], 'Original files changed during run'
        progress['status'] = 'complete'
        atomic_json(args.output / 'progress.json', progress)
    except BaseException as e:
        progress.update(status='failed_or_interrupted', error=repr(e))
        atomic_json(args.output / 'progress.json', progress)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()

