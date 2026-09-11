"""Read-only audit of formal plans plus explicitly retrospective diagnostics.

Never modifies an official plan, configuration, forecast, or result workbook.
The scalar replay below intentionally does not call q2_controller.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from q2_config import HERE, ATTACHMENTS, load_config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def scalar_replay(G, reference, E0, actual, price, cfg):
    E = np.empty(145); E[0] = E0
    out = {key: np.zeros(144) for key in ('charge', 'discharge', 'H', 'U', 'W')}
    for t in range(144):
        surplus = float(G[t] + actual[t, 1] - actual[t, 0])
        if surplus >= 0:
            C = min(surplus, cfg.power / 6, max(0., (cfg.e_max-E[t])/cfg.eta_c))
            D = H = 0.
            excess = surplus-C
            U = min(float(G[t]), excess); W = excess-U
        else:
            C = U = W = 0.
            D = min(-surplus, cfg.power / 6, cfg.eta_d*max(0., E[t]-reference[t+1]))
            H = -surplus-D
        for k,v in zip(('charge','discharge','H','U','W'), (C,D,H,U,W)):
            out[k][t] = v
        E[t+1] = E[t] + cfg.eta_c*C - D/cfg.eta_d
    out['E'] = E
    out['plan_cost'] = float(sum(float(p)*float(g) for p,g in zip(price,G)))
    out['emergency_cost'] = float(sum(5*float(p)*float(h) for p,h in zip(price,out['H'])))
    out['cash_cost'] = out['plan_cost'] + out['emergency_cost']
    balance = G-out['U']+actual[:,1]-out['W']+out['discharge']+out['H']-actual[:,0]-out['charge']
    assert np.max(np.abs(balance)) < 1e-6
    assert E.min() >= cfg.e_min-1e-6 and E.max() <= cfg.e_max+1e-6
    assert np.max(np.minimum(out['H'],out['charge'])) < 1e-6
    assert np.max(np.minimum(out['discharge'],out['charge'])) < 1e-6
    return out


def audit(root, output, cfg):
    price = pd.read_excel(ATTACHMENTS/'附件1.xlsx')['电价'].to_numpy(float)
    sheets = [pd.read_excel(ATTACHMENTS/'附件2.xlsx', sheet_name=n)
              for n in ('小区负载','光伏发电实际功率')]
    actual = np.stack([s.iloc[:,1:].to_numpy(float)/6 for s in sheets], axis=-1)
    dates = pd.to_datetime(sheets[0].iloc[:,0]).dt.strftime('%Y-%m-%d').tolist()
    assert actual.shape == (365,144,2) and len(price) == 144
    freeze = json.loads((root/'freeze.json').read_text(encoding='utf-8'))
    sums = {}; all_daily = []
    source_hashes = {}
    for p in list(HERE.glob('q2_*.py'))+[HERE/'config_formal_v2.json', root/'freeze.json',
                      ATTACHMENTS/'附件1.xlsx',ATTACHMENTS/'附件2.xlsx',HERE/'result/result2.xlsx']:
        source_hashes[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    max_replay = 0.; max_settlement = 0.
    for model in ('M0','M1','M2','M3'):
        strategy = 'star_'+model
        E0 = greedy_E0 = freeze['shared_E0']
        rows = []; greedy_rows = []; starts = Counter(); stops = Counter(); seed_status = Counter()
        rounds = Counter(); step_counts = Counter(); reserve_emergency = 0.; reserve_slots = 0
        example = None
        for i in range(31,365):
            date = dates[i]
            f = root/'runs'/strategy/(date+'.npz')
            logp = f.with_suffix('.json'); lockedp=root/'locked_plans'/strategy/(date+'.npz')
            for p in (f,logp,lockedp):
                source_hashes[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(f) as z: saved = {k:z[k].copy() for k in z.files}
            with np.load(lockedp) as z:
                G = z['G'].copy(); R = z['reference'].copy()
            np.testing.assert_array_equal(G,saved['G'])
            np.testing.assert_array_equal(R,saved['reference'])
            np.testing.assert_allclose(actual[i,:,0],saved['load'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(actual[i,:,1],saved['pv'],rtol=0,atol=1e-10)
            assert abs(E0-saved['E'][0]) < 1e-6
            out = scalar_replay(G,R,E0,actual[i],price,cfg)
            for k in ('charge','discharge','H','U','W','E'):
                difference=float(np.max(np.abs(out[k]-saved[k]))); max_replay=max(max_replay,difference)
                assert difference < 1e-7
            meta=json.loads(logp.read_text(encoding='utf-8'))
            for k in ('plan_cost','emergency_cost','cash_cost'):
                err=abs(meta['settlement'][k]-out[k]); max_settlement=max(max_settlement,err)
                assert err < 1e-6
            plan=meta['plan']; starts[plan['start']]+=1; stops[plan['stop_reason']]+=1
            seed_status[plan['seed_solver']['status']]+=1
            for s in plan['starts']:
                rounds[len(s['rounds'])]+=1
                for r in s['rounds']: step_counts[str(r['step_kwh'])]+=1
            # Same pre-action state: physically possible extra discharge withheld by R.
            # This is a slot diagnostic, NOT a feasible additive savings estimate.
            gap=np.maximum(actual[i,:,0]-actual[i,:,1]-G,0)
            unrestricted=np.minimum(np.minimum(gap,cfg.B),cfg.eta_d*np.maximum(out['E'][:-1]-cfg.e_min,0))
            withheld=np.maximum(unrestricted-out['discharge'],0)
            reserve_emergency+=float(withheld.sum()); reserve_slots+=int((withheld>1e-6).sum())
            if example is None and np.max(withheld)>100:
                t=int(np.argmax(withheld))
                example=dict(date=date,slot=t,start_minutes=t*10,end_minutes=(t+1)*10,
                             E=float(out['E'][t]),R=float(R[t+1]),net_deficit=float(gap[t]),
                             discharge=float(out['discharge'][t]),H=float(out['H'][t]),
                             extra_discharge_physically_possible=float(withheld[t]))
            row=dict(strategy=strategy,date=date,plan_kwh=float(G.sum()),emergency_kwh=float(out['H'].sum()),
                     plan_cost=out['plan_cost'],emergency_cost=out['emergency_cost'],cash_cost=out['cash_cost'],
                     E0=E0,Eend=float(out['E'][-1]))
            rows.append(row); E0=float(out['E'][-1])
            # Fixed-plan counterfactual, not a replacement formal policy or reoptimized plan.
            greedy=scalar_replay(G,np.full(145,cfg.e_min),greedy_E0,actual[i],price,cfg)
            greedy_E0=float(greedy['E'][-1])
            gr=dict(date=date,cash_cost=greedy['cash_cost'],emergency_cost=greedy['emergency_cost'],
                    emergency_kwh=float(greedy['H'].sum()),Eend=greedy_E0)
            greedy_rows.append(gr)
            np.savez_compressed(output/'diagnostic_traces'/f'{strategy}_{date}.npz',G=G,
                                formal_E=out['E'],formal_H=out['H'],greedy_E=greedy['E'],greedy_H=greedy['H'])
            all_daily.append(dict(**row,greedy_cash=gr['cash_cost'],greedy_emergency_kwh=gr['emergency_kwh']))
        sums[strategy]={k:sum(r[k] for r in rows) for k in ('plan_kwh','emergency_kwh','plan_cost','emergency_cost','cash_cost')}
        sums[strategy].update(days=len(rows),final_E=E0,selected_starts=dict(starts),stop_reasons=dict(stops),
                             rounds_per_start=dict(rounds),attempted_steps_kwh=dict(step_counts),seed_status=dict(seed_status),
                             emergency_withheld_by_reference_kwh=reserve_emergency,reserve_binding_slots=reserve_slots,
                             reserve_example=example,
                             fixed_plans_greedy_diagnostic={k:sum(r[k] for r in greedy_rows) for k in ('cash_cost','emergency_cost','emergency_kwh')})
        sums[strategy]['fixed_plans_greedy_diagnostic']['final_E']=greedy_E0
        print(json.dumps({strategy:sums[strategy]},ensure_ascii=False),flush=True)
    changed=[p for p,h in source_hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed
    report=dict(timestamp=datetime.now().astimezone().isoformat(),raw_shape=list(actual.shape),formal_days=334,
                audited_slots=4*334*144,independent_scalar_implementation=True,max_replay_error=max_replay,
                max_cash_error=max_settlement,source_files_unchanged=True,models=sums,
                old_screenshot=dict(source='User screenshot, matched total in public shared conversation; old raw trajectory unavailable',
                    plan_kwh=21352978.93,emergency_kwh=124459.52,plan_cost=13425684.43,
                    emergency_cost=600405.51,cash_cost=14026089.94),
                diagnostics_only='Greedy uses R=Emin with fixed original plans and carries its own E; no plan reoptimization. Not a new formal run.')
    write_json(output/'accounting_audit.json',report); write_json(output/'source_hashes.json',source_hashes)
    pd.DataFrame(all_daily).to_csv(output/'daily_diagnostics.csv',index=False,encoding='utf-8-sig')
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='result/cost_diagnosis_20260911')
    a=p.parse_args(); output=HERE/a.output; (output/'diagnostic_traces').mkdir(parents=True,exist_ok=True)
    audit(HERE/'result/formal_v2',output,load_config(HERE/'config_formal_v2.json'))


if __name__=='__main__': main()
