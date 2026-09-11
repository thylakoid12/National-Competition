"""Locked-plan stress branches; only already-realized shocks enter later history."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,hashlib
import numpy as np
from q2_config import load_config
from q2_data import load_data
from q2_forecast import Forecaster,combine
from q2_scenarios import build_scenarios
from q2_controller import rollout
from q2_optimizer import optimize_plan
from q2_archive import save_json,save_npz,save_issue,save_rollout,save_scenarios
from q2_backtest import STRATEGIES,daily_row
from q2_metrics import summarize

def read_issue(root,date):
    issued=[]
    for system in ['F0','F1']:
        meta=json.loads((root/'forecasts'/system/(date+'.json')).read_text(encoding='utf-8'))
        with np.load(root/'forecasts'/system/(date+'.npz')) as f:
            issued.append(dict(meta=meta,path=f['path'].copy(),context=f['context'].copy()))
    with np.load(root/'candidates'/(date+'.npz')) as f:
        arrays=(f['load'].copy(),f['pv'].copy())
    return dict(date=date,issued=issued,candidates=arrays)

def run_stress(normal_root,output,cfg):
    normal_root,output=Path(normal_root),Path(output)
    if output.resolve()==normal_root.resolve() or normal_root.resolve() in output.resolve().parents:
        raise ValueError('Stress outputs must be outside the normal run directory')
    data=load_data()
    start=data.dates.index(cfg.stress_start)
    engine=Forecaster(cfg)
    for i in range(1,start):
        engine.settle(read_issue(normal_root,data.dates[i]),data.values[i])
    engine.freeze()
    signature=hashlib.sha256((normal_root/'checkpoint.pkl').read_bytes()).hexdigest()
    # Magnitudes fixed before injection. A fluctuation, not a claimed event probability.
    sigma=data.history(start)[-60:].std(axis=0)
    for shock in ['load_up','pv_down','joint_multiday']:
        import copy
        branch=output/shock
        local=copy.deepcopy(engine)
        history=data.history(start).copy()
        days=cfg.stress_days if shock=='joint_multiday' else 1
        states={}
        rows={s:[] for s in STRATEGIES}
        save_json(branch/'design.json',dict(config=asdict(cfg),shock=shock,
                  start=cfg.stress_start,days=days,load_window='18:00-21:00',pv_window='10:00-16:00',
                  magnitude='pre-start 60-day slot standard deviation times gamma',normal_checkpoint_hash=signature))
        save_npz(branch/'magnitude.npz',sigma=sigma)
        for offset in range(days):
            i=start+offset
            date=data.dates[i]
            issued=read_issue(normal_root,date) if offset==0 else local.issue(history,date)
            save_issue(branch,issued)
            plans={}
            for system,composition in [('star',local.composition),('dagger',[1-x for x in local.composition])]:
                f=combine(issued,composition,system)
                s=build_scenarios(f,local.records,composition,cfg)
                save_scenarios(branch,system,date,s)
                for strategy in [v for v in STRATEGIES if v.startswith(system+'_')]:
                    if offset==0:
                        with np.load(normal_root/'runs'/strategy/(date+'.npz')) as r:
                            states[strategy]=float(r['E'][0])
                            plan=dict(G=r['G'].copy(),reference=r['reference'].copy(),
                                      Q=r['scenario_costs'].copy(),worst_q=r['worst_q'].copy())
                        log=json.loads((normal_root/'runs'/strategy/(date+'.json')).read_text(encoding='utf-8'))['plan']
                        plan.update(log=log,objective=log['best_objective'])
                    else:
                        plan=optimize_plan(strategy.split('_')[1],f,s,states[strategy],data.price,cfg)
                    plans[strategy]=plan
                    save_npz(branch/'locked_plans'/strategy/(date+'.npz'),G=plan['G'],reference=plan['reference'])
            # Inject only AFTER every strategy's day-ahead plan is locked.
            actual=data.values[i].copy()
            if shock in ['load_up','joint_multiday']:
                a,b=cfg.stress_load_slots
                actual[a:b,0]+=cfg.stress_gamma*sigma[a:b,0]
            if shock in ['pv_down','joint_multiday']:
                a,b=cfg.stress_pv_slots
                actual[a:b,1]=np.maximum(0,actual[a:b,1]-cfg.stress_gamma*sigma[a:b,1])
            for strategy,plan in plans.items():
                o=rollout(plan['G'],plan['reference'],states[strategy],iter(actual),data.price,cfg)
                save_rollout(branch,strategy,date,plan,o)
                states[strategy]=float(o['E'][-1])
                row=daily_row(date,o,plan)
                normal=json.loads((normal_root/'runs'/strategy/(date+'.json')).read_text(encoding='utf-8'))['settlement']
                row['cash_change_vs_normal']=o['cash_cost']-normal['cash_cost']
                row['emergency_fee_change_vs_normal']=o['emergency_cost']-normal['emergency_cost']
                rows[strategy].append(row)
            record,_=local.settle(issued,actual)
            save_npz(branch/'residuals'/(date+'.npz'),residual=record['residual'],context=record['context'])
            history=np.concatenate([history,actual[None]])
        save_json(branch/'summary.json',dict(rows=rows,totals={k:summarize(v) for k,v in rows.items()}))
    assert signature==hashlib.sha256((normal_root/'checkpoint.pkl').read_bytes()).hexdigest()
    save_json(output/'isolation_check.json',dict(normal_checkpoint_unchanged=True,hash=signature))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--normal',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config',required=True)
    a=p.parse_args()
    run_stress(a.normal,a.output,load_config(a.config))

if __name__=='__main__':
    main()
