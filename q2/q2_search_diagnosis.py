"""Continue archived plans on four fixed dates to diagnose search truncation.

Uses the same 144-dimensional coordinate best-improvement rule and fixed inputs.
This is an ex-post sensitivity experiment, not a new annual result or tuning.
Actual observations are read for settlement only AFTER each diagnostic plan locks.
"""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from q2_config import HERE,load_config
from q2_optimizer import Evaluator
from q2_forecast import combine
from q2_stress import read_issue
from q2_controller import rollout
from q2_data import load_data
from q2_cost_diagnosis import write_json


def continuation(evaluator, G, model, max_rounds=20):
    G=G.copy(); value,_,_,_=evaluator.evaluate(G,model)
    initial=value; rounds=[]; scale=0; evaluations=1
    for r in range(max_rounds):
        step=evaluator.cfg.steps[scale]*evaluator.cfg.B
        best=(value,G); attempts=0
        for t in range(144):
            for sign in (1,-1):
                trial=G.copy(); trial[t]=max(0.,trial[t]+sign*step)
                v,_,_,_=evaluator.evaluate(trial,model); evaluations+=1; attempts+=1
                if v<best[0]-evaluator.cfg.objective_tolerance: best=(v,trial)
        improved=best[0]<value-evaluator.cfg.objective_tolerance
        value,G=best
        rounds.append(dict(round=r+1,step_kwh=step,attempts=attempts,accepted=improved,objective=value))
        if not improved:
            scale+=1
            if scale==len(evaluator.cfg.steps): break
    value,item,_,_=evaluator.evaluate(G,model)
    return dict(G=G,reference=item['reference'],initial=initial,objective=value,rounds=rounds,evaluations=evaluations,
                stop='all_scales_no_improvement' if scale==len(evaluator.cfg.steps) else 'diagnostic_round_limit')


def main():
    cfg=load_config(HERE/'config_formal_v2.json'); root=HERE/'result/formal_v2'
    output=HERE/'result/cost_diagnosis_20260911'; rows=[]
    data=load_data()
    for date in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
        for model in ('M0','M1','M2','M3'):
            strategy='star_'+model; start=perf_counter()
            meta=json.loads((root/'systems/star'/(date+'.json')).read_text(encoding='utf-8'))
            f=combine(read_issue(root,date),meta['composition'],'star')
            sm=json.loads((root/'scenarios/star'/(date+'.json')).read_text(encoding='utf-8'))
            with np.load(root/'scenarios/star'/(date+'.npz')) as z: scenes=dict(sm,**{k:z[k].copy() for k in z.files})
            with np.load(root/'runs'/strategy/(date+'.npz')) as z: E0=float(z['E'][0])
            with np.load(root/'locked_plans'/strategy/(date+'.npz')) as z: G=z['G'].copy()
            old=json.loads((root/'runs'/strategy/(date+'.json')).read_text(encoding='utf-8'))
            evaluator=Evaluator(f,scenes,E0,data.price,cfg)
            new=continuation(evaluator,G,model)
            assert abs(new['initial']-old['plan']['best_objective'])<1e-5
            assert new['objective']<=new['initial']+1e-7
            # Plan is locked above; true path used exclusively in settlement.
            actual=data.values[data.dates.index(date)]
            execution=rollout(new['G'],new['reference'],E0,iter(actual),data.price,cfg)
            row=dict(date=date,strategy=strategy,E0=E0,old_objective=new['initial'],new_objective=new['objective'],
                     objective_change=new['objective']-new['initial'],old_cash=old['settlement']['cash_cost'],
                     new_cash=execution['cash_cost'],cash_change=execution['cash_cost']-old['settlement']['cash_cost'],
                     changed_slots=int((np.abs(G-new['G'])>1e-7).sum()),extra_evaluations=new['evaluations'],
                     seconds=perf_counter()-start,rounds=new['rounds'],stop=new['stop'],
                     retrospective_diagnostic_not_formal=True,original_initial_state_fixed=True)
            rows.append(row)
            np.savez_compressed(output/f'search_{strategy}_{date}.npz',G=new['G'],reference=new['reference'],
                                E=execution['E'],H=execution['H'])
            write_json(output/'search_continuation.json',rows)
            print(json.dumps({k:v for k,v in row.items() if k!='rounds'},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
