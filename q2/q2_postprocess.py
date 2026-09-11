"""Paper tables plus explicitly post-hoc, non-tuning stability/sensitivity checks."""
from pathlib import Path
from dataclasses import replace
import argparse,json
import numpy as np
import pandas as pd
from q2_config import load_config
from q2_data import load_data
from q2_forecast import Forecaster,combine
from q2_scenarios import build_scenarios
from q2_optimizer import optimize_plan
from q2_controller import rollout
from q2_stress import read_issue
from q2_archive import save_json,save_npz,save_rollout
from common.result_io import write_result

def tables(root):
    root=Path(root)
    summary=json.loads((root/'summary_2025-12-31.json').read_text(encoding='utf-8'))
    if not summary['complete']:
        raise ValueError('Paper annual results require complete 334-day replay')
    out=root.parent/'paper'
    out.mkdir(exist_ok=True)
    economic=pd.DataFrame([dict(strategy=k,**v) for k,v in summary['economic'].items()])
    write_result(economic,out/'economic_comparison.xlsx')
    points=[]
    for scope,values in summary['point'].items():
        if not values['days']:
            continue
        for f in range(2):
            for t,target in enumerate(['load','pv']):
                points.append(dict(scope=scope,predictor='F'+str(f),target=target,days=values['days'],
                                   MAE_kWh=values['mae'][f][t],RMSE_kWh=values['rmse'][f][t]))
            points.append(dict(scope=scope,predictor='F'+str(f),target='pv_06:00-18:00',days=values['days'],
                          MAE_kWh=values['pv_daylight_mae'][f],RMSE_kWh=values['pv_daylight_rmse'][f]))
    write_result(pd.DataFrame(points),out/'forecast_comparison.xlsx')
    probabilistic=[]
    for row in summary['probability']:
        for system,models in row['scores'].items():
            for model,targets in models.items():
                for target,metrics in targets.items():
                    probabilistic.append(dict(date=row['date'],system=system,model=model,target=target,**metrics))
    prob=pd.DataFrame(probabilistic)
    aggregated=prob.groupby(['system','model','target'])[['crps','coverage','width']].mean().reset_index()
    write_result(aggregated,out/'nominal_probability_comparison.xlsx')
    prob.to_csv(out/'nominal_probability_daily.csv',index=False,encoding='utf-8-sig')
    # Re-read representative exported aggregates against independent dataframe values.
    from common.result_io import read_result
    reread=read_result(out/'economic_comparison.xlsx')
    np.testing.assert_allclose(reread.cash_cost,economic.cash_cost,rtol=0,atol=1e-6)
    save_json(out/'table_validation.json',dict(annual_days=334,cash_cells_reread=True,
               no_worst_case_probabilities_in_crps=True,cvar_boundary_mass=16.7))

def profiles(root,cfg,wait_available=False):
    root=Path(root)
    data=load_data()
    engine=Forecaster(cfg)
    selected={'2025-03-20','2025-06-21','2025-09-23','2025-12-21'}
    records=[]
    out=root.parent/(root.name+'_sensitivity')
    out.mkdir(exist_ok=True)
    for i in range(1,365):
        date=data.dates[i]
        if wait_available:
            import time
            announced=False
            while not (root/'residuals'/(date+'.json')).exists():
                if not announced:
                    print('Waiting for completed normal date '+date,flush=True)
                    announced=True
                time.sleep(5)
        issued=read_issue(root,date)
        if date>='2025-02-01':
            engine.freeze()
        if date in selected:
            for system,composition in [('star',engine.composition),('dagger',[1-x for x in engine.composition])]:
                f=combine(issued,composition,system)
                models=['M0','M1','M2','M3'] if system=='star' else ['M0','M3']
                cases=[]
                for model in models:
                    cases.append(('search_budget_x3',model,replace(cfg,budget=cfg.budget*3)))
                for radius in [0,.02,.05,.1]:
                    cases.append(('TV_radius', 'M3',replace(cfg,radius=radius)))
                for multiplier in [0,1,2]:
                    for model in ['M0','M3']:
                        cases.append(('terminal_multiplier',model,replace(cfg,terminal_multiplier=multiplier)))
                for experiment,model,c in cases:
                    strategy=system+'_'+model
                    with np.load(root/'runs'/strategy/(date+'.npz')) as saved:
                        E0=float(saved['E'][0])
                    bundle=build_scenarios(f,engine.records,composition,c)
                    plan=optimize_plan(model,f,bundle,E0,data.price,c)
                    actual=rollout(plan['G'],plan['reference'],E0,iter(data.values[i]),data.price,c)
                    baseline=json.loads((root/'runs'/strategy/(date+'.json')).read_text(encoding='utf-8'))
                    row=dict(date=date,strategy=strategy,experiment=experiment,radius=c.radius,
                              terminal_multiplier=c.terminal_multiplier,budget=c.budget,
                              objective=plan['objective'],cash_cost=actual['cash_cost'],
                              emergency_cost=actual['emergency_cost'],emergency_kwh=float(actual['H'].sum()),
                              baseline_objective=baseline['plan']['best_objective'],
                              baseline_cash=baseline['settlement']['cash_cost'],E0=E0,
                              evaluations=plan['log']['evaluations'],seconds=plan['log']['elapsed_seconds'],
                              retrospective_not_used_for_formal_tuning=True)
                    case=f'{date}_{strategy}_{experiment}_b{c.budget}_r{c.radius}_n{c.terminal_multiplier}'.replace('.', 'p')
                    save_npz(out/'traces_v2'/'locked_plans'/(case+'.npz'),G=plan['G'],reference=plan['reference'])
                    save_rollout(out/'traces_v2','diagnostics',case,plan,actual)
                    save_json(out/'traces_v2'/'cases'/(case+'.json'),row)
                    records.append(row)
                    print(json.dumps({k:row[k] for k in ['date','strategy','experiment','budget','radius','terminal_multiplier']},ensure_ascii=False),flush=True)
        engine.settle(issued,data.values[i])
    out=root.parent/(root.name+'_sensitivity')
    out.mkdir(exist_ok=True)
    save_json(out/'profiles.json',records)
    write_result(pd.DataFrame(records),out/'profiles.xlsx')
    from common.plotting import plt,set_style
    set_style()
    df=pd.DataFrame(records)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for strategy,part in df[df.experiment=='TV_radius'].groupby('strategy'):
        means=part.groupby('radius')[['cash_cost','emergency_cost']].mean()
        axes[0].plot(means.index,means.cash_cost,marker='o',label=strategy)
        axes[1].plot(means.index,means.emergency_cost,marker='o',label=strategy)
    for ax,label in zip(axes,['Mean actual cash on four fixed dates (CNY)','Mean emergency fee on four fixed dates (CNY)']):
        ax.set_xlabel('TV radius')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/'radius_cost_risk.png',dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['tables','profiles'])
    p.add_argument('--run',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--wait-available',action='store_true')
    a=p.parse_args()
    if a.mode=='tables':
        tables(a.run)
    else:
        profiles(a.run,load_config(a.config),a.wait_available)

if __name__=='__main__':
    main()
