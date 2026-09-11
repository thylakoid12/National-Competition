"""Strict day-by-day replay; forecasts precede plans, observations and residuals."""
from dataclasses import asdict
from pathlib import Path
from datetime import date as Date,timedelta
import argparse,copy,json,pickle,sys
import numpy as np
from q2_config import Config,HERE,load_config
from q2_data import load_data
from q2_forecast import Forecaster,combine
from q2_scenarios import build_scenarios
from q2_optimizer import optimize_plan
from q2_controller import rollout,validate
from q2_archive import save_json,save_npz,save_issue,save_scenarios,save_rollout
from q2_metrics import summarize,probability_scores,point_scores

STRATEGIES=('star_M0','star_M1','star_M2','star_M3','dagger_M0','dagger_M3')

def cold_start(actual,price,cfg):
    net=actual[:,0]-actual[:,1]
    z=np.zeros(144)
    o=dict(G=z.copy(),load=actual[:,0].copy(),pv=actual[:,1].copy(),charge=z.copy(),discharge=z.copy(),
           H=np.maximum(net,0),U=z.copy(),W=np.maximum(-net,0),E=np.full(145,cfg.initial_energy),plan_cost=0.)
    o['emergency_cost']=float(cfg.emergency_multiplier*price@o['H'])
    o['cash_cost']=o['emergency_cost']
    o['residuals']=validate(o,price,cfg)
    return o

def daily_row(date,out,plan):
    return dict(date=date,plan_cost=out['plan_cost'],emergency_cost=out['emergency_cost'],cash_cost=out['cash_cost'],
                emergency_kwh=float(out['H'].sum()),curtailment_kwh=float(out['W'].sum()),
                unused_kwh=float(out['U'].sum()),emergency_slots=int((out['H']>1e-6).sum()),
                E0=float(out['E'][0]),Eend=float(out['E'][-1]),objective=plan['objective'],
                evaluations=plan['log']['evaluations'],seconds=plan['log']['elapsed_seconds'],
                max_residual=max(out['residuals'].values()))

def write_checkpoint(root,state):
    target=Path(root)/'checkpoint.pkl'
    temporary=target.with_suffix('.tmp')
    with temporary.open('wb') as f:
        pickle.dump(state,f,protocol=5)
    temporary.replace(target)

def execute(cfg,output,end='2025-12-31',resume=False):
    root=Path(output)
    root.mkdir(parents=True,exist_ok=True)
    data=load_data()
    save_json(root/'config.json',asdict(cfg))
    save_json(root/'data_validation.json',dict(raw_days=365,slots=144,formal_days=334,
              units='kWh, converted from source kW exactly once',first=data.interval_start[0],last=data.interval_end[-1],
              price_source='附件1.xlsx',actual_source='附件2.xlsx',source_shape=list(data.values.shape)))
    if resume:
        # Only load checkpoints produced locally by this program in the selected run directory.
        with (root/'checkpoint.pkl').open('rb') as f:
            state=pickle.load(f)
        if state['config_version']!=cfg.version:
            raise ValueError('Configuration changed; use a new experiment directory')
    else:
        if (root/'checkpoint.pkl').exists():
            raise FileExistsError('Use --resume or a new output directory')
        engine=Forecaster(cfg)
        cold=cold_start(data.values[0],data.price,cfg)
        save_npz(root/'warmup/2025-01-01.npz',**{k:v for k,v in cold.items() if isinstance(v,np.ndarray)})
        save_json(root/'warmup/2025-01-01.json',dict(policy='zero_plan_battery_idle',cash_cost=cold['cash_cost'],
                    emergency_cost=cold['emergency_cost'],residuals=cold['residuals'],excluded_from_formal=True))
        state=dict(config_version=cfg.version,next_day=1,engine=engine,warm_energy=cfg.initial_energy,
                   states={},rows={s:[] for s in STRATEGIES},points=[],probability=[],warm_rows=[])
        write_checkpoint(root,state)
    from concurrent.futures import ProcessPoolExecutor
    pool=ProcessPoolExecutor(max_workers=cfg.parallel_workers) if cfg.parallel_workers>1 else None
    for i in range(state['next_day'],len(data.dates)):
        date=data.dates[i]
        if date>end:
            break
        engine=state['engine']
        history=data.history(i)
        issued=engine.issue(history,date)
        save_issue(root,issued)  # Persist BEFORE seeing the actual day or choosing any G.
        if date<'2025-02-01':
            f=combine(issued,[0,0],'warmup_F0')
            s=build_scenarios(f,engine.records,[0,0],cfg)
            plan=optimize_plan('M0',f,s,state['warm_energy'],data.price,cfg)
            save_npz(root/'locked_plans/warmup'/(date+'.npz'),G=plan['G'],reference=plan['reference'])
            out=rollout(plan['G'],plan['reference'],state['warm_energy'],iter(data.values[i]),data.price,cfg)
            save_rollout(root,'warmup',date,plan,out)
            state['warm_rows'].append(daily_row(date,out,plan))
            state['warm_energy']=float(out['E'][-1])
        else:
            if not state['states']:
                state['states']={strategy:state['warm_energy'] for strategy in STRATEGIES}
                save_json(root/'freeze.json',dict(engine.freeze_report,shared_E0=state['warm_energy']))
            all_plans={}
            bundles={}
            # All forecasts and plans are locked before any actual observation is revealed.
            for system,composition in [('star',engine.composition),('dagger',[1-x for x in engine.composition])]:
                f=combine(issued,composition,system)
                s=build_scenarios(f,engine.records,composition,cfg)
                bundles[system]=s
                save_scenarios(root,system,date,s)
                save_json(root/'systems'/system/(date+'.json'),f['meta'])
                for strategy in [x for x in STRATEGIES if x.startswith(system+'_')]:
                    arguments=(strategy.split('_')[1],f,s,state['states'][strategy],data.price,cfg)
                    all_plans[strategy]=(pool.submit(optimize_plan,*arguments) if pool else optimize_plan(*arguments))
            for strategy,pending in list(all_plans.items()):
                plan=pending.result() if pool else pending
                all_plans[strategy]=plan
                save_npz(root/'locked_plans'/strategy/(date+'.npz'),G=plan['G'],reference=plan['reference'])
            for strategy,plan in all_plans.items():
                locked=plan['G'].copy()
                E0=state['states'][strategy]
                out=rollout(plan['G'],plan['reference'],E0,iter(data.values[i]),data.price,cfg)
                np.testing.assert_array_equal(plan['G'],locked)
                if state['rows'][strategy]:
                    assert abs(E0-state['rows'][strategy][-1]['Eend'])<cfg.physical_tolerance
                save_rollout(root,strategy,date,plan,out)
                state['states'][strategy]=float(out['E'][-1])
                state['rows'][strategy].append(daily_row(date,out,plan))
            state['probability'].append(dict(date=date,scores={system:probability_scores(s,data.values[i],data.price)
                                                               for system,s in bundles.items()}))
        # Actual-minus-issued residuals are created only after settlement.
        record,comparison=engine.settle(issued,data.values[i])
        save_npz(root/'residuals'/(date+'.npz'),residual=record['residual'],context=record['context'])
        save_json(root/'residuals'/(date+'.json'),dict(date=date,created_at=date+'T24:00:00',
                    forecast_versions=[f['meta']['version'] for f in issued['issued']],
                    fallback=[f['meta']['fallback'] for f in issued['issued']]))
        error=np.stack([data.values[i]-f['path'] for f in issued['issued']])
        comparison.update(pv_daylight_mae=np.abs(error[:,36:108,1]).mean(axis=1).tolist(),
                          pv_daylight_mse=(error[:,36:108,1]**2).mean(axis=1).tolist())
        if date>='2025-02-01':
            state['points'].append(comparison)
        state['next_day']=i+1
        write_checkpoint(root,state)
        print(json.dumps(dict(date=date,completed=True,ml_fallback=issued['issued'][1]['meta']['fallback_reason'],
                              formal_days=len(state['rows']['star_M0'])),ensure_ascii=False),flush=True)
    if pool:
        pool.shutdown()
    # Summary filename is tied to completed date so resumed runs do not rewrite old summaries.
    completed=data.dates[state['next_day']-1]
    formal_days=len(state['rows']['star_M0'])
    summary=dict(completed_through=completed,formal_days=formal_days,complete=formal_days==334,
                  economic={s:summarize(rows) for s,rows in state['rows'].items() if rows},
                  point=point_scores(state['points']),probability=state['probability'],
                  warmup=state['warm_rows'])
    save_json(root/('summary_'+completed+'.json'),summary)
    return state

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output',required=True)
    parser.add_argument('--end',default='2025-12-31')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    execute(load_config(args.config),args.output,args.end,args.resume)

if __name__=='__main__':
    main()
