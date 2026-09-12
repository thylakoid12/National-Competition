"""对真实实验独立重放；仅审计已完成的模型，不修改计划与冻结选型。"""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from microgrid.config import STATISTICAL_RESULT
from microgrid.data import load_data
from microgrid.statistical_experiment import load_run, build_bank, scenes_at
from microgrid.storage import read_arrays, read_json, write_json, file_hash
from microgrid.models import get_model
from microgrid.optimizer import optimize_risk
from microgrid.seed import seed_plan
from tests.oracle import scalar_replay


def audit(source, reoptimize=True):
    source=Path(source)
    cfg, protocol=load_run(source)
    selection=read_json(source/'selection_frozen.json')
    for field, name in (('protocol_sha256','protocol_frozen.json'),
                        ('january_bank_sha256','january_bank.npz'),
                        ('initial_states_sha256','initial_states.json')):
        assert selection[field] == file_hash(source/name), name
    bank=read_arrays(source/'evaluation_bank.npz')
    data=load_data(protocol['attachments'])
    np.testing.assert_array_equal(data.values,bank['actual'])
    np.testing.assert_array_equal(data.price,bank['price'])
    rebuilt, _=build_bank(data,cfg,365)
    for key in ('f0','contexts','actual','price'):
        np.testing.assert_allclose(rebuilt[key],bank[key],rtol=0,atol=1e-9)
    # 每一个评价日均验证：将当前与未来实测毒化，不影响日初场景。
    for day in range(31,365):
        changed=dict(bank)
        changed['actual']=bank['actual'].copy()
        changed['actual'][day:]=np.nan
        for mode in ('point','uniform','conditional'):
            f,a=scenes_at(bank,day,mode,cfg)
            g,b=scenes_at(changed,day,mode,cfg)
            for key in ('path','context'):
                np.testing.assert_array_equal(f[key],g[key])
            for key in ('paths','weights'):
                np.testing.assert_array_equal(a[key],b[key])
            assert all(d < f['meta']['date'] for d in b['dates'])
    states=read_json(source/'initial_states.json')
    phases={}
    for phase,count,initial in (('warmup',31,cfg.initial_energy),
                                ('validation',17,states['validation_initial']),
                                ('evaluation',334,states['evaluation_initial'])):
        directories=[source/'warmup'] if phase=='warmup' else sorted((source/phase).iterdir())
        phases[phase]={}
        for directory in directories:
            if not directory.is_dir():
                continue
            if phase!='warmup' and (not (directory/'complete.json').exists()
                    or not read_json(directory/'complete.json')['complete']):
                phases[phase][directory.name]={'complete':False,'audited':False}
                continue
            previous=initial
            cash=H=0.0
            max_delta=0.0
            paths=sorted(directory.glob('2025-??-??.json'))
            assert len(paths)==count, directory
            for path in paths:
                row=read_json(path)
                date=row['date']
                assert path.stem==date
                day=list(bank['dates']).index(date)
                arrays=read_arrays(path.with_suffix('.npz'))
                locked_path=directory/'locked_plans'/(date+'.npz')
                locked=read_arrays(locked_path)
                assert row['array_sha256']==file_hash(path.with_suffix('.npz'))
                assert row['locked_plan_sha256']==file_hash(locked_path)
                assert abs(row['E0']-previous)<1e-7
                np.testing.assert_array_equal(arrays['G'],locked['G'])
                np.testing.assert_array_equal(arrays['reference'],locked['reference'])
                np.testing.assert_array_equal(arrays['load'],bank['actual'][day,:,0])
                np.testing.assert_array_equal(arrays['pv'],bank['actual'][day,:,1])
                replay=scalar_replay(locked['G'],locked['reference'],previous,
                                     bank['actual'][day],bank['price'],cfg)
                for key in ('E','charge','discharge','H','U','W'):
                    delta=float(np.max(np.abs(replay[key]-arrays[key])))
                    assert delta < 1e-7, (directory,date,key,delta)
                    max_delta=max(max_delta,delta)
                assert abs(replay['cash_cost']-row['cash_cost'])<1e-7
                assert abs(replay['E'][-1]-row['Eend'])<1e-7
                if phase!='warmup':
                    assert row['forecast_train_end'] < date
                    assert row['scene_end'] is None or row['scene_end'] < date
                    assert row['optimizer']['evaluations']<=cfg.search_evaluations
                    lock_meta=read_json(locked_path.with_suffix('.json'))
                    assert lock_meta['G_sha256']==file_hash(locked_path)
                    assert lock_meta['protocol_sha256']==file_hash(source/'protocol_frozen.json')
                    assert row['plan_feasible']==lock_meta['feasible']
                    if row['plan_feasible']:
                        for value,limit in (('risk_cvar','risk_budget'),('design_stress_cost','stress_budget')):
                            if row[limit] is not None:
                                assert row[value]<=row[limit]+1e-6+1e-7
                cash+=replay['cash_cost']; H+=float(replay['H'].sum())
                previous=row['Eend']
            if phase!='warmup':
                summary=read_json(directory/'summary.json')
                assert abs(summary['cash_cost']-cash)<1e-6
                assert abs(summary['H']-H)<1e-6
                assert summary['days']==count and summary['complete']
                manifest=read_json(directory/'run_manifest.json')
                assert manifest['bank_sha256']==file_hash(source/('january_bank.npz' if phase=='validation' else 'evaluation_bank.npz'))
            phases[phase][directory.name]=dict(complete=True,audited=True,days=count,
                independent_cash_cost=cash,independent_emergency_kwh=H,max_array_delta=max_delta,
                initial_energy=initial,end_energy=previous)
    optimized=[]
    selected=selection['selected']
    if reoptimize and phases['evaluation'].get(selected,{}).get('audited'):
        for date in ('2025-02-01','2025-06-21','2025-12-31'):
            day=list(bank['dates']).index(date)
            row=read_json(source/'evaluation'/selected/(date+'.json'))
            altered=dict(bank)
            altered['actual']=bank['actual'].copy()
            altered['actual'][day:]=np.nan
            spec=get_model(selected)
            f,scenes=scenes_at(altered,day,spec.scenario_mode,cfg)
            seed,_=seed_plan(f,row['E0'],bank['price'],cfg)
            plan=optimize_risk(f,scenes,row['E0'],bank['price'],seed,cfg,spec)
            locked=read_arrays(source/'evaluation'/selected/'locked_plans'/(date+'.npz'))
            for key in ('G','reference','alpha'):
                np.testing.assert_allclose(plan[key],locked[key],atol=1e-7,rtol=0)
            optimized.append(dict(date=date,current_and_future_actual='NaN',saved_plan_matches=True))
    complete=all(v.get('audited',False) for v in phases['evaluation'].values())
    result=dict(created=datetime.now().isoformat(),input_files_match=True,
        all_365_day_forecasts_rebuilt_from_past=True,
        all_334_day_scenarios_ignore_current_and_future=True,phases=phases,
        selected_reoptimization=optimized,all_evaluation_models_audited=complete)
    write_json(source/'audit'/'independent_replay.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=STATISTICAL_RESULT)
    parser.add_argument('--skip-reoptimization',action='store_true')
    args=parser.parse_args()
    result=audit(args.source,not args.skip_reoptimization)
    print('独立重放核验通过；全部评价模型已核验：',result['all_evaluation_models_audited'])
