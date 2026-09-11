"""Independent re-read checks across actual source data, forecasts, scenarios and cash."""
from pathlib import Path
import argparse,json
import numpy as np
from q2_config import load_config
from q2_data import load_data
from q2_forecast import Forecaster,combine
from q2_scenarios import build_scenarios
from q2_controller import rollout
from q2_stress import read_issue
from q2_export import read_records
from q2_archive import save_json
from q2_backtest import STRATEGIES

def audit(root,cfg):
    root=Path(root)
    data=load_data()
    engine=Forecaster(cfg)
    completed=sorted((root/'residuals').glob('*.npz'))
    if len(completed)!=364:
        raise ValueError('Annual audit requires 364 issued forecast/residual days after the cold start')
    max_residual_error=0.
    for i in range(1,365):
        date=data.dates[i]
        issue=read_issue(root,date)
        for f in issue['issued']:
            assert f['meta']['train_end']==data.dates[i-1]
            assert f['meta']['issued_at']==date+'T00:00:00'
            assert f['meta']['historical_days']==i
            assert f['meta']['config_version']==cfg.version
        if i==31:
            engine.freeze()
        if i>=31:
            for system,composition in [('star',engine.composition),('dagger',[1-x for x in engine.composition])]:
                forecast=combine(issue,composition,system)
                expected=build_scenarios(forecast,engine.records,composition,cfg)
                with np.load(root/'scenarios'/system/(date+'.npz')) as saved:
                    for key in ['paths','weights','clipping']:
                        np.testing.assert_allclose(saved[key],expected[key],rtol=0,atol=1e-10)
                meta=json.loads((root/'scenarios'/system/(date+'.json')).read_text(encoding='utf-8'))
                assert meta['dates']==expected['dates']
                assert meta['sources']==expected['sources']
                assert meta['version']==expected['version']
        record,_=engine.settle(issue,data.values[i])
        with np.load(root/'residuals'/(date+'.npz')) as saved:
            error=np.abs(saved['residual']-record['residual']).max()
            assert error<1e-10
            max_residual_error=max(max_residual_error,float(error))
    freeze=json.loads((root/'freeze.json').read_text(encoding='utf-8'))
    assert freeze['star']==engine.composition
    totals={}
    for strategy in STRATEGIES:
        rows=read_records(root,strategy,data.price,cfg)
        assert [date for date,_ in rows]==list(data.dates[31:])
        assert abs(rows[0][1]['E'][0]-freeze['shared_E0'])<1e-6
        cash=0.
        maximum=0.
        for date,o in rows:
            index=data.dates.index(date)
            np.testing.assert_array_equal(o['load'],data.values[index,:,0])
            np.testing.assert_array_equal(o['pv'],data.values[index,:,1])
            replay=rollout(o['G'],o['reference'],o['E'][0],iter(data.values[index]),data.price,cfg)
            for key in ['G','E','H','charge','discharge','U','W']:
                np.testing.assert_allclose(o[key],replay[key],rtol=0,atol=1e-8)
            maximum=max(maximum,max(replay['residuals'].values()))
            cash+=float(data.price@o['G']+5*data.price@o['H'])
        totals[strategy]=dict(days=len(rows),slots=len(rows)*144,cash_recomputed=cash,max_constraint_residual=maximum)
    failures=[]
    for i in range(14,365):
        meta=json.loads((root/'forecasts/F1'/(data.dates[i]+'.json')).read_text(encoding='utf-8'))
        if any(meta['fallback']):
            failures.append(dict(date=data.dates[i],reason=meta['fallback_reason']))
        else:
            assert meta['training_rows']==min(i,cfg.ml_window)*144
    report=dict(annual_complete=True,raw_days=365,formal_days=334,slots_per_day=144,
                joint_residual_pairing_verified=True,scenario_versions_and_order_verified=True,
                residual_max_error=max_residual_error,training_failure_dates=failures,
                common_initial_energy=freeze['shared_E0'],strategies=totals)
    save_json(root/'audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--config',required=True)
    a=p.parse_args()
    audit(a.run,load_config(a.config))

if __name__=='__main__':
    main()
