"""Independent Q-vector ordering and whole-path TV checks on final stored plans."""
from pathlib import Path
import json
import numpy as np
from q2_config import load_config
from q2_data import load_data
from q2_forecast import combine
from q2_stress import read_issue
from q2_controller import batch_costs
from q2_risk import worst_case_tv
from q2_archive import save_json
from q2_backtest import STRATEGIES

def main(root=Path('result/formal_v2'),config='config_formal_v2.json'):
    cfg=load_config(config)
    data=load_data()
    freeze=json.loads((root/'freeze.json').read_text(encoding='utf-8'))
    max_error=max_obj_error=0.
    count=0
    for date in data.dates[31:]:
        issued=read_issue(root,date)
        for strategy in STRATEGIES:
            system,model=strategy.split('_')
            f=combine(issued,freeze[system],system)
            with np.load(root/'scenarios'/system/(date+'.npz')) as s:
                paths=s['paths'].copy()
                weights=s['weights'].copy()
            with np.load(root/'runs'/strategy/(date+'.npz')) as r:
                G,reference,E0=r['G'],r['reference'],float(r['E'][0])
                Q=batch_costs(G,reference,E0,f['path'][None] if model=='M0' else paths,
                              data.price,cfg.terminal(date,data.price),cfg)
                err=float(np.abs(Q-r['scenario_costs']).max())
                assert err<1e-6
                max_error=max(max_error,err)
                if model=='M0':
                    risk=Q[0]
                    expected_q=np.ones(1)
                elif model=='M1':
                    expected_q=np.ones(len(Q))/len(Q)
                    risk=expected_q@Q
                elif model=='M2':
                    expected_q=weights
                    risk=weights@Q
                else:
                    risk,expected_q=worst_case_tv(Q,weights,cfg.radius)
                    q=r['worst_q']
                    assert q.min()>=-1e-12 and abs(q.sum()-1)<1e-10
                    assert .5*np.abs(q-weights).sum()<=cfg.radius+1e-10
                    assert q@Q>=weights@Q-1e-7
                if model!='M3':
                    np.testing.assert_allclose(r['worst_q'],expected_q,rtol=0,atol=1e-9)
                else:
                    assert abs(r['worst_q']@Q-risk)<1e-6
                meta=json.loads((root/'runs'/strategy/(date+'.json')).read_text(encoding='utf-8'))
                objective=float(data.price@G+risk)
                error=abs(objective-meta['plan']['best_objective'])
                assert error<1e-6
                max_obj_error=max(max_obj_error,error)
                count+=1
    save_json(root/'risk_audit.json',dict(plans_checked=count,scenario_cost_order_verified=True,
              whole_day_tv_verified=True,max_Q_error=max_error,max_objective_error=max_obj_error))
    print('Risk audit passed:',count,flush=True)

if __name__=='__main__':
    main()
