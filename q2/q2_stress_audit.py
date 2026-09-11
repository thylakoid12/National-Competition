"""Replay and check all stress branches without training on unrevealed shocks."""
from pathlib import Path
import json
import numpy as np
from q2_config import load_config
from q2_data import load_data
from q2_forecast import Forecaster,combine,features
from q2_scenarios import build_scenarios
from q2_stress import read_issue
from q2_export import read_records
from q2_backtest import STRATEGIES
from q2_controller import rollout
from q2_archive import save_json

def main(root=Path('result/formal_v2'),cfgfile='config_formal_v2.json'):
    data=load_data()
    cfg=load_config(cfgfile)
    first=data.dates.index(cfg.stress_start)
    output=root.parent/(root.name+'_stress')
    counts={}
    for shock in ['load_up','pv_down','joint_multiday']:
        branch=output/shock
        engine=Forecaster(cfg)
        for i in range(1,first):
            engine.settle(read_issue(root,data.dates[i]),data.values[i])
        engine.freeze()
        history=data.history(first).copy()
        days=cfg.stress_days if shock=='joint_multiday' else 1
        with np.load(branch/'magnitude.npz') as s:
            sigma=s['sigma']
            np.testing.assert_array_equal(sigma,data.history(first)[-60:].std(axis=0))
        records={strategy:dict(read_records(branch,strategy,data.price,cfg)) for strategy in STRATEGIES}
        for offset in range(days):
            date=data.dates[first+offset]
            issued=read_issue(branch,date)
            # Recompute the entire day-ahead forecast from this branch's past only.
            reconstructed=engine.issue(history,date)
            for original,new in zip(issued['issued'],reconstructed['issued']):
                np.testing.assert_allclose(original['path'],new['path'],rtol=0,atol=1e-9)
                np.testing.assert_array_equal(original['context'],new['context'])
            for system,composition in [('star',engine.composition),('dagger',[1-x for x in engine.composition])]:
                expected=build_scenarios(combine(issued,composition,system),engine.records,composition,cfg)
                with np.load(branch/'scenarios'/system/(date+'.npz')) as s:
                    for key in ['paths','weights','clipping']:
                        np.testing.assert_allclose(s[key],expected[key],rtol=0,atol=1e-9)
            actual=data.values[first+offset].copy()
            if shock in ['load_up','joint_multiday']:
                a,b=cfg.stress_load_slots
                actual[a:b,0]+=cfg.stress_gamma*sigma[a:b,0]
            if shock in ['pv_down','joint_multiday']:
                a,b=cfg.stress_pv_slots
                actual[a:b,1]=np.maximum(0,actual[a:b,1]-cfg.stress_gamma*sigma[a:b,1])
            for strategy in STRATEGIES:
                saved=records[strategy][date]
                np.testing.assert_array_equal(saved['load'],actual[:,0])
                np.testing.assert_array_equal(saved['pv'],actual[:,1])
                if offset==0:
                    with np.load(root/'runs'/strategy/(date+'.npz')) as normal:
                        np.testing.assert_array_equal(saved['G'],normal['G'])
                        assert abs(saved['E'][0]-normal['E'][0])<1e-9
                replay=rollout(saved['G'],saved['reference'],saved['E'][0],iter(actual),data.price,cfg)
                for key in ['E','H','charge','discharge','U','W']:
                    np.testing.assert_allclose(replay[key],saved[key],rtol=0,atol=1e-9)
            record,_=engine.settle(issued,actual)
            with np.load(branch/'residuals'/(date+'.npz')) as saved:
                np.testing.assert_allclose(saved['residual'],record['residual'],rtol=0,atol=1e-9)
            history=np.concatenate([history,actual[None]])
        counts[shock]=dict(days=days,strategies=6,forecast_and_residual_replay=True,
                          locked_before_shock=True,state_continuity=True)
    save_json(output/'audit.json',counts)
    print('Stress independent audit passed',flush=True)

if __name__=='__main__':
    main()
