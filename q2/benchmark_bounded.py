"""January-only confirmation of the fixed v2 search and auxiliary budgets."""
from dataclasses import replace
import json
from q2_config import Config,HERE
from q2_data import load_data
from q2_forecast import Forecaster,combine
from q2_scenarios import build_scenarios
from q2_optimizer import optimize_plan
from q2_archive import save_json
from prepare_fast_run import peak_memory

def main():
    cfg=replace(Config(),reference_backend='highspy',accelerated_paths=True,bounded_seed=True,
                 seed_time_limit_seconds=5.,seed_mip_gap=1e-4,
                 optimizer_version='coordinate-best-improvement-bounded-seed-v2.0')
    engine=Forecaster(replace(cfg,ml_min_days=10000))
    data=load_data()
    logs=[]
    for i in range(1,22):
        issued=engine.issue(data.history(i),data.dates[i])
        if i in [7,21]:
            f=combine(issued,[0,0],'F0_benchmark')
            s=build_scenarios(f,engine.records,[0,0],cfg)
            for model in ['M0','M1','M2','M3']:
                p=optimize_plan(model,f,s,6000,data.price,cfg)
                row=dict(date=data.dates[i],model=model,seconds=p['log']['elapsed_seconds'],
                          evaluations=p['log']['evaluations'],seed=p['log']['seed_solver'],
                          peak_rss_bytes=peak_memory(),scenario_count=len(s['paths']))
                logs.append(row)
                print(json.dumps(row,ensure_ascii=False),flush=True)
        engine.settle(issued,data.values[i])
    save_json(HERE/'result/verification/benchmark_bounded_final.json',logs)

if __name__=='__main__':
    main()
