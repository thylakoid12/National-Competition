"""独立留出压力验证：先锁计划再揭示冲击，多日只使用已结算压力历史。"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
from .controller import rollout
from .forecast import StatisticalForecaster
from .metrics import summarize_day
from .models import get_model
from .optimizer import optimize_risk
from .scenarios import build_scenarios
from .seed import seed_plan
from .statistical_experiment import load_run
from .storage import read_arrays, read_json, write_arrays, write_json, file_hash
from .stress import Shock, HELDOUT, apply_shock

DEFAULT_ANCHORS = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


def simulate_episode(bank, start, E0, shock, cfg, model, directory, first_plan_cache=None,
                     recovery_days=0):
    """冲击信息只交给实测揭示器；预测器和优化器没有 shock 参数。"""
    directory = Path(directory)
    model = get_model(model)
    engine = StatisticalForecaster(cfg)
    history = [p.copy() for p in bank["actual"][:start]]
    residuals, contexts, dates = [], [], []
    for i in range(1, start):
        f = engine.issue(bank["actual"][:i], str(bank["dates"][i]))
        residuals.append(bank["actual"][i]-f["path"])
        contexts.append(f["context"]); dates.append(str(bank["dates"][i]))
        engine.settle(bank["actual"][i], dates[-1])
    rows = []
    for offset in range(shock.duration+recovery_days):
        i = start+offset
        if i >= len(bank["dates"]):
            raise ValueError("Stress episode extends beyond available dates")
        day = str(bank["dates"][i])
        f = engine.issue(np.array(history), day)
        s = build_scenarios(f, residuals, contexts, dates, model.scenario_mode, cfg)
        cache_key = (model.name, day, float(E0)) if offset == 0 else None
        if first_plan_cache is not None and cache_key in first_plan_cache:
            plan = first_plan_cache[cache_key]
        else:
            G0, _ = seed_plan(f, E0, bank["price"], cfg)
            plan = optimize_risk(f, s, E0, bank["price"], G0, cfg, model)
            if first_plan_cache is not None and cache_key is not None:
                first_plan_cache[cache_key] = plan
        lock = directory/"locked_plans"/(day+".npz")
        write_arrays(lock, G=plan["G"], reference=plan["reference"], alpha=np.array(plan["alpha"]))
        write_json(lock.with_suffix(".json"), dict(date=day, E0=E0, forecast=f["meta"],
                   model=model.to_dict(), plan_feasible=plan["feasible"],
                   G_sha256=file_hash(lock), shock_parameters_passed_to_planner=False))
        # 当前实测在计划落盘后揭示；恢复期衔接受压后的历史与库存。
        raw = bank["actual"][i]
        realized = apply_shock(raw, shock) if offset < shock.duration else raw.copy()
        o = rollout(plan["G"], plan["reference"], E0, iter(realized), bank["price"], cfg)
        row = summarize_day(o,day,cfg)
        row.update(alpha=plan["alpha"], plan_feasible=plan["feasible"],
                   phase="shock" if offset < shock.duration else "recovery",
                   forecast_train_end=f["meta"]["train_end"], locked_plan_sha256=file_hash(lock))
        write_arrays(directory/(day+".npz"), **{k:v for k,v in o.items() if isinstance(v,np.ndarray)},
                     reference=plan["reference"])
        write_json(directory/(day+".json"),row)
        rows.append(row)
        history.append(realized); residuals.append(realized-f["path"])
        contexts.append(f["context"]); dates.append(day)
        engine.settle(realized,day)
        E0=row["Eend"]
    return rows


def stress_model(out, model_name, anchors=DEFAULT_ANCHORS):
    out = Path(out)
    cfg, protocol = load_run(out)
    if not (out/"selection_frozen.json").exists():
        raise ValueError("Held-out stress may only run after freezing selection")
    model = get_model(model_name)
    bank = read_arrays(out/"evaluation_bank.npz")
    directory=out/"stress"/model.name
    manifest=dict(protocol_sha256=file_hash(out/"protocol_frozen.json"),model=model.to_dict(),
        anchors=list(anchors), shocks=[s.to_dict() for s in HELDOUT],
        evaluator_sha256=file_hash(Path(__file__)), bank_sha256=file_hash(out/"evaluation_bank.npz"),
        initial_energy=cfg.initial_energy, recovery_days_for_multiday_shocks=2,
        used_for_selection=False, same_initial_state_across_models=True)
    old=directory/"manifest.json"
    if old.exists() and read_json(old)!=manifest:
        raise ValueError("Stress settings changed; use a separate run directory")
    write_json(old,manifest)
    rows=[]
    first_cache={}
    for anchor in anchors:
        start=list(bank["dates"]).index(anchor)
        for shock in HELDOUT:
            case=directory/anchor/shock.name
            summary_path=case/"summary.json"
            if summary_path.exists():
                rows.append(read_json(summary_path)); continue
            initial=(cfg.initial_energy if shock.initial_fraction is None else
                     cfg.e_min+shock.initial_fraction*(cfg.e_max-cfg.e_min))
            recovery=2 if shock.duration>1 else 0
            horizon=shock.duration+recovery
            baseline_dir=directory/anchor/f"nominal_E{initial:.6f}_days{horizon}"
            baseline_file=baseline_dir/"summary.json"
            if baseline_file.exists():
                baseline=read_json(baseline_file)["days"]
            else:
                baseline=simulate_episode(bank,start,initial,Shock("nominal",duration=horizon),
                    cfg,model,baseline_dir,first_cache)
                write_json(baseline_file,dict(days=baseline))
            stressed=simulate_episode(bank,start,initial,shock,cfg,model,case,first_cache,recovery)
            # 同初值、同历史的第一天必须发布同一份计划。
            a=read_arrays(baseline_dir/"locked_plans"/(anchor+".npz"))
            b=read_arrays(case/"locked_plans"/(anchor+".npz"))
            for key in ("G","reference","alpha"):
                np.testing.assert_array_equal(a[key],b[key])
            result=dict(model=model.name,anchor=anchor,shock=shock.to_dict(),initial_energy=initial,
                days=horizon,shock_days=shock.duration,recovery_days=recovery,
                cash_cost=sum(r["cash_cost"] for r in stressed),
                emergency_cost=sum(r["emergency_cost"] for r in stressed),
                emergency_kwh=sum(r["H"] for r in stressed),
                baseline_cash_cost=sum(r["cash_cost"] for r in baseline),
                baseline_emergency_cost=sum(r["emergency_cost"] for r in baseline),
                additional_cash_cost=sum(a["cash_cost"]-b["cash_cost"] for a,b in zip(stressed,baseline)),
                peak_emergency_kw=max(r["peak_emergency_kw"] for r in stressed),
                min_energy=min(r["min_energy"] for r in stressed),
                end_energy=stressed[-1]["Eend"],
                end_inventory_gap_vs_nominal=stressed[-1]["Eend"]-baseline[-1]["Eend"],
                recovery_emergency_cost=sum(r["emergency_cost"] for r in stressed[shock.duration:]),
                risk_budget_unmet_days=sum(not r["plan_feasible"] for r in stressed),
                max_physical_residual=max(r["max_residual"] for r in stressed),
                first_day_plan_identical=True, used_for_selection=False)
            write_json(summary_path,result)
            rows.append(result)
            print(json.dumps(dict(phase="heldout_stress",model=model.name,anchor=anchor,
                   shock=shock.name,emergency_kwh=result["emergency_kwh"]),ensure_ascii=False),flush=True)
    summary=dict(model=model.name,cases=len(rows),complete=True,results=rows,
                 used_for_selection=False,unlimited_emergency_supply=True)
    write_json(directory/"summary.json",summary)
    return summary


def run_stress(out, models=None, anchors=DEFAULT_ANCHORS, workers=1):
    out=Path(out)
    selection=read_json(out/"selection_frozen.json")
    models=list(dict.fromkeys(models or [selection["selected"],"M2"]))
    if workers==1:
        results={m:stress_model(out,m,anchors) for m in models}
    else:
        results={}
        with ProcessPoolExecutor(max_workers=min(workers,len(models))) as pool:
            jobs={pool.submit(stress_model,str(out),m,anchors):m for m in models}
            for future in as_completed(jobs):
                results[jobs[future]]=future.result()
    summary=dict(models=results,used_for_selection=False,anchors=list(anchors),
                 interpretation="人工指定冲击下的费用和应急依赖；未假定发生概率，不是停电可靠性评估")
    write_json(out/"stress_summary.json",summary)
    return summary
