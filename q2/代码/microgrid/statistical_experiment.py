"""可独立复现的统计风险实验：一月预热/选型，二月至十二月连续评价。"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import numpy as np

from .config import Config, CODE_DIR, resolve_attachments
from .controller import rollout
from .data import load_data
from .forecast import StatisticalForecaster
from .metrics import summarize_day, risk_totals, pareto_frontier
from .models import get_model, DEFAULT_MODELS
from .optimizer import optimize_risk
from .risk import risk_summary
from .scenarios import build_scenarios
from .seed import seed_plan
from .solver import FastReference, batch_metrics
from .storage import read_json, write_json, read_arrays, write_arrays, file_hash
from .stress import DEVELOPMENT, HELDOUT, fixed_plan_stress

SCHEMA = "statistical-risk-v1"
VALIDATION = ("2025-01-15", "2025-01-31")
EVALUATION = ("2025-02-01", "2025-12-31")
COMPUTE_FILES = ("config.py", "models.py", "data.py", "forecast.py", "scenarios.py", "risk.py",
                 "solver.py", "controller.py", "optimizer.py", "seed.py", "seed_model.py",
                 "stress.py", "metrics.py", "statistical_experiment.py")


def setup(out, cfg, attachments=None):
    out = Path(out)
    attachments = resolve_attachments(attachments)
    protocol = dict(schema=SCHEMA, config=cfg.to_dict(), attachments=str(attachments),
        data_hashes={name: file_hash(attachments/name) for name in ("附件1.xlsx", "附件2.xlsx")},
        code_hashes={name: file_hash(CODE_DIR/"microgrid"/name) for name in COMPUTE_FILES},
        validation=list(VALIDATION), evaluation=list(EVALUATION),
        development_shocks=[s.to_dict() for s in DEVELOPMENT],
        heldout_shocks=[s.to_dict() for s in HELDOUT],
        selection_rule="Within M2 cash cost*(1+premium), minimize the worse normalized validation emergency CVaR and development-stress CVaR; cash breaks ties",
        bootstrap_rule="Jan 1: G=0, discharge only down to E_min, emergency supply; Jan 2–31: causal statistical forecast and deterministic seed with alpha=1",
        retrospective_redesign=True, future_guarantee=False,
        online_updates="Issued forecast residuals, bounded bias correction and past-only conditional weights; hyperparameters fixed after calibration")
    target = out/"protocol_frozen.json"
    if target.exists() and read_json(target) != protocol:
        raise ValueError("代码、附件或配置改变，不能复用旧断点；请指定新的 --output 目录。")
    write_json(target, protocol)
    return protocol


def load_run(out, verify=True):
    out = Path(out)
    protocol = read_json(out/"protocol_frozen.json")
    if protocol.get("schema") != SCHEMA:
        raise ValueError("This command requires a statistical-risk-v1 run")
    cfg = Config(**protocol["config"])
    if verify:
        for name, digest in protocol["code_hashes"].items():
            if file_hash(CODE_DIR/"microgrid"/name) != digest:
                raise ValueError(f"计算代码已改变，禁止混用断点：{name}")
        for name, digest in protocol["data_hashes"].items():
            if file_hash(Path(protocol["attachments"])/name) != digest:
                raise ValueError(f"附件已改变：{name}")
    return cfg, protocol


def build_bank(data, cfg, count):
    count = min(count, len(data.dates))
    actual = data.values[:count].copy()
    f0, contexts = np.zeros_like(actual), np.zeros((count, 12))
    metadata = [json.dumps(dict(date=data.dates[0], bootstrap=True))]
    engine = StatisticalForecaster(cfg)
    for i in range(1, count):
        forecast = engine.issue(actual[:i], data.dates[i])
        f0[i], contexts[i] = forecast["path"], forecast["context"]
        metadata.append(json.dumps(forecast["meta"], ensure_ascii=False))
        engine.settle(actual[i], data.dates[i])
    return dict(dates=np.array(data.dates[:count]), actual=actual, f0=f0, contexts=contexts,
                metadata=np.array(metadata), price=data.price.copy()), engine.report()


def forecast_at(bank, day):
    if day < 1:
        raise ValueError("Day zero is explicit bootstrap, not a fitted forecast")
    return dict(path=bank["f0"][day].copy(), context=bank["contexts"][day].copy(),
                meta=json.loads(str(bank["metadata"][day])))


def scenes_at(bank, day, mode, cfg):
    forecast = forecast_at(bank, day)
    start = max(1, day-cfg.residual_window)
    return forecast, build_scenarios(forecast, bank["actual"][start:day]-bank["f0"][start:day],
        bank["contexts"][start:day], bank["dates"][start:day].tolist(), mode, cfg)


def prepare(out, cfg, attachments=None):
    out = Path(out)
    protocol = setup(out, cfg, attachments)
    path = out/"january_bank.npz"
    if path.exists():
        bank = read_arrays(path)
    else:
        bank, report = build_bank(load_data(protocol["attachments"]), cfg, 31)
        write_arrays(path, **bank)
        write_json(out/"statistical_forecast_january.json", report)
    warmup(out, bank, cfg)
    return dict(schema=SCHEMA, january_days=len(bank["dates"]),
                initial_states=read_json(out/"initial_states.json"), output=str(out))


def _store_execution(directory, date, o, row, reference, extra=None):
    array_path = directory/(date+".npz")
    write_arrays(array_path, **{k: v for k, v in o.items() if isinstance(v, np.ndarray)},
                 reference=reference, **(extra or {}))
    row["array_sha256"] = file_hash(array_path)
    write_json(directory/(date+".json"), row)


def _read_checkpoint(directory, date, E0):
    path = directory/(date+".json")
    if not path.exists():
        return None
    row = read_json(path)
    arrays = directory/(date+".npz")
    locked = directory/"locked_plans"/(date+".npz")
    if (row["date"] != date or abs(row["E0"]-E0) > 1e-6 or not arrays.is_file()
            or file_hash(arrays) != row["array_sha256"] or not locked.is_file()
            or file_hash(locked) != row["locked_plan_sha256"]):
        raise ValueError(f"断点内容或库存连续性不匹配：{path}")
    return row


def warmup(out, bank, cfg):
    out = Path(out)
    directory = out/"warmup"
    E0, states = cfg.initial_energy, {}
    for i, date in enumerate(bank["dates"]):
        date = str(date)
        row = _read_checkpoint(directory, date, E0)
        if row is None:
            if i == 0:
                G, reference = np.zeros(144), np.full(145, cfg.e_min)
                reference[0] = E0
                seed_log = dict(status="explicit_no_history_bootstrap")
            else:
                forecast = forecast_at(bank, i)
                G, seed_log = seed_plan(forecast, E0, bank["price"], cfg)
                reference, _ = FastReference(forecast["path"], E0, bank["price"],
                                    cfg.terminal(date, bank["price"]), cfg).solve(G)
            lock = directory/"locked_plans"/(date+".npz")
            write_arrays(lock, G=G, reference=reference, alpha=np.array(1.0))
            o = rollout(G, reference, E0, iter(bank["actual"][i]), bank["price"], cfg)
            row = summarize_day(o, date, cfg)
            row.update(seed=seed_log, locked_plan_sha256=file_hash(lock), common_warmup=True)
            _store_execution(directory, date, o, row, reference)
        E0 = row["Eend"]
        if date == "2025-01-14":
            states["validation_initial"] = E0
        if date == "2025-01-31":
            states["evaluation_initial"] = E0
    states.update(initial_energy=cfg.initial_energy, source="reproducible common warmup/2025-01-01..31",
                  policy="Fixed causal warmup; each compared trajectory then carries its own realized inventory")
    write_json(out/"initial_states.json", states)


def prepare_evaluation(out):
    out = Path(out)
    cfg, protocol = load_run(out)
    selection = read_json(out/"selection_frozen.json")
    if (selection["protocol_sha256"] != file_hash(out/"protocol_frozen.json")
            or selection["january_bank_sha256"] != file_hash(out/"january_bank.npz")
            or selection["initial_states_sha256"] != file_hash(out/"initial_states.json")):
        raise ValueError("Frozen selection inputs changed")
    path = out/"evaluation_bank.npz"
    if not path.exists():
        bank, report = build_bank(load_data(protocol["attachments"]), cfg, 365)
        january = read_arrays(out/"january_bank.npz")
        for key in ("actual", "f0", "contexts"):
            np.testing.assert_allclose(bank[key][:31], january[key], atol=1e-9, rtol=0)
        write_arrays(path, **bank)
        write_json(out/"statistical_forecast_final.json", report)
    return path


def run_candidate(out, model_name, phase="validation", max_days=None):
    out = Path(out)
    cfg, protocol = load_run(out)
    model = get_model(model_name)
    if phase not in ("validation", "evaluation"):
        raise ValueError("Unknown phase")
    bank_path = out/("january_bank.npz" if phase == "validation" else "evaluation_bank.npz")
    bank = read_arrays(bank_path)
    if phase == "validation" and len(bank["dates"]) != 31:
        raise ValueError("Validation can only see a January bank")
    if phase == "evaluation" and not (out/"selection_frozen.json").exists():
        raise ValueError("Freeze January selection first")
    full_indices = list(range(14, 31) if phase == "validation" else range(31, 365))
    indices = full_indices[:max_days] if max_days else full_indices
    states = read_json(out/"initial_states.json")
    E0 = states["validation_initial" if phase == "validation" else "evaluation_initial"]
    directory = out/phase/model.name
    if max_days and len(list(directory.glob("2025-??-??.json"))) > max_days:
        raise ValueError("已有更多日期的结果；不能用较短冒烟范围覆盖完整性记录")
    manifest = dict(model=model.to_dict(), phase=phase, bank_sha256=file_hash(bank_path),
                    initial_states_sha256=file_hash(out/"initial_states.json"),
                    protocol_sha256=file_hash(out/"protocol_frozen.json"))
    manifest_path = directory/"run_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Candidate inputs changed; cannot resume")
    write_json(manifest_path, manifest)
    rows = []
    for i in indices:
        date = str(bank["dates"][i])
        row = _read_checkpoint(directory, date, E0)
        if row is None:
            forecast, scenes = scenes_at(bank, i, model.scenario_mode, cfg)
            G0, seed_log = seed_plan(forecast, E0, bank["price"], cfg)
            plan = optimize_risk(forecast, scenes, E0, bank["price"], G0, cfg, model)
            # 所有模型的事前风险另用共同的条件历史场景评估；这些数据不回流 M0 的搜索。
            _, audit_scenes = scenes_at(bank, i, "conditional", cfg)
            audit_metrics = batch_metrics(plan["G"], plan["reference"], E0,
                                           audit_scenes["paths"], bank["price"], cfg)
            forecast_risk = risk_summary(audit_metrics, audit_scenes["weights"], bank["price"], plan["G"], cfg)
            lock = directory/"locked_plans"/(date+".npz")
            write_arrays(lock, G=plan["G"], reference=plan["reference"], alpha=np.array(plan["alpha"]))
            lock_meta = dict(date=date, model=model.to_dict(), forecast=forecast["meta"],
                scenarios=scenes["diagnostics"], forecast_risk=forecast_risk,
                risk_budget=plan["risk_budget"], risk_cvar=plan["risk_cvar"],
                stress_budget=plan["stress_budget"], design_stress_cost=plan["stress_cost"],
                feasible=plan["feasible"], budget_scale=plan["budget_scale"],
                protocol_sha256=manifest["protocol_sha256"], G_sha256=file_hash(lock))
            write_json(lock.with_suffix(".json"), lock_meta)
            # 当日实测仅在计划落盘之后揭示。
            actual = bank["actual"][i]
            o = rollout(plan["G"], plan["reference"], E0, iter(actual), bank["price"], cfg)
            development = fixed_plan_stress(plan["G"], plan["reference"], E0, actual, bank["price"], cfg)
            row = summarize_day(o, date, cfg)
            row.update(model=model.name, alpha=plan["alpha"], plan_feasible=plan["feasible"],
                objective=plan["objective"], constraint_violation=plan["violation"],
                risk_budget=plan["risk_budget"], risk_cvar=plan["risk_cvar"],
                stress_budget=plan["stress_budget"], design_stress_cost=plan["stress_cost"],
                forecast_train_end=forecast["meta"]["train_end"],
                scene_end=max(scenes["dates"]) if scenes["dates"] else None,
                forecast_risk=forecast_risk, effective_scenarios=audit_scenes["diagnostics"]["effective_n"],
                var_exceeded=o["emergency_cost"] > forecast_risk["emergency_var"]+1e-6,
                development_stress=development,
                development_stress_worst_cost=max(s["emergency_cost"] for s in development.values()),
                seed=seed_log, optimizer=plan["log"], locked_plan_sha256=file_hash(lock))
            _store_execution(directory, date, o, row, plan["reference"],
                dict(Q=plan["Q"], worst_q=plan["worst_q"],
                     audit_weights=audit_scenes["weights"], **{"audit_"+k: v for k,v in audit_metrics.items()}))
        rows.append(row)
        E0 = row["Eend"]
        if len(rows) == 1 or len(rows) % 5 == 0 or len(rows) == len(indices):
            print(json.dumps(dict(phase=phase, model=model.name, days=len(rows), date=date,
                  cash_cost=sum(r["cash_cost"] for r in rows), emergency_kwh=sum(r["H"] for r in rows),
                  infeasible_days=sum(not r["plan_feasible"] for r in rows)), ensure_ascii=False), flush=True)
    complete = len(rows) == len(full_indices)
    summary = dict(model=model.name, complete=complete, **risk_totals(rows, cfg))
    write_json(directory/"summary.json", summary)
    write_json(directory/"complete.json", dict(complete=complete, days=len(rows), expected_days=len(full_indices)))
    return summary


def run_models(out, models, phase, workers=1, max_days=None):
    models = list(dict.fromkeys(models))
    if not models:
        raise ValueError("Choose at least one model")
    if workers == 1:
        return {m: run_candidate(out, m, phase, max_days) for m in models}
    results = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(models))) as pool:
        jobs = {pool.submit(run_candidate, str(out), m, phase, max_days): m for m in models}
        for future in as_completed(jobs):
            results[jobs[future]] = future.result()
    return {m: results[m] for m in models}


def choose_model(summaries, cfg):
    if "M2" not in summaries:
        raise ValueError("M2 is required as the common statistical cost/risk baseline")
    baseline = summaries["M2"]
    cap = baseline["cash_cost"]*(1+cfg.cost_premium)
    records = {}
    for name, summary in summaries.items():
        ratios = [summary[k]/max(baseline[k], 1e-6)
                  if baseline[k] > 1e-6 or summary[k] > 1e-6 else 1.0
                  for k in ("emergency_cvar", "development_stress_cvar")]
        records[name] = dict(cost_premium=summary["cash_cost"]/baseline["cash_cost"]-1,
            emergency_tail_ratio=ratios[0], development_stress_tail_ratio=ratios[1],
            risk_score=max(ratios), cost_eligible=summary["cash_cost"] <= cap+1e-6,
            feasible=summary["risk_budget_unmet_days"] == 0 and summary["max_physical_residual"] <= cfg.physical_tolerance)
    eligible = [m for m,r in records.items() if r["cost_eligible"] and r["feasible"]]
    selected = min(eligible, key=lambda m: (records[m]["risk_score"], summaries[m]["cash_cost"], m)) if eligible else None
    return dict(selected=selected, baseline="M2", cost_premium_limit=cfg.cost_premium,
        validation_cash_cap=cap, comparison=records, pareto_models=pareto_frontier(summaries),
        status="selected" if selected else "no_eligible_candidate",
        risk_improvement_observed=selected is not None and records[selected]["risk_score"] < 1-1e-8)


def calibrate(out, models=DEFAULT_MODELS, workers=1):
    out = Path(out)
    cfg, protocol = load_run(out)
    models = list(dict.fromkeys(models))
    if "M2" not in models:
        raise ValueError("Calibration must include M2")
    frozen = out/"selection_frozen.json"
    if frozen.exists():
        selection = read_json(frozen)
        if selection["models"] != models or selection["protocol_sha256"] != file_hash(out/"protocol_frozen.json"):
            raise ValueError("Selection already frozen with other candidates or settings")
        return selection
    if (out/"evaluation_bank.npz").exists():
        raise ValueError("Cannot calibrate after preparing evaluation data in this run")
    summaries = run_models(out, models, "validation", workers)
    if not all(s["complete"] and s["days"] == 17 for s in summaries.values()):
        raise ValueError("All candidates require the complete January validation")
    selection = dict(**choose_model(summaries, cfg), models=models, evaluations=summaries,
        selection_dates=list(VALIDATION), selection_days=17,
        protocol_sha256=file_hash(out/"protocol_frozen.json"),
        january_bank_sha256=file_hash(out/"january_bank.npz"),
        initial_states_sha256=file_hash(out/"initial_states.json"),
        frozen_at=datetime.now().isoformat(), effective_date="2025-02-01",
        february_december_outcomes_used=False, heldout_stress_used=False,
        retrospective_redesign=True, future_cost_cap_guarantee=False)
    write_json(out/"validation_comparison.json", selection)
    if selection["selected"] is not None:
        write_json(frozen, selection)
    return selection


def evaluate(out, models=None, workers=1, max_days=None):
    out = Path(out)
    prepare_evaluation(out)
    selection = read_json(out/"selection_frozen.json")
    models = models or list(dict.fromkeys([selection["selected"], "M2"]))
    results = run_models(out, models, "evaluation", workers, max_days)
    cfg, _ = load_run(out)
    summary = dict(selected=selection["selected"], models=results,
                   selection_unchanged=True, complete=all(r["complete"] for r in results.values()))
    if "M2" in results and selection["selected"] in results:
        chosen, base = results[selection["selected"]], results["M2"]
        premium = chosen["cash_cost"]/base["cash_cost"]-1
        summary.update(realized_cost_premium=premium, cost_premium_limit=cfg.cost_premium,
                       realized_cost_cap_met=premium <= cfg.cost_premium+1e-9,
                       note="3%等费用上限用于一月选型；后续实测是否满足另行核验，不构成保证")
    write_json(out/"evaluation_summary.json", summary)
    return summary
