"""一月滚动选型、固定方案和逐日连续评价。"""

from pathlib import Path
from time import perf_counter
from datetime import datetime
import json
import numpy as np
from .config import CODE_DIR, ATTACHMENTS
from .data import load_data
from .forecast import Forecaster, rolling
from .scenarios import inputs
from .seed import seed_plan
from .optimizer import optimize
from .controller import rollout
from .metrics import totals
from .storage import evaluation_choice
from .storage import read_json, read_arrays, write_json, write_arrays, file_hash

VARIANTS = ("search_only", "joint_reserve", "adaptive_joint")


def lock_protocol(out, cfg, initial_states):
    out = Path(out)
    protocol = dict(
        config=cfg.to_dict(),
        initial_states=read_json(initial_states),
        data_hashes={
            name: file_hash(ATTACHMENTS / name) for name in ("附件1.xlsx", "附件2.xlsx")
        },
        validation=["2025-01-15", "2025-01-31"],
        evaluation=["2025-02-01", "2025-12-31"],
        variants=list(VARIANTS),
        selection_rule="Minimum January cash cost; candidate order breaks ties",
        retrospective_redesign=True,
        online_updates="Past observations only; fixed model family and hyperparameters after January",
    )
    path = out / "protocol_frozen.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Run settings changed; use a new output directory.")
    write_json(path, protocol)
    write_json(out / "initial_states.json", protocol["initial_states"])


def prepare_bank(out, cfg, january=True, forecast_archive=None):
    out = Path(out)
    phase = "january" if january else "evaluation"
    path = out / (phase + "_bank.npz")
    if path.exists():
        return read_arrays(path)
    if not january and not (out / "selection_frozen.json").exists():
        raise ValueError("Freeze January selection before preparing evaluation data.")
    data = load_data()
    n = 31 if january else 365
    dates, actual = data.dates[:n], data.values[:n].copy()
    f0, f1, adaptive = (np.zeros_like(actual) for _ in range(3))
    contexts = np.zeros((n, 12))
    engine, losses, records = Forecaster(cfg), [[], []], []
    use_ml = (
        january
        or read_json(out / "selection_frozen.json")["selected"] == "adaptive_joint"
    )
    # An explicit archive is a cache of past issued predictions, never optimized plans.
    if forecast_archive:
        forecast_archive = Path(forecast_archive)
        archived_cfg = read_json(CODE_DIR / "inputs/forecast_archive_config.json")
        if cfg.to_dict() != archived_cfg:
            raise ValueError(
                "Forecast archive requires its original config; omit --forecast-archive to rebuild."
            )
    for i in range(1, n):
        date = str(dates[i])
        issue = engine.issue(actual[:i], date, use_ml=use_ml and not forecast_archive)
        f0[i], contexts[i] = issue["f0"], issue["context"]
        if forecast_archive:
            cached = read_arrays(forecast_archive / "forecasts/F0" / (date + ".npz"))
            meta = read_json(forecast_archive / "forecasts/F1" / (date + ".json"))
            if meta["date"] != date or meta["train_end"] >= date:
                raise ValueError("Noncausal forecast archive")
            if not np.allclose(cached["path"], f0[i], atol=1e-9, rtol=0):
                raise ValueError("Forecast archive does not match the current data.")
            f1[i] = read_arrays(forecast_archive / "forecasts/F1" / (date + ".npz"))[
                "path"
            ]
        else:
            f1[i] = issue["f1"]
        chosen = []
        for t in (0, 1):
            current = np.concatenate(
                [issue["candidates"][t], f1[i, :, t][None]], axis=0
            )
            pred, ids, weights = rolling(current, losses[t])
            adaptive[i, :, t] = np.maximum(f0[i, :, t] if i == 1 else pred, 0)
            chosen.append(dict(candidates=ids, weights=weights))
            losses[t].append(np.abs(current - actual[i, :, t][None]).mean(axis=1))
        engine.settle(issue, actual[i], date)
        records.append(dict(date=date, train_end=str(dates[i - 1]), models=chosen))
    write_arrays(
        path,
        dates=np.asarray(dates),
        actual=actual,
        price=data.price,
        f0=f0,
        f1=f1,
        adaptive=adaptive,
        contexts=contexts,
    )
    write_json(out / (phase + "_forecast_selection.json"), records)
    write_json(out / (phase + "_forecast_report.json"), engine.report())
    return read_arrays(path)


def run_variant(out, variant, phase, cfg):
    if variant not in VARIANTS or phase not in ("january", "evaluation"):
        raise ValueError("Unknown variant or phase")
    out = Path(out)
    bank = read_arrays(out / (phase + "_bank.npz"))
    if phase == "january":
        if len(bank["dates"]) != 31 or max(bank["dates"]) != "2025-01-31":
            raise ValueError("January selection must only contain January data")
        indices = range(14, 31)
        E0 = read_json(out / "initial_states.json")["validation_initial"]
    else:
        frozen = evaluation_choice(out)
        if frozen["selected"] != variant:
            raise ValueError("Evaluation variant differs from frozen selection")
        if file_hash(out / "january_bank.npz") != frozen["january_bank_sha256"]:
            raise ValueError("January data changed after selection")
        indices = range(31, 365)
        E0 = read_json(out / "initial_states.json")["evaluation_initial"]
    dest = out / phase / variant
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    started = perf_counter()
    for i in indices:
        date = str(bank["dates"][i])
        jp = dest / (date + ".json")
        zp = dest / (date + ".npz")
        if jp.exists():
            r = read_json(jp)
            if r["date"] != date or abs(r["E0"] - E0) >= 1e-6 or not zp.exists():
                raise ValueError(f"Invalid checkpoint: {jp}")
            E0 = r["Eend"]
            rows.append(r)
            continue
        f, s = inputs(bank, i, variant, cfg)
        G0, seed_log = seed_plan(f, E0, bank["price"], cfg)
        plan = optimize(f, s, E0, bank["price"], G0, cfg, variant != "search_only")
        write_arrays(
            dest / "locked_plans" / (date + ".npz"),
            G=plan["G"],
            reference=plan["reference"],
            alpha=np.array(plan["alpha"]),
        )
        actual = bank["actual"][i]
        o = rollout(plan["G"], plan["reference"], E0, iter(actual), bank["price"], cfg)
        r = dict(
            date=date,
            variant=variant,
            E0=E0,
            Eend=float(o["E"][-1]),
            alpha=plan["alpha"],
            G=float(o["G"].sum()),
            H=float(o["H"].sum()),
            plan_cost=o["plan_cost"],
            emergency_cost=o["emergency_cost"],
            cash_cost=o["cash_cost"],
            emergency_days=int(np.any(o["H"] > 1e-06)),
            emergency_slots=int(np.count_nonzero(o["H"] > 1e-06)),
            unused=float(o["U"].sum()),
            curtailment=float(o["W"].sum()),
            pv=float(o["pv"].sum()),
            storage_loss=float(
                (1 - cfg.eta_c) * o["charge"].sum()
                + (1 / cfg.eta_d - 1) * o["discharge"].sum()
            ),
            max_residual=max(o["residuals"].values()),
            forecast_train_end=f["meta"]["train_end"],
            scene_end=max(s["dates"]),
            seed=seed_log,
            optimizer=plan["log"],
            objective=plan["objective"],
        )
        write_arrays(
            zp,
            **{k: v for k, v in o.items() if isinstance(v, np.ndarray)},
            reference=plan["reference"],
            Q=plan["Q"],
            worst_q=plan["worst_q"],
        )
        write_json(jp, r)
        rows.append(r)
        E0 = r["Eend"]
        if len(rows) % 5 == 0 or len(rows) == 1:
            print(
                json.dumps(
                    dict(
                        phase=phase,
                        variant=variant,
                        days=len(rows),
                        last_date=date,
                        cash_cost=sum((x["cash_cost"] for x in rows)),
                        H=sum((x["H"] for x in rows)),
                        seconds=perf_counter() - started,
                    )
                ),
                flush=True,
            )
    write_json(
        dest / "complete.json",
        dict(days=len(rows), complete=True, seconds=perf_counter() - started),
    )
    return rows


def select_january(groups):
    for name, rows in groups.items():
        if name not in VARIANTS or [r["date"] for r in rows] != [
            f"2025-01-{d:02d}" for d in range(15, 32)
        ]:
            raise ValueError("Selection requires January 15–31 for every candidate")
    return min(
        VARIANTS,
        key=lambda name: (
            sum((r["cash_cost"] for r in groups[name])),
            VARIANTS.index(name),
        ),
    )




def select(out, cfg):
    out = Path(out)
    frozen = out / "selection_frozen.json"
    if frozen.exists():
        return read_json(frozen)
    if (out / "evaluation_bank.npz").exists():
        raise ValueError("Evaluation data must not be prepared before selection.")
    groups = {name: run_variant(out, name, "january", cfg) for name in VARIANTS}
    choice = dict(
        selected=select_january(groups),
        selection_dates=["2025-01-15", "2025-01-31"],
        selection_days=17,
        evaluations={name: totals(rows) for name, rows in groups.items()},
        january_bank_sha256=file_hash(out / "january_bank.npz"),
        protocol_sha256=file_hash(out / "protocol_frozen.json"),
        frozen_at=datetime.now().isoformat(),
        effective_date="2025-02-01",
        february_december_outcomes_used=False,
    )
    write_json(frozen, choice)
    return choice


def evaluate(out, cfg, forecast_archive=None):
    out = Path(out)
    choice = read_json(out / "selection_frozen.json")
    if choice["protocol_sha256"] != file_hash(out / "protocol_frozen.json"):
        raise ValueError("Protocol changed after January selection.")
    prepare_bank(out, cfg, january=False, forecast_archive=forecast_archive)
    rows = run_variant(out, choice["selected"], "evaluation", cfg)
    summary = dict(selected=choice["selected"], **totals(rows))
    write_json(out / "evaluation_summary.json", summary)
    return summary
