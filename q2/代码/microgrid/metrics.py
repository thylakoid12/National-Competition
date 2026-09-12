"""从逐日结算汇总费用、电量和应急指标。"""


def totals(rows):
    keys = (
        "G",
        "H",
        "plan_cost",
        "emergency_cost",
        "cash_cost",
        "emergency_days",
        "emergency_slots",
        "unused",
        "curtailment",
        "pv",
        "storage_loss",
    )
    r = {k: sum((x[k] for x in rows)) for k in keys}
    r.update(
        days=len(rows),
        Eend=rows[-1]["Eend"],
        pv_utilization=1 - r["curtailment"] / r["pv"] if r["pv"] else 1.0,
    )
    return r


def summarize_day(o, date, cfg):
    import numpy as np
    return dict(date=date, E0=float(o["E"][0]), Eend=float(o["E"][-1]),
        min_energy=float(o["E"].min()), G=float(o["G"].sum()), H=float(o["H"].sum()),
        peak_emergency_kw=float(o["H"].max()/ (1/6)),
        plan_cost=o["plan_cost"], emergency_cost=o["emergency_cost"], cash_cost=o["cash_cost"],
        emergency_days=int(np.any(o["H"] > 1e-6)),
        emergency_slots=int(np.count_nonzero(o["H"] > 1e-6)),
        unused=float(o["U"].sum()), curtailment=float(o["W"].sum()), pv=float(o["pv"].sum()),
        storage_loss=float((1-cfg.eta_c)*o["charge"].sum()+(1/cfg.eta_d-1)*o["discharge"].sum()),
        max_residual=max(o["residuals"].values()))


def risk_totals(rows, cfg, bootstrap=True):
    import numpy as np
    from .risk import weighted_var_cvar
    if not rows:
        raise ValueError("No completed days")
    weights = np.full(len(rows), 1/len(rows))
    emergency = np.array([r["emergency_cost"] for r in rows])
    cash = np.array([r["cash_cost"] for r in rows])
    stress = np.array([r.get("development_stress_worst_cost", r["emergency_cost"]) for r in rows])
    summary = totals(rows)
    summary.update(emergency_cvar=weighted_var_cvar(emergency, weights, cfg.risk_beta)[1],
        daily_cash_cvar=weighted_var_cvar(cash, weights, cfg.risk_beta)[1],
        development_stress_cvar=weighted_var_cvar(stress, weights, cfg.risk_beta)[1],
        worst_daily_emergency_cost=float(emergency.max()),
        worst_daily_cash_cost=float(cash.max()),
        min_energy=min(r["min_energy"] for r in rows),
        peak_emergency_kw=max(r["peak_emergency_kw"] for r in rows),
        risk_budget_unmet_days=sum(not r.get("plan_feasible", True) for r in rows),
        max_physical_residual=max(r["max_residual"] for r in rows),
        var_exceedance_rate=float(np.mean([r.get("var_exceeded", False) for r in rows])),
        expected_var_exceedance_rate=1-cfg.risk_beta,
        beta=cfg.risk_beta, small_tail_sample=(1-cfg.risk_beta)*len(rows) < 10,
        start_date=rows[0]["date"], end_date=rows[-1]["date"])
    if bootstrap and cfg.bootstrap_repetitions:
        rng = np.random.default_rng(cfg.seed)
        n, block = len(rows), min(cfg.bootstrap_block_days, len(rows))
        estimates = []
        for _ in range(cfg.bootstrap_repetitions):
            starts = rng.integers(0, n-block+1, size=(n+block-1)//block)
            indices = np.concatenate([np.arange(s, s+block) for s in starts])[:n]
            estimates.append(weighted_var_cvar(emergency[indices], weights, cfg.risk_beta)[1])
        summary["emergency_cvar_bootstrap_interval"] = np.quantile(estimates, [0.025, 0.975]).tolist()
        summary["bootstrap_note"] = "移动日块重采样敏感性区间；不增加独立历史样本，不构成未来安全保证"
    return summary


def pareto_frontier(summaries):
    keys = ("cash_cost", "emergency_cvar", "development_stress_cvar")
    return [name for name, row in summaries.items() if not any(
        other != name and all(value[k] <= row[k]+1e-8 for k in keys)
        and any(value[k] < row[k]-1e-8 for k in keys)
        for other, value in summaries.items())]
