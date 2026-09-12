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
        pv_utilization=1 - r["curtailment"] / r["pv"],
    )
    return r
