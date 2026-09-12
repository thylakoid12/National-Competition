"""M3的总变差概率集合。"""

import numpy as np


def check(Q, w, rho):
    Q, w = (np.asarray(Q, float), np.asarray(w, float))
    if Q.ndim != 1 or not len(Q) or Q.shape != w.shape or (not np.isfinite(Q).all()):
        raise ValueError("Invalid cost vector")
    if (
        not np.isfinite(w).all()
        or (w < 0).any()
        or abs(w.sum() - 1) > 1e-08
        or (not 0 <= rho <= 1)
    ):
        raise ValueError("Invalid nominal probabilities or radius")
    return (Q, w)


def worst_case_tv(Q, w, rho):
    Q, w = check(Q, w, rho)
    q = w.copy()
    order = np.argsort(Q, kind="stable")
    receiver = order[-1]
    remaining = min(rho, 1 - q[receiver])
    for donor in order:
        if remaining <= 0 or Q[donor] >= Q[receiver]:
            break
        move = min(remaining, q[donor])
        q[donor] -= move
        q[receiver] += move
        remaining -= move
    assert 0.5 * np.abs(q - w).sum() <= rho + 1e-08
    return (float(Q @ q), q)


def weighted_var_cvar(values, weights, beta=0.9):
    """离散分布的精确分位点与尾部均值，包含分位点的部分概率质量。"""
    values, weights = check(values, weights, 0.0)
    if not 0 <= beta < 1:
        raise ValueError("CVaR beta must belong to [0,1)")
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    index = min(int(np.searchsorted(np.cumsum(weights), beta, side="left")), len(values)-1)
    var = float(values[index])
    # 用 Rockafellar–Uryasev 表达式处理离散原子，比取条件平均更精确。
    cvar = var + float(weights @ np.maximum(values-var, 0.0)) / (1-beta)
    return var, cvar


def worst_case_cvar(values, weights, rho, beta=0.9):
    """TV 球内的最坏 CVaR；成本从小到大搬至最大值的一阶随机占优分布。

    对任意低于最大成本的阈值 x，该分布的 CDF = max(F_w(x)-rho,0)，
    达到 TV 下界。因此同时最大化一切递增损失函数期望及 CVaR。
    注意必须按本函数的 values 排序，不能复用另一种成本的最坏概率。
    tests/test_statistical_risk.py 用独立线性规划逐例验证。
    """
    _, q = worst_case_tv(values, weights, rho)
    return weighted_var_cvar(values, q, beta)[1], q


def risk_summary(metrics, weights, price, G, cfg):
    """共用历史场景上的事前风险；对所有模型使用同一评估口径。"""
    K = metrics["emergency_cost"]
    var, cvar = weighted_var_cvar(K, weights, cfg.risk_beta)
    robust_cvar, q = worst_case_cvar(K, weights, cfg.radius, cfg.risk_beta)
    cash_var, cash_cvar = float(price @ G) + var, float(price @ G) + cvar
    return dict(beta=cfg.risk_beta, mean_emergency_cost=float(weights @ K),
                emergency_var=var, emergency_cvar=cvar,
                robust_emergency_cvar=robust_cvar,
                cash_var=cash_var, cash_cvar=cash_cvar,
                emergency_probability=float(weights @ (metrics["emergency_kwh"] > 1e-6)),
                mean_end_energy=float(weights @ metrics["end_energy"]),
                low_end_energy=weighted_var_cvar(metrics["end_energy"], weights, 0.1)[0],
                minimum_energy=float(metrics["min_energy"].min()),
                worst_emergency_kwh=float(metrics["emergency_kwh"].max()),
                effective_n=float(1.0 / (weights @ weights)),
                worst_case_probabilities=q.tolist())
