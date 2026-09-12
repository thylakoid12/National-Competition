"""Scenario-cost optimization of a 144-slot plan and optional battery reserve.
All candidates use issued forecasts and past residuals, never current actuals.
"""

from time import perf_counter
import numpy as np
from .solver import FastReference, fast_batch
from .risk import worst_case_tv


class PolicyEvaluator:

    def __init__(self, forecast, scenes, E0, price, cfg):
        self.forecast, self.scenes, self.E0, self.price, self.cfg = (
            forecast,
            scenes,
            E0,
            price,
            cfg,
        )
        self.nu = cfg.terminal(forecast["meta"]["date"], price)
        self.solver = FastReference(forecast["path"], E0, price, self.nu, cfg)
        self.refs = {}
        self.cache = {}
        self.evaluations = 0
        self.lp_calls = 0

    def evaluate(self, G, alpha):
        if not 0 <= alpha <= 1:
            raise ValueError("Reserve factor outside [0,1]")
        gkey = np.asarray(G, dtype="<f8").tobytes()
        key = (gkey, float(alpha))
        if key in self.cache:
            return self.cache[key]
        self.evaluations += 1
        if gkey not in self.refs:
            self.refs[gkey] = self.solver.solve(G)
            self.lp_calls += 2
        raw, lp = self.refs[gkey]
        ref = (
            raw.copy()
            if alpha == 1.0
            else self.cfg.e_min + alpha * (raw - self.cfg.e_min)
        )
        ref[0] = self.E0
        Q = fast_batch(
            G, ref, self.E0, self.scenes["paths"], self.price, self.nu, self.cfg
        )
        risk, q = worst_case_tv(Q, self.scenes["weights"], self.cfg.radius)
        result = (float(self.price @ G + risk), ref, q, Q)
        self.cache[key] = result
        return result

    def prune(self, G, alpha):
        gkey = np.asarray(G, dtype="<f8").tobytes()
        self.refs = {gkey: self.refs[gkey]}
        self.cache = {(gkey, float(alpha)): self.cache[gkey, float(alpha)]}


def optimize(forecast, scenes, E0, price, initial_G, cfg, learn_reserve=True):
    start = perf_counter()
    ev = PolicyEvaluator(forecast, scenes, E0, price, cfg)
    G = np.asarray(initial_G, dtype=float).copy()
    alpha = 1.0
    value = ev.evaluate(G, alpha)[0]
    initial_value = value
    accepted = []

    def choose(trials, tag):
        nonlocal G, alpha, value
        best = (value, G, alpha)
        for candidate, a in trials:
            candidate = np.maximum(np.asarray(candidate, dtype=float), 0.0)
            v = ev.evaluate(candidate, float(a))[0]
            if v < best[0] - cfg.objective_tolerance:
                best = (v, candidate, float(a))
        if best[0] < value - cfg.objective_tolerance:
            accepted.append(dict(move=tag, old=value, new=best[0], alpha=best[2]))
            value, G, alpha = best

    def reserve(step=None):
        if not learn_reserve:
            return
        aa = (
            [0.0, 0.25, 0.5, 0.75, 1.0]
            if step is None
            else [max(0.0, alpha - step), min(1.0, alpha + step)]
        )
        choose([(G, a) for a in aa], "reserve")

    reserve()
    net = scenes["paths"][:, :, 0] - scenes["paths"][:, :, 1]
    predicted = forecast["path"][:, 0] - forecast["path"][:, 1]
    excess = np.maximum(np.quantile(net - predicted[None], 0.8, axis=0), 0.0)
    seeds = [
        G * 1.02,
        G * 1.05,
        G + 0.5 * excess,
        G + excess,
        np.maximum(np.mean(net, axis=0), 0.0),
    ]
    choose([(s, alpha) for s in seeds], "profile_start")
    reserve()
    for scale, astep in zip((0.1, 0.02, 0.005), (0.1, 0.05, 0.025)):
        step = scale * cfg.B
        for width in (12, 6, 1):
            for t in range(0, 144, width):
                trials = []
                for sign in (1, -1):
                    trial = G.copy()
                    trial[t : t + width] = np.maximum(
                        0.0, trial[t : t + width] + sign * step
                    )
                    trials.append((trial, alpha))
                choose(trials, f"width{width}_step{step:.6f}")
            reserve(astep)
            ev.prune(G, alpha)
    reserve(0.025)
    objective, ref, q, Q = ev.evaluate(G, alpha)
    assert objective <= initial_value + 1e-07 and np.all(G >= 0)
    return dict(
        G=G,
        reference=ref,
        alpha=alpha,
        objective=objective,
        Q=Q,
        worst_q=q,
        log=dict(
            initial_objective=initial_value,
            final_objective=objective,
            evaluations=ev.evaluations,
            lp_calls=ev.lp_calls,
            accepted=accepted,
            seconds=perf_counter() - start,
            full_144_slot_freedom=True,
            global_optimum_claim=False,
        ),
    )


class RiskPolicyEvaluator:
    """显式现金风险约束；输入只包含决策时已知的预测和过去场景。"""

    def __init__(self, forecast, scenes, E0, price, cfg, model):
        from .models import get_model
        from .stress import design_paths
        self.forecast, self.scenes, self.E0, self.price, self.cfg = forecast, scenes, E0, price, cfg
        self.model = get_model(model)
        self.nu = cfg.terminal(forecast["meta"]["date"], price)
        self.solver = FastReference(forecast["path"], E0, price, self.nu, cfg)
        self.refs, self.cache = {}, {}
        self.evaluations = self.lp_calls = 0
        self.count = len(scenes["paths"])
        self.paths = (np.concatenate([scenes["paths"], design_paths(forecast)])
                      if self.model.stress_guard else scenes["paths"])
        net = np.maximum(forecast["path"][:, 0]-forecast["path"][:, 1], 0.0)
        # 预算规模仅依赖已发布点预测，与被优化的 G 无关，避免通过多买电放宽限额。
        self.budget_scale = max(float(price @ net), float(min(price)*cfg.B))
        self.risk_budget = cfg.risk_budget_ratio*self.budget_scale if self.model.risk_limit else None
        self.stress_budget = cfg.stress_budget_ratio*self.budget_scale if self.model.stress_guard else None

    def evaluate(self, G, alpha):
        from .solver import batch_metrics
        from .risk import worst_case_cvar
        G = np.asarray(G, dtype=float)
        if G.shape != (144,) or not np.isfinite(G).all() or (G < 0).any() or not 0 <= alpha <= 1:
            raise ValueError("Invalid purchase plan or reserve factor")
        if self.model.fixed_alpha is not None and alpha != self.model.fixed_alpha:
            raise ValueError("This ablation fixes the reserve factor")
        gkey = np.asarray(G, dtype="<f8").tobytes()
        key = (gkey, float(alpha))
        if key in self.cache:
            return self.cache[key]
        self.evaluations += 1
        if alpha == 0.0:
            # 此时参考线严格等于 E_min，原 LP 的结果在数学上被乘以零。
            ref = np.full(145, self.cfg.e_min)
            lp = dict(skipped="alpha_zero_reference_constant", residual=0.0)
        else:
            # LP 只依赖裁剪后的充放电上界；超出相同边界的 G 共享完全相同的 LP。
            balance = G+self.forecast["path"][:, 1]-self.forecast["path"][:, 0]
            upper = np.r_[np.minimum(np.maximum(balance, 0), self.cfg.B),
                          np.minimum(np.maximum(-balance, 0), self.cfg.B)]
            rkey = upper.astype("<f8").tobytes()
            if rkey not in self.refs:
                self.refs[rkey] = self.solver.solve(G)
                self.lp_calls += 2
            raw, lp = self.refs[rkey]
            # 缓存的是相同上界下的参考轨迹，购电相关的目标常数可能不同。
            lp = {k: v for k, v in lp.items() if k != "primary_objective"}
            ref = self.cfg.e_min + alpha*(raw-self.cfg.e_min)
        ref[0] = self.E0
        all_metrics = batch_metrics(G, ref, self.E0, self.paths, self.price, self.cfg)
        metrics = {k: v[:self.count] for k, v in all_metrics.items()}
        Q = metrics["emergency_cost"]-self.nu*(metrics["end_energy"]-self.cfg.e_min)
        expectation, q = worst_case_tv(Q, self.scenes["weights"], self.model.radius(self.cfg))
        tail = None
        violation = 0.0
        if self.risk_budget is not None:
            tail, _ = worst_case_cvar(metrics["emergency_cost"], self.scenes["weights"],
                                     self.model.radius(self.cfg), self.cfg.risk_beta)
            violation += max(0.0, tail-self.risk_budget-1e-6)/max(1.0, self.risk_budget)
        stress = None
        if self.stress_budget is not None:
            stress = float(all_metrics["emergency_cost"][self.count:].max())
            violation += max(0.0, stress-self.stress_budget-1e-6)/max(1.0, self.stress_budget)
        result = dict(G=G.copy(), alpha=float(alpha), reference=ref,
                      objective=float(self.price @ G+expectation), Q=Q, worst_q=q,
                      metrics=metrics, violation=float(violation), feasible=violation == 0,
                      risk_cvar=tail, risk_budget=self.risk_budget,
                      stress_cost=stress, stress_budget=self.stress_budget,
                      budget_scale=self.budget_scale, reference_lp=lp)
        self.cache[key] = result
        return result


def _better(candidate, incumbent, tolerance):
    if candidate["feasible"] != incumbent["feasible"]:
        return candidate["feasible"]
    if not candidate["feasible"]:
        delta = candidate["violation"]-incumbent["violation"]
        if abs(delta) > 1e-10:
            return delta < 0
    return candidate["objective"] < incumbent["objective"]-tolerance


def optimize_risk(forecast, scenes, E0, price, initial_G, cfg, model):
    """独立多起点、分阶段预算的重复坐标搜索；无全局最优性声明。"""
    started = perf_counter()
    ev = RiskPolicyEvaluator(forecast, scenes, E0, price, cfg, model)
    net = np.maximum(forecast["path"][:, 0]-forecast["path"][:, 1], 0.0)
    seeds = [np.maximum(initial_G, 0.0), net, net*1.1]
    seed_names = ["deterministic_schedule", "point_net_load", "point_net_load_110pct"]
    if ev.model.risk_limit or ev.model.stress_guard:
        # 约束可行性起点：覆盖所用全部场景的净负荷，故每段都无需紧急补购。
        # 它只属于显式风险组件，并占用同样的总评价预算。
        seeds[2] = np.maximum((ev.paths[:, :, 0]-ev.paths[:, :, 1]).max(axis=0), 0.0)
        seed_names[2] = "finite_scenario_feasibility_seed"
    seeds = seeds[:cfg.search_starts]
    initial_alpha = ev.model.fixed_alpha if ev.model.fixed_alpha is not None else 1.0
    initial = ev.evaluate(seeds[0], initial_alpha)
    best = initial
    accepted, starts = [], []
    quota = cfg.search_evaluations // len(seeds)
    cheap = np.argsort(price, kind="stable")[::12]
    expensive = np.argsort(-np.asarray(price), kind="stable")[::12]

    for start_index, seed in enumerate(seeds):
        start_count = ev.evaluations
        stop_at = min(cfg.search_evaluations, start_count+quota)
        current = ev.evaluate(seed, initial_alpha)
        phases = []

        def choose(trials, tag, cap):
            nonlocal current, best
            changed = False
            for plan, alpha in trials:
                if ev.evaluations >= cap:
                    break
                trial = ev.evaluate(np.maximum(plan, 0.0), float(alpha))
                if _better(trial, current, cfg.objective_tolerance):
                    accepted.append(dict(start=start_index, move=tag,
                        old=current["objective"], new=trial["objective"],
                        violation=trial["violation"], alpha=float(alpha)))
                    current, changed = trial, True
                if _better(trial, best, cfg.objective_tolerance):
                    best = trial
            return changed

        if _better(current, best, cfg.objective_tolerance):
            best = current
        if ev.model.fixed_alpha is None:
            choose([(current["G"], a) for a in (0, 0.25, 0.5, 0.75, 1)], "reserve_grid", stop_at)
        if start_index == 2 and (ev.model.risk_limit or ev.model.stress_guard):
            # 从保守可行种子向共同的便宜种子作整条曲线移动；每个试点都实际核验约束，
            # 不假设 CVaR 随插值单调，也不只靠逐时微调来消除大规模多购电。
            repair = seed.copy()
            for target in (seeds[0], net):
                choose([((1-t)*repair+t*target, current["alpha"])
                        for t in (0.25, 0.5, 0.75, 0.875, 0.9375, 0.96875, 1)],
                       "feasible_profile_interpolation", stop_at)
        # 每个起点都完整经过粗、中、细三个阶段；每阶段循环到无改进或其预算用尽。
        for scale, width, astep, fraction in zip(cfg.search_step_scales, (12, 6, 1),
                                               (0.1, 0.05, 0.025), (0.2, 0.4, 1.0)):
            cap = min(stop_at, start_count+max(1, int(quota*fraction)))
            step = scale*cfg.B
            sweeps, converged = 0, False
            while ev.evaluations < cap:
                changed = False
                completed = True
                for t in range(0, 144, width):
                    if ev.evaluations >= cap:
                        completed = False
                        break
                    G, alpha = current["G"], current["alpha"]
                    trials = []
                    for sign in (-1, 1):
                        candidate = G.copy()
                        candidate[t:t+width] = np.maximum(0.0, candidate[t:t+width]+sign*step)
                        trials.append((candidate, alpha))
                    changed |= choose(trials, f"block_{width}_step_{step:.6g}", cap)
                if ev.model.fixed_alpha is None:
                    a = current["alpha"]
                    changed |= choose([(current["G"], max(0.0, a-astep)),
                                       (current["G"], min(1.0, a+astep))], "reserve_local", cap)
                # 同时移动两个时段，允许单坐标难以跨越的费用/风险权衡。
                for low, high in zip(cheap, expensive):
                    if ev.evaluations >= cap:
                        completed = False
                        break
                    candidates = []
                    for sign in (-1, 1):
                        candidate = current["G"].copy()
                        amount = min(step, candidate[high if sign == 1 else low])
                        candidate[low] += sign*amount
                        candidate[high] -= sign*amount
                        candidates.append((candidate, current["alpha"]))
                    changed |= choose(candidates, "paired_time_shift", cap)
                sweeps += 1
                if completed and not changed:
                    converged = True
                    break
            phases.append(dict(step_kwh=step, width=width, sweeps=sweeps,
                               stop="no_improvement" if converged else "phase_budget"))
        starts.append(dict(seed=seed_names[start_index], evaluations=ev.evaluations-start_count,
                           objective=current["objective"], feasible=current["feasible"], phases=phases))
    best = dict(best)
    best["log"] = dict(initial_objective=initial["objective"], final_objective=best["objective"],
        initial_violation=initial["violation"], final_violation=best["violation"],
        evaluations=ev.evaluations, evaluation_budget=cfg.search_evaluations,
        lp_calls=ev.lp_calls, starts=starts, accepted=accepted, seconds=perf_counter()-started,
        status="feasible" if best["feasible"] else "risk_budget_unmet",
        stop_reason="completed_budgeted_multistart", global_optimum_claim=False,
        full_144_slot_freedom=True)
    return best
