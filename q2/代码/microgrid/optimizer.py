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
