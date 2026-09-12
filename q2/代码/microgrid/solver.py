"""Optional acceleration of the SAME LP and causal state recurrence.

HiGHS API: https://ergo-code.github.io/HiGHS/stable/interfaces/python/example-py/
Reset the solver basis before each candidate so call order cannot select a
different reference trajectory. Reuse model structure, not prior G's basis.
"""

import numpy as np
from scipy.sparse import vstack, csr_matrix
from .controller import reference_structure


class FastReference:

    def __init__(self, forecast, E0, price, nu, cfg):
        import highspy

        self.highspy = highspy
        self.forecast, self.E0, self.price, self.nu, self.cfg = (
            forecast,
            E0,
            price,
            nu,
            cfg,
        )
        n = len(price)
        self.n = n
        self.c = np.zeros(3 * n + 1)
        self.c[n : 2 * n] = -cfg.emergency_multiplier * price
        self.c[-1] = -nu
        self.secondary = np.zeros_like(self.c)
        self.secondary[2 * n + 1 :] = -1
        self.A = reference_structure(n, cfg.eta_c, cfg.eta_d)
        matrix = vstack([self.A, csr_matrix(self.c[None])]).tocsc()
        self.indices = np.arange(3 * n + 1, dtype=np.int32)
        self.flow_indices = np.arange(2 * n, dtype=np.int32)
        lp = highspy.HighsLp()
        lp.num_col_ = 3 * n + 1
        lp.num_row_ = n + 1
        lp.col_cost_ = self.c
        lp.col_lower_ = np.r_[np.zeros(2 * n), E0, np.full(n, cfg.e_min)]
        lp.col_upper_ = np.r_[np.zeros(2 * n), E0, np.full(n, cfg.e_max)]
        lp.row_lower_ = np.r_[np.zeros(n), -np.inf]
        lp.row_upper_ = np.r_[np.zeros(n), np.inf]
        lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
        lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
        lp.a_matrix_.value_ = matrix.data
        self.h = highspy.Highs()
        for k, v in [
            ("output_flag", False),
            ("threads", 1),
            ("solver", "simplex"),
            ("random_seed", cfg.seed),
            ("dual_feasibility_tolerance", 1e-08),
            ("primal_feasibility_tolerance", 1e-08),
        ]:
            self.h.setOptionValue(k, v)
        self.h.passModel(lp)

    def solve(self, G):
        n, cfg, h = (self.n, self.cfg, self.h)
        a = G + self.forecast[:, 1] - self.forecast[:, 0]
        upper = np.r_[
            np.minimum(np.maximum(a, 0), cfg.B), np.minimum(np.maximum(-a, 0), cfg.B)
        ]
        h.changeColsBounds(2 * n, self.flow_indices, np.zeros(2 * n), upper)
        h.changeRowBounds(n, -np.inf, np.inf)
        h.changeColsCost(len(self.c), self.indices, self.c)
        h.clearSolver()
        h.run()
        if h.getModelStatus() != self.highspy.HighsModelStatus.kOptimal:
            raise RuntimeError("Native reference primary: " + str(h.getModelStatus()))
        first = h.getObjectiveValue()
        h.changeRowBounds(n, -np.inf, first + cfg.objective_tolerance)
        h.changeColsCost(len(self.c), self.indices, self.secondary)
        h.run()
        if h.getModelStatus() != self.highspy.HighsModelStatus.kOptimal:
            raise RuntimeError("Native reference secondary: " + str(h.getModelStatus()))
        x = np.array(h.getSolution().col_value)
        residual = max(
            float(np.max(np.abs(self.A @ x))),
            max(0.0, float(self.c @ x - first - cfg.objective_tolerance)),
            max(
                0.0,
                float(cfg.e_min - x[2 * n + 1 :].min()),
                float(x[2 * n + 1 :].max() - cfg.e_max),
            ),
            max(0.0, float((x[: 2 * n] - upper).max()), float(-x[: 2 * n].min())),
        )
        if residual > cfg.physical_tolerance:
            raise RuntimeError("Native reference residual: " + str(residual))
        return (
            x[2 * n :],
            dict(
                primary_status=0,
                secondary_status=0,
                residual=residual,
                primary_objective=float(
                    first
                    + cfg.emergency_multiplier * self.price @ np.maximum(-a, 0)
                    + self.nu * cfg.e_min
                ),
                backend="highspy-" + self.h.version(),
                deterministic_basis_reset=True,
            ),
        )


def _batch(
    G, reference, E0, paths, price, nu, e_min, e_max, B, eta_c, eta_d, multiplier
):
    result = np.empty(len(paths))
    for s in range(len(paths)):
        E = E0
        cost = 0.0
        for t in range(len(G)):
            a = G[t] + paths[s, t, 1] - paths[s, t, 0]
            C = min(max(a, 0.0), B, max((e_max - E) / eta_c, 0.0))
            D = min(max(-a, 0.0), B, eta_d * max(E - reference[t + 1], 0.0))
            cost += multiplier * price[t] * (max(-a, 0.0) - D)
            E += eta_c * C - D / eta_d
        result[s] = cost - nu * (E - e_min)
    return result


_compiled = None


def _metrics_batch(G, reference, E0, paths, price, e_min, e_max, B, eta_c, eta_d, multiplier):
    result = np.zeros((len(paths), 9))
    for s in range(len(paths)):
        E, minimum = E0, E0
        for t in range(len(G)):
            a = G[t] + paths[s, t, 1] - paths[s, t, 0]
            C = min(max(a, 0.0), B, max((e_max-E)/eta_c, 0.0))
            D = min(max(-a, 0.0), B, eta_d*max(E-reference[t+1], 0.0))
            H = max(-a, 0.0)-D
            Z = max(a, 0.0)-C
            U = min(G[t], Z)
            E += eta_c*C-D/eta_d
            minimum = min(minimum, E)
            result[s, 0] += multiplier*price[t]*H
            result[s, 1] += H
            result[s, 2] = max(result[s, 2], H)
            result[s, 5] += U
            result[s, 6] += Z-U
            result[s, 7] += C
            result[s, 8] += D
        result[s, 3], result[s, 4] = minimum, E
    return result


_metrics_compiled = None
METRIC_KEYS = ("emergency_cost", "emergency_kwh", "peak_emergency_kwh",
               "min_energy", "end_energy", "unused", "curtailment", "charge", "discharge")


def batch_metrics(G, reference, E0, paths, price, cfg):
    """与逐时控制器一致的场景模拟，保留费用与库存的独立分解。"""
    global _metrics_compiled
    paths = np.asarray(paths, float)
    if paths.ndim != 3 or paths.shape[1:] != (len(G), 2) or not np.isfinite(paths).all() or (paths < 0).any():
        raise ValueError("Invalid scenario paths")
    if _metrics_compiled is None:
        from numba import njit
        _metrics_compiled = njit(cache=True, fastmath=False)(_metrics_batch)
    matrix = _metrics_compiled(np.asarray(G, float), np.asarray(reference, float),
        float(E0), paths, np.asarray(price, float), cfg.e_min, cfg.e_max, cfg.B,
        cfg.eta_c, cfg.eta_d, cfg.emergency_multiplier)
    return {key: matrix[:, i] for i, key in enumerate(METRIC_KEYS)}


def fast_batch(G, reference, E0, paths, price, nu, cfg):
    global _compiled
    if _compiled is None:
        from numba import njit

        _compiled = njit(cache=True, fastmath=False)(_batch)
    return _compiled(
        G,
        reference,
        E0,
        paths,
        price,
        nu,
        cfg.e_min,
        cfg.e_max,
        cfg.B,
        cfg.eta_c,
        cfg.eta_d,
        cfg.emergency_multiplier,
    )
