"""储能反馈、能量守恒与结算。"""

from functools import lru_cache
import numpy as np
from scipy.sparse import lil_matrix
from .config import Config


@lru_cache(maxsize=8)
def reference_structure(n, eta_c, eta_d):
    A = lil_matrix((n, 3 * n + 1))
    for t in range(n):
        A[t, [t, n + t, 2 * n + t, 2 * n + t + 1]] = [-eta_c, 1 / eta_d, -1, 1]
    return A.tocsr()


def action(G, R, E, load, pv, cfg=Config()):
    a = G + pv - load
    charge = min(max(a, 0.0), cfg.B, max((cfg.e_max - E) / cfg.eta_c, 0.0))
    discharge = min(max(-a, 0.0), cfg.B, cfg.eta_d * max(E - R, 0.0))
    H = max(-a, 0.0) - discharge
    Z = max(a, 0.0) - charge
    U = min(G, Z)
    W = Z - U
    return (charge, discharge, H, U, W, E + cfg.eta_c * charge - discharge / cfg.eta_d)


def rollout(G, reference, E0, revealed_path_iterator, price, cfg=Config()):
    G = np.array(G, dtype=float, copy=True)
    G.flags.writeable = False
    n = len(G)
    E = np.empty(n + 1)
    E[0] = E0
    out = {k: np.zeros(n) for k in ["load", "pv", "charge", "discharge", "H", "U", "W"]}
    iterator = iter(revealed_path_iterator)
    for t in range(n):
        L, V = next(iterator)
        out["load"][t], out["pv"][t] = (L, V)
        vals = action(G[t], reference[t + 1], E[t], L, V, cfg)
        for key, value in zip(["charge", "discharge", "H", "U", "W"], vals[:-1]):
            out[key][t] = value
        E[t + 1] = vals[-1]
    if next(iterator, None) is not None:
        raise ValueError("Extra observation")
    out.update(
        G=G,
        E=E,
        plan_cost=float(price @ G),
        emergency_cost=float(cfg.emergency_multiplier * price @ out["H"]),
    )
    out["cash_cost"] = out["plan_cost"] + out["emergency_cost"]
    out["residuals"] = validate(out, price, cfg)
    return out


def validate(o, price, cfg=Config()):
    G, E, C, D, H, U, W = [
        o[k] for k in ["G", "E", "charge", "discharge", "H", "U", "W"]
    ]
    errors = dict(
        balance=float(np.max(np.abs(G - U + o["pv"] - W + D + H - o["load"] - C))),
        storage=float(np.max(np.abs(np.diff(E) - cfg.eta_c * C + D / cfg.eta_d))),
        bounds=max(0.0, float(cfg.e_min - E.min()), float(E.max() - cfg.e_max)),
        power=max(0.0, float(max(C.max(), D.max()) - cfg.B)),
        exclusive=float(np.minimum(C, D).max()),
        emergency_charging=float(np.minimum(C, H).max()),
        nonnegative=max(
            0.0,
            -min(
                (float(o[k].min()) for k in ["G", "H", "U", "W", "charge", "discharge"])
            ),
        ),
        unused=max(0.0, float((U - G).max()), float((W - o["pv"]).max())),
        cash=abs(
            o["cash_cost"] - float(price @ G + cfg.emergency_multiplier * price @ H)
        ),
    )
    if max(errors.values()) > cfg.physical_tolerance:
        raise AssertionError(errors)
    return errors
