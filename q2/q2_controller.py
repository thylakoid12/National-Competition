"""Common reference LP and causal V1 control, without per-scenario foresight."""
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
from q2_config import Config

@lru_cache(maxsize=8)
def reference_structure(n, eta_c, eta_d):
    A = lil_matrix((n, 3*n+1))
    for t in range(n):
        A[t, [t, n+t, 2*n+t, 2*n+t+1]] = [-eta_c, 1/eta_d, -1, 1]
    return A.tocsr()

def solve_reference(G, forecast, E0, price, nu, cfg=Config()):
    G, forecast, price = np.asarray(G), np.asarray(forecast), np.asarray(price)
    n = len(G)
    if forecast.shape != (n, 2) or not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError('Invalid point forecast')
    if (G < 0).any() or not np.isfinite(G).all() or not cfg.e_min <= E0 <= cfg.e_max:
        raise ValueError('Invalid plan or E0')
    a = G + forecast[:, 1] - forecast[:, 0]
    bounds = [(0, x) for x in np.minimum(np.maximum(a, 0), cfg.B)]
    bounds += [(0, x) for x in np.minimum(np.maximum(-a, 0), cfg.B)]
    bounds += [(E0, E0)] + [(cfg.e_min, cfg.e_max)] * n
    c = np.zeros(3*n+1)
    c[n:2*n] = -cfg.emergency_multiplier * price
    c[-1] = -nu
    A = reference_structure(n, cfg.eta_c, cfg.eta_d)
    options = dict(dual_feasibility_tolerance=1e-8, primal_feasibility_tolerance=1e-8)
    first = linprog(c, A_eq=A, b_eq=np.zeros(n), bounds=bounds, method='highs-ds', options=options)
    if not first.success:
        raise RuntimeError('Reference primary LP failed: ' + first.message)
    secondary = np.zeros_like(c)
    secondary[2*n+1:] = -1
    second = linprog(secondary, A_ub=csr_matrix(c[None]),
                     b_ub=[first.fun + cfg.objective_tolerance], A_eq=A,
                     b_eq=np.zeros(n), bounds=bounds, method='highs-ds', options=options)
    if not second.success:
        raise RuntimeError('Reference tie-break LP failed: ' + second.message)
    E = second.x[2*n:].copy()
    residual = max(float(np.max(np.abs(A @ second.x))),
                   max(0., float(c @ second.x-first.fun-cfg.objective_tolerance)))
    if residual > cfg.physical_tolerance:
        raise RuntimeError('Reference LP residual exceeds tolerance')
    return E, dict(primary_status=first.status, secondary_status=second.status,
                   primary_objective=float(first.fun+cfg.emergency_multiplier*price@np.maximum(-a, 0)+nu*cfg.e_min),
                   residual=residual)

def action(G, R, E, load, pv, cfg=Config()):
    a = G+pv-load
    charge = min(max(a, 0.), cfg.B, max((cfg.e_max-E)/cfg.eta_c, 0.))
    discharge = min(max(-a, 0.), cfg.B, cfg.eta_d*max(E-R, 0.))
    H = max(-a, 0.)-discharge
    Z = max(a, 0.)-charge
    U = min(G, Z)
    W = Z-U
    return charge, discharge, H, U, W, E+cfg.eta_c*charge-discharge/cfg.eta_d

def rollout(G, reference, E0, revealed_path_iterator, price, cfg=Config()):
    G = np.array(G, dtype=float, copy=True)
    G.flags.writeable = False
    n = len(G)
    E = np.empty(n+1)
    E[0] = E0
    out = {k: np.zeros(n) for k in ['load', 'pv', 'charge', 'discharge', 'H', 'U', 'W']}
    iterator = iter(revealed_path_iterator)
    for t in range(n):
        L, V = next(iterator)
        if not np.isfinite([L, V]).all() or min(L, V) < 0:
            raise ValueError('Invalid revealed observation')
        out['load'][t], out['pv'][t] = L, V
        vals = action(G[t], reference[t+1], E[t], L, V, cfg)
        for key, value in zip(['charge', 'discharge', 'H', 'U', 'W'], vals[:-1]):
            out[key][t] = value
        E[t+1] = vals[-1]
    if next(iterator, None) is not None:
        raise ValueError('Extra observation')
    out.update(G=G, E=E, plan_cost=float(price@G),
               emergency_cost=float(cfg.emergency_multiplier*price@out['H']))
    out['cash_cost'] = out['plan_cost']+out['emergency_cost']
    out['residuals'] = validate(out, price, cfg)
    return out

def validate(o, price, cfg=Config()):
    G,E,C,D,H,U,W = [o[k] for k in ['G','E','charge','discharge','H','U','W']]
    errors = dict(balance=float(np.max(np.abs(G-U+o['pv']-W+D+H-o['load']-C))),
                  storage=float(np.max(np.abs(np.diff(E)-cfg.eta_c*C+D/cfg.eta_d))),
                  bounds=max(0., float(cfg.e_min-E.min()), float(E.max()-cfg.e_max)),
                  power=max(0., float(max(C.max(), D.max())-cfg.B)),
                  exclusive=float(np.minimum(C,D).max()),
                  emergency_charging=float(np.minimum(C,H).max()),
                  nonnegative=max(0., -min(float(o[k].min()) for k in ['G','H','U','W','charge','discharge'])),
                  unused=max(0., float((U-G).max()), float((W-o['pv']).max())),
                  cash=abs(o['cash_cost']-float(price@G+cfg.emergency_multiplier*price@H)))
    if max(errors.values()) > cfg.physical_tolerance:
        raise AssertionError(errors)
    return errors

def batch_costs(G, reference, E0, paths, price, nu, cfg=Config()):
    """Vectorize over scenarios only; advance each day's time sequentially."""
    E = np.full(len(paths), E0, dtype=float)
    cost = np.zeros(len(paths))
    for t in range(len(G)):
        a = G[t]+paths[:,t,1]-paths[:,t,0]
        C = np.minimum(np.minimum(np.maximum(a,0),cfg.B),np.maximum((cfg.e_max-E)/cfg.eta_c,0))
        D = np.minimum(np.minimum(np.maximum(-a,0),cfg.B),cfg.eta_d*np.maximum(E-reference[t+1],0))
        cost += cfg.emergency_multiplier*price[t]*(np.maximum(-a,0)-D)
        E += cfg.eta_c*C-D/cfg.eta_d
    return cost-nu*(E-cfg.e_min)
