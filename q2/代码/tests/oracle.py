import numpy as np


def scalar_replay(G, reference, E0, actual, price, cfg):
    E = np.empty(145)
    E[0] = E0
    out = {key: np.zeros(144) for key in ("charge", "discharge", "H", "U", "W")}
    for t in range(144):
        surplus = float(G[t] + actual[t, 1] - actual[t, 0])
        if surplus >= 0:
            C = min(surplus, cfg.power / 6, max(0.0, (cfg.e_max - E[t]) / cfg.eta_c))
            D = H = 0.0
            excess = surplus - C
            U = min(float(G[t]), excess)
            W = excess - U
        else:
            C = U = W = 0.0
            D = min(
                -surplus, cfg.power / 6, cfg.eta_d * max(0.0, E[t] - reference[t + 1])
            )
            H = -surplus - D
        for k, v in zip(("charge", "discharge", "H", "U", "W"), (C, D, H, U, W)):
            out[k][t] = v
        E[t + 1] = E[t] + cfg.eta_c * C - D / cfg.eta_d
    out["E"] = E
    out["plan_cost"] = float(sum((float(p) * float(g) for p, g in zip(price, G))))
    out["emergency_cost"] = float(
        sum((5 * float(p) * float(h) for p, h in zip(price, out["H"])))
    )
    out["cash_cost"] = out["plan_cost"] + out["emergency_cost"]
    balance = (
        G
        - out["U"]
        + actual[:, 1]
        - out["W"]
        + out["discharge"]
        + out["H"]
        - actual[:, 0]
        - out["charge"]
    )
    assert np.max(np.abs(balance)) < 1e-06
    assert E.min() >= cfg.e_min - 1e-06 and E.max() <= cfg.e_max + 1e-06
    assert np.max(np.minimum(out["H"], out["charge"])) < 1e-06
    assert np.max(np.minimum(out["discharge"], out["charge"])) < 1e-06
    return out
