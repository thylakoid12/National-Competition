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
