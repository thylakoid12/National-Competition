"""从过去的整日误差构造场景与条件概率。"""

import numpy as np


def compute_weights(current, historical, h=1.0, shrinkage=0.25):
    historical = np.asarray(historical, float)
    if not len(historical):
        return (np.empty(0), dict(effective_n=0))
    if h <= 0 or not 0 <= shrinkage <= 1:
        raise ValueError("Invalid kernel settings")
    mu = historical[:, 7:].mean(axis=0)
    scale = historical[:, 7:].std(axis=0)
    numeric = np.divide(
        historical[:, 7:] - current[7:],
        scale,
        out=np.zeros_like(historical[:, 7:]),
        where=scale > 0,
    )
    delta = np.c_[historical[:, :7] - current[:7], numeric]
    distance = (delta**2).mean(axis=1)
    logw = -distance / (2 * h * h)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    w = (1 - shrinkage) * w + shrinkage / len(w)
    return (
        w,
        dict(
            effective_n=float(1 / (w @ w)),
            mean=mu.tolist(),
            scale=scale.tolist(),
            distance=distance.tolist(),
            bandwidth=h,
            shrinkage=shrinkage,
        ),
    )


def inputs(bank, day, variant, cfg):
    date = str(bank["dates"][day])
    key = "adaptive" if variant == "adaptive_joint" else "f0"
    path = bank[key][day].copy()
    forecast = {
        "path": path,
        "context": bank["contexts"][day].copy(),
        "meta": {"date": date, "train_end": str(bank["dates"][day - 1])},
    }
    start = max(1, day - cfg.residual_window)
    scene_dates = bank["dates"][start:day].tolist()
    if not scene_dates or any(d >= date for d in scene_dates):
        raise ValueError("场景必须来自当日之前的历史")
    residual = bank["actual"][start:day] - bank[key][start:day]
    weights, _ = compute_weights(
        forecast["context"], bank["contexts"][start:day], cfg.bandwidth, cfg.shrinkage
    )
    return forecast, {
        "paths": np.maximum(path[None] + residual, 0),
        "weights": weights,
        "dates": scene_dates,
    }
