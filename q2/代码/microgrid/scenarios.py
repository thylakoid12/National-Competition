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


def build_scenarios(forecast, residuals, historical_contexts, dates, mode, cfg):
    """仅接收已结算历史；该接口不接收包含未来实测的全年数组。"""
    from .models import get_model
    if mode not in ("point", "uniform", "conditional"):
        mode = get_model(mode).scenario_mode
    path = np.asarray(forecast["path"], float)
    if mode == "point":
        return dict(paths=path[None].copy(), weights=np.ones(1), dates=[],
                    diagnostics=dict(mode=mode, effective_n=1.0, history_days=0, scale=[1.0, 1.0]))
    count = min(len(residuals), cfg.residual_window)
    if count == 0:
        result = build_scenarios(forecast, [], [], [], "point", cfg)
        result["diagnostics"]["fallback"] = "no_completed_residuals"
        return result
    residual = np.asarray(residuals[-count:], float)
    historical = np.asarray(historical_contexts[-count:], float)
    scene_dates = list(dates[-count:])
    if len(scene_dates) != count or len(historical) != count or any(
            d >= forecast["meta"]["date"] for d in scene_dates):
        raise ValueError("Scenario history must precede the decision date")
    if residual.shape[1:] != path.shape or not np.isfinite(residual).all():
        raise ValueError("Invalid joint residual paths")
    scale = np.ones(2)
    if cfg.residual_scale and count >= 2 * cfg.scale_recent_window:
        recent = residual[-cfg.scale_recent_window:]
        # 按目标统一缩放一整日，保留时序和负荷/光伏联合的样本对应关系。
        old_std = residual.std(axis=(0, 1))
        scale = np.clip(np.divide(recent.std(axis=(0, 1)), old_std,
                        out=np.ones(2), where=old_std > 1e-9), cfg.scale_min, cfg.scale_max)
    if mode == "uniform":
        weights = np.full(count, 1.0 / count)
        diagnostics = dict(effective_n=float(count))
    else:
        weights, diagnostics = compute_weights(forecast["context"], historical,
                                              cfg.bandwidth, cfg.shrinkage)
    paths = np.maximum(path[None] + residual * scale, 0.0)
    night = forecast["meta"].get("night_slots", [])
    paths[:, night, 1] = 0.0
    diagnostics.update(mode=mode, history_days=count, scale=scale.tolist(),
                       tail_effective_n=(1-cfg.risk_beta)*diagnostics["effective_n"],
                       weak_tail_evidence=(1-cfg.risk_beta)*diagnostics["effective_n"] < 10)
    return dict(paths=paths, weights=weights, dates=scene_dates, diagnostics=diagnostics)
