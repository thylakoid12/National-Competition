"""逐日预测；一月误差用于选型，二月起固定统计方法。"""

from datetime import date as Date, timedelta
import numpy as np
from .config import Config

LOAD_METHODS = ("previous_day", "previous_week", "weekday_2", "weekday_3", "mean_7")
PV_METHODS = ("mean_1", "mean_3", "mean_7", "mean_14", "linear_7", "linear_14")


def context(history, date):
    weekday = np.eye(7)[Date.fromisoformat(date).weekday()]
    if not len(history):
        return np.r_[weekday, np.zeros(5)]
    totals = history[-7:].sum(axis=1)
    return np.r_[
        weekday,
        totals[-1, 0],
        totals[:, 0].mean(),
        totals[-1, 1],
        totals[:, 1].mean(),
        totals[:, 1].std(),
    ]


def candidates(history, target):
    if not len(history):
        raise ValueError("F0 requires at least one completed day")
    y = history[:, :, target]
    result, flags = ([], [])
    for method in LOAD_METHODS if target == 0 else PV_METHODS:
        fallback = False
        if method == "previous_day":
            out = y[-1]
        elif method == "previous_week" or method.startswith("weekday_"):
            weeks = 1 if method == "previous_week" else int(method.split("_")[1])
            indices = [len(y) - 7 * k for k in range(1, weeks + 1) if len(y) >= 7 * k]
            fallback = not indices
            out = y[indices].mean(axis=0) if indices else y.mean(axis=0)
        else:
            kind, k = method.split("_")
            part = y[-int(k) :]
            if kind == "linear" and len(part) >= 2:
                x = np.arange(len(part), dtype=float)
                slope = (x - x.mean()) @ part / np.sum((x - x.mean()) ** 2)
                out = part.mean(axis=0) + slope * (len(part) - x.mean())
            else:
                fallback = kind == "linear"
                out = y.mean(axis=0) if fallback else part.mean(axis=0)
        result.append(np.maximum(out, 0))
        flags.append(fallback)
    return (np.array(result), flags)


def features(history, date):
    """144 rows, using only days strictly before date, including for training rows."""
    n = 144
    slot = np.arange(n)
    columns = [slot, np.sin(2 * np.pi * slot / n), np.cos(2 * np.pi * slot / n)]
    columns += [
        np.full(n, float(Date.fromisoformat(date).weekday() == w)) for w in range(7)
    ]
    for lag in (1, 7):
        for target in (0, 1):
            columns.append(
                history[-lag, :, target] if len(history) >= lag else np.full(n, np.nan)
            )
    for window in (3, 7, 14):
        for target in (0, 1):
            part = history[-window:, :, target]
            columns += (
                [part.mean(axis=0), part.std(axis=0)]
                if len(part)
                else [np.full(n, np.nan)] * 2
            )
    if len(history):
        totals = history.sum(axis=1)
        constants = np.r_[totals[-1], totals[-7:].mean(axis=0)]
    else:
        constants = np.full(4, np.nan)
    columns += [np.full(n, v) for v in constants]
    return np.column_stack(columns)


class Forecaster:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scores = [np.zeros(len(LOAD_METHODS)), np.zeros(len(PV_METHODS))]
        self.scored_days = 0
        self.frozen = None

    def issue(self, history, date, use_ml=True):
        if date >= "2025-02-01" and self.frozen is None:
            self.frozen = [int(np.argmin(s)) for s in self.scores]
        arrays = [candidates(history, t)[0] for t in (0, 1)]
        chosen = self.frozen or [int(np.argmin(s)) for s in self.scores]
        f0 = np.column_stack([arrays[t][chosen[t]] for t in (0, 1)])
        f1 = (
            self.predict_ml(history, date)
            if use_ml and len(history) >= self.cfg.ml_min_days
            else f0.copy()
        )
        return dict(f0=f0, f1=f1, candidates=arrays, context=context(history, date))

    def settle(self, issue, actual, date):
        if date < "2025-02-01":
            for t in (0, 1):
                self.scores[t] += np.abs(issue["candidates"][t] - actual[:, t]).mean(
                    axis=1
                )
            self.scored_days += 1

    def report(self):
        chosen = self.frozen or [int(np.argmin(s)) for s in self.scores]
        return dict(
            methods=[LOAD_METHODS[chosen[0]], PV_METHODS[chosen[1]]],
            candidates=[list(LOAD_METHODS), list(PV_METHODS)],
            mae=[(s / self.scored_days).tolist() for s in self.scores],
            scored_days=self.scored_days,
        )

    def predict_ml(self, history, date):
        from catboost import CatBoostRegressor

        cfg = self.cfg
        first = max(0, len(history) - cfg.ml_window)
        start = Date.fromisoformat(date) - timedelta(days=len(history))
        X = np.concatenate(
            [
                features(history[:j], (start + timedelta(days=j)).isoformat())
                for j in range(first, len(history))
            ]
        )
        Y = history[first:].reshape(-1, 2)
        query = features(history, date)
        predictions = []
        for t in (0, 1):
            model = CatBoostRegressor(
                depth=cfg.ml_depth,
                iterations=cfg.ml_iterations,
                learning_rate=cfg.ml_learning_rate,
                loss_function="RMSE",
                random_seed=cfg.seed,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(X, Y[:, t])
            predictions.append(np.maximum(model.predict(query), 0))
        return np.column_stack(predictions)


def rolling(current, losses):
    if not losses:
        return (current[0].copy(), [0], [1.0])
    scores = np.mean(np.asarray(losses[-28:]), axis=0)
    idx = np.argsort(scores, kind="stable")[:2]
    w = 1 / np.maximum(scores[idx], 1e-08) ** 2
    w /= w.sum()
    return (w @ current[idx], idx.tolist(), w.tolist())
