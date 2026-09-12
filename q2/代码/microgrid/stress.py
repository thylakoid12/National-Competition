"""无假定发生概率的压力情景；开发组与留出组严格分开。"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Shock:
    name: str
    load_factor: float = 1.0
    pv_factor: float = 1.0
    peak_factor: float = 1.0
    duration: int = 1
    initial_fraction: float | None = None

    def to_dict(self):
        return asdict(self)


DEVELOPMENT = (
    Shock("dev_load_10", load_factor=1.1),
    Shock("dev_pv_30", pv_factor=0.7),
    Shock("dev_joint_10_30", load_factor=1.1, pv_factor=0.7),
)
HELDOUT = (
    Shock("test_load_20", load_factor=1.2),
    Shock("test_pv_50", pv_factor=0.5),
    Shock("test_peak_30", peak_factor=1.3),
    Shock("test_joint_15_40", load_factor=1.15, pv_factor=0.6),
    Shock("test_joint_15_40_3days", load_factor=1.15, pv_factor=0.6, duration=3),
    Shock("test_low_inventory_3days", load_factor=1.1, pv_factor=0.7,
          duration=3, initial_fraction=0.1),
)


def apply_shock(path, shock):
    out = np.asarray(path, float).copy()
    if out.shape != (144, 2) or not np.isfinite(out).all() or (out < 0).any():
        raise ValueError("Stress input must be a nonnegative 144-slot joint path")
    out[:, 0] *= shock.load_factor
    out[:, 1] *= shock.pv_factor
    # 17:00–21:00，使用时段起点索引，与控制器时段一致。
    out[17*6:21*6, 0] *= shock.peak_factor
    return out


def design_paths(forecast):
    return np.stack([apply_shock(forecast["path"], shock) for shock in DEVELOPMENT])


def fixed_plan_stress(G, reference, E0, realized_path, price, cfg, shocks=DEVELOPMENT):
    """调用前 G 已锁定；本函数没有任何优化器或预测器调用。"""
    from .solver import batch_metrics
    paths = np.stack([apply_shock(realized_path, shock) for shock in shocks])
    metrics = batch_metrics(G, reference, E0, paths, price, cfg)
    return {shock.name: {key: float(value[i]) for key, value in metrics.items()}
            for i, shock in enumerate(shocks)}
