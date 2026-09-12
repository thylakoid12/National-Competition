"""模型参数和项目路径。"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT = CODE_DIR.parent
RESULT_DIR = PROJECT / "结果"
def resolve_attachments(path=None):
    """显式路径优先；兼容附件在 Code/附件 或 Code 的上一级。"""
    if path is not None:
        candidate = Path(path).expanduser().resolve()
        if not all((candidate / name).is_file() for name in ("附件1.xlsx", "附件2.xlsx")):
            raise FileNotFoundError(f"附件目录缺少附件1.xlsx或附件2.xlsx：{candidate}")
        return candidate
    candidates = (PROJECT.parent / "附件", PROJECT.parent, PROJECT.parent.parent)
    return next((p for p in candidates if all((p / n).is_file() for n in
                ("附件1.xlsx", "附件2.xlsx"))), candidates[0])


ATTACHMENTS = resolve_attachments()
CURRENT_RESULT = RESULT_DIR / "m3_january_only_20260911"
STATISTICAL_RESULT = RESULT_DIR / "statistical_risk_v1"
DT_HOURS = 1 / 6
SLOTS = 144


@dataclass(frozen=True)
class Config:
    e_min: float = 1200.0
    e_max: float = 10800.0
    initial_energy: float = 6000.0
    power: float = 5000.0
    eta_c: float = 0.9
    eta_d: float = 0.9
    emergency_multiplier: float = 5.0
    residual_window: int = 60
    bandwidth: float = 1.0
    shrinkage: float = 0.25
    radius: float = 0.05
    ml_min_days: int = 14
    ml_window: int = 120
    ml_depth: int = 4
    ml_iterations: int = 300
    ml_learning_rate: float = 0.05
    seed: int = 2026
    seed_time_limit_seconds: float = 5.0
    seed_mip_gap: float = 0.0001
    seed_node_limit: int = 16
    objective_tolerance: float = 1e-7
    physical_tolerance: float = 1e-6
    terminal_multiplier: float = 1.0
    forecast_bias_window: int = 7
    forecast_bias_strength: float = 0.25
    forecast_bias_cap: float = 0.25
    residual_scale: bool = False
    scale_recent_window: int = 14
    scale_min: float = 0.67
    scale_max: float = 1.5
    risk_beta: float = 0.9
    risk_budget_ratio: float = 0.5
    stress_budget_ratio: float = 0.75
    cost_premium: float = 0.03
    search_evaluations: int = 3000
    search_starts: int = 3
    search_step_scales: tuple = (0.1, 0.02, 0.005)
    bootstrap_block_days: int = 3
    bootstrap_repetitions: int = 300

    def __post_init__(self):
        import math
        if not (0 <= self.e_min < self.e_max and self.e_min <= self.initial_energy <= self.e_max):
            raise ValueError("Invalid battery energy bounds")
        if not (0 < self.eta_c <= 1 and 0 < self.eta_d <= 1 and self.power > 0):
            raise ValueError("Invalid battery efficiency or power")
        if not (0 <= self.radius <= 1 and 0 <= self.shrinkage <= 1 and self.bandwidth > 0):
            raise ValueError("Invalid scenario probability parameters")
        if not (0 <= self.risk_beta < 1 and self.residual_window >= 1):
            raise ValueError("Invalid risk level or residual window")
        if not (0 <= self.forecast_bias_strength <= 1 and self.forecast_bias_cap >= 0
                and self.forecast_bias_window >= 1 and self.scale_recent_window >= 2):
            raise ValueError("Invalid statistical calibration parameters")
        if not (0 < self.scale_min <= 1 <= self.scale_max):
            raise ValueError("Invalid residual scale limits")
        if min(self.risk_budget_ratio, self.stress_budget_ratio, self.cost_premium) < 0:
            raise ValueError("Risk budgets and cost premium must be nonnegative")
        if not (1 <= self.search_starts <= 3 and self.search_evaluations >= 30 * self.search_starts):
            raise ValueError("Search requires 1–3 starts and at least 30 evaluations per start")
        if len(self.search_step_scales) != 3 or any(x <= 0 for x in self.search_step_scales):
            raise ValueError("Provide three positive search step scales")
        if self.bootstrap_block_days < 1 or self.bootstrap_repetitions < 0:
            raise ValueError("Invalid bootstrap settings")
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Configuration values must be finite")

    @property
    def B(self):
        return self.power * DT_HOURS

    def terminal(self, date, price):
        if str(date)[:10] == "2025-12-31":
            return 0.0
        return self.terminal_multiplier * float(min(price)) / self.eta_c

    def to_dict(self):
        result = asdict(self)
        result["search_step_scales"] = list(self.search_step_scales)
        return result


def load_config(path=None):
    path = Path(path) if path else CODE_DIR / "config.json"
    return Config(**json.loads(path.read_text(encoding="utf-8")))
