"""模型参数和项目路径。"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT = CODE_DIR.parent
RESULT_DIR = PROJECT / "结果"
ATTACHMENTS = PROJECT.parent / "附件"
CURRENT_RESULT = RESULT_DIR / "m3_january_only_20260911"
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

    @property
    def B(self):
        return self.power * DT_HOURS

    def terminal(self, date, price):
        if str(date)[:10] == "2025-12-31":
            return 0.0
        return self.terminal_multiplier * float(min(price)) / self.eta_c

    def to_dict(self):
        return asdict(self)


def load_config(path=None):
    path = Path(path) if path else CODE_DIR / "config.json"
    return Config(**json.loads(path.read_text(encoding="utf-8")))
