"""Q2 configuration, reusing the shared time and battery constants."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
from common.config import DT_HOURS, DEFAULTS, interval_labels, state_labels

HERE = Path(__file__).resolve().parent
ATTACHMENTS = HERE.parent / '附件'

@dataclass(frozen=True)
class Config:
    nominal_capacity: float = 12000.
    e_min: float = DEFAULTS.min_energy_kwh
    e_max: float = DEFAULTS.capacity_kwh
    initial_energy: float = DEFAULTS.initial_energy_kwh
    power: float = DEFAULTS.power_kw
    eta_c: float = DEFAULTS.efficiency
    eta_d: float = DEFAULTS.efficiency
    emergency_multiplier: float = 5.
    parallel_workers: int = 1
    reference_backend: str = 'scipy'
    accelerated_paths: bool = False
    residual_window: int = 60
    bandwidth: float = 1.
    shrinkage: float = .25
    radius: float = .05
    ml_min_days: int = 14
    ml_window: int = 120
    ml_depth: int = 4
    ml_iterations: int = 300
    ml_learning_rate: float = .05
    seed: int = 2026
    budget: int = 1156
    deterministic_start: bool = True
    bounded_seed: bool = False
    seed_time_limit_seconds: float = 1.
    seed_mip_gap: float = 1e-4
    seed_node_limit: int = 16
    steps: tuple = (.1, .02, .005)
    objective_tolerance: float = 1e-7
    physical_tolerance: float = 1e-6
    controller_version: str = 'reference-feedback-v1.0'
    optimizer_version: str = 'coordinate-best-improvement-v1.1'
    forecast_version: str = 'issued-history-weekday-onehot-v1.1'
    formal_strategy: str = 'star_M3'
    strategy_basis: str = 'Predeclared engineering choice, not annual hindsight.'
    template_policy: str = 'correct_copy_headers'
    stress_start: str = '2025-06-21'
    stress_days: int = 3
    stress_gamma: float = 1.
    terminal_multiplier: float = 1.
    stress_load_slots: tuple = (108,126)
    stress_pv_slots: tuple = (60,96)
    parameter_basis: str = 'engineering_defaults; January MAE selects F0 and F-star identities'

    @property
    def B(self):
        return self.power * DT_HOURS

    def terminal(self, date, price):
        return 0. if str(date)[:10] == '2025-12-31' else self.terminal_multiplier * float(min(price)) / self.eta_c

    @property
    def version(self):
        return digest(asdict(self))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def load_config(path=None):
    return Config() if path is None else Config(**json.loads(Path(path).read_text(encoding='utf-8')))
