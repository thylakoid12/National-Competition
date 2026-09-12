"""论文模型的显式定义；控制器与搜索预算不随模型改变。"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelSpec:
    name: str
    scenario_mode: str = "conditional"
    robust: bool = False
    risk_limit: bool = False
    stress_guard: bool = False
    fixed_alpha: float | None = None

    def __post_init__(self):
        if self.scenario_mode not in ("point", "uniform", "conditional"):
            raise ValueError("Unknown scenario mode")
        if self.fixed_alpha is not None and not 0 <= self.fixed_alpha <= 1:
            raise ValueError("Invalid fixed reserve factor")

    def radius(self, cfg):
        return cfg.radius if self.robust else 0.0

    def to_dict(self):
        return asdict(self)


MODELS = {m.name: m for m in (
    ModelSpec("M0", "point"),
    ModelSpec("M1", "uniform"),
    ModelSpec("M2"),
    ModelSpec("M3", robust=True),
    ModelSpec("M3-R", robust=True, risk_limit=True),
    ModelSpec("M3-S", robust=True, stress_guard=True),
    ModelSpec("M3-RS", robust=True, risk_limit=True, stress_guard=True),
    ModelSpec("M3-RS-uniform", "uniform", True, True, True),
    ModelSpec("M3-RS-rho0", risk_limit=True, stress_guard=True),
    ModelSpec("M3-RS-alpha1", robust=True, risk_limit=True, stress_guard=True, fixed_alpha=1.0),
)}
DEFAULT_MODELS = ("M0", "M1", "M2", "M3", "M3-R", "M3-S", "M3-RS")


def get_model(name):
    if isinstance(name, ModelSpec):
        return name
    try:
        return MODELS[name]
    except KeyError as error:
        raise ValueError(f"Unknown model {name}; choose from {tuple(MODELS)}") from error
