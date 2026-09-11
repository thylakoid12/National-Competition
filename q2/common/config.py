"""复用的时间与设备常量，数值及标签规则保持不变。"""
from dataclasses import dataclass
MINUTES_PER_PERIOD = 10
PERIODS_PER_HOUR = 60 // MINUTES_PER_PERIOD
PERIODS_PER_DAY = 24 * PERIODS_PER_HOUR
DT_HOURS = MINUTES_PER_PERIOD / 60


@dataclass(frozen=True)
class ModelDefaults:
    """储能模型的基准参数，单位分别为 kWh、kW 和无量纲效率。"""

    capacity_kwh: float = 10_800
    power_kw: float = 5_000
    efficiency: float = 0.9
    min_energy_kwh: float = 1_200
    initial_energy_kwh: float = 6_000


DEFAULTS = ModelDefaults()
MODEL_REVISION = "cost-curtailment-cost-v1"


def minute_label(minutes: int) -> str:
    """把当天分钟数转换为时刻标签，并正确显示次日零点。"""

    if minutes == 24 * 60:
        return "24:00"
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def interval_labels() -> list[str]:
    """返回全天 144 个连续 10 分钟区间标签。"""

    return [
        f"{minute_label(start)}-{minute_label(start + MINUTES_PER_PERIOD)}"
        for start in range(0, 24 * 60, MINUTES_PER_PERIOD)
    ]


def state_labels() -> list[str]:
    """返回包含首末时刻的 145 个储能状态标签。"""

    return [
        minute_label(index * MINUTES_PER_PERIOD)
        for index in range(PERIODS_PER_DAY + 1)
    ]
