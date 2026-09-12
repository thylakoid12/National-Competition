"""复用的储能优化内核：为问题二提供仅基于日前预测的辅助起点。

保留原模型构建、求解和校验函数，不含问题一入口及文件读取。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .timegrid import (
    DEFAULTS,
    DT_HOURS,
    MODEL_REVISION,
    PERIODS_PER_DAY,
    state_labels,
)

FLOW_NAMES = (
    "grid_to_load",
    "grid_to_battery",
    "pv_to_load",
    "pv_to_battery",
    "pv_curtailment",
    "battery_discharge",
)
SCHEDULE_COLUMNS = (
    "time",
    "load",
    "pv",
    "price",
    "grid_to_load",
    "grid_to_battery",
    "grid_purchase",
    "pv_to_load",
    "pv_to_battery",
    "pv_curtailment",
    "battery_charge",
    "battery_discharge",
    "energy_start",
    "energy_end",
    "charge_mode",
)


@dataclass(frozen=True)
class VariableIndex:
    """MILP 决策变量在一维向量中的位置。"""

    flow: dict[str, np.ndarray]
    energy: np.ndarray
    charge_mode: np.ndarray
    size: int


def _make_variable_index(periods: int) -> VariableIndex:
    flow = {
        name: np.arange(offset * periods, (offset + 1) * periods)
        for offset, name in enumerate(FLOW_NAMES)
    }
    energy = np.arange(6 * periods, 7 * periods + 1)
    charge_mode = np.arange(7 * periods + 1, 8 * periods + 1)
    return VariableIndex(flow, energy, charge_mode, 8 * periods + 1)


def solve_model(
    data: pd.DataFrame,
    e_max=DEFAULTS.capacity_kwh,
    p_max_kw=DEFAULTS.power_kw,
    eta=DEFAULTS.efficiency,
    e_min=DEFAULTS.min_energy_kwh,
    e_initial=DEFAULTS.initial_energy_kwh,
    solver=milp,
):
    """分层求解：最低购电费优先，其次最小弃光，最后恢复最低费用。"""

    periods = len(data)
    power_limit = p_max_kw * DT_HOURS
    index = _make_variable_index(periods)
    gl = index.flow["grid_to_load"]
    gb = index.flow["grid_to_battery"]
    pl = index.flow["pv_to_load"]
    pb = index.flow["pv_to_battery"]
    curtailed = index.flow["pv_curtailment"]
    discharged = index.flow["battery_discharge"]
    energy = index.energy
    mode = index.charge_mode

    lower = np.zeros(index.size)
    upper = np.full(index.size, np.inf)
    lower[energy], upper[energy] = e_min, e_max
    lower[energy[[0, -1]]] = upper[energy[[0, -1]]] = e_initial
    upper[mode] = 1
    integrality = np.zeros(index.size, dtype=int)
    integrality[mode] = 1

    matrix = lil_matrix((5 * periods, index.size))
    constraint_lower = np.zeros(5 * periods)
    constraint_upper = np.zeros(5 * periods)
    for period in range(periods):
        matrix[period, [pl[period], pb[period], curtailed[period]]] = 1
        constraint_lower[period] = constraint_upper[period] = data["pv"].iloc[period]

        matrix[periods + period, [pl[period], gl[period], discharged[period]]] = 1
        constraint_lower[periods + period] = constraint_upper[periods + period] = data[
            "load"
        ].iloc[period]

        matrix[
            2 * periods + period,
            [
                energy[period + 1],
                energy[period],
                gb[period],
                pb[period],
                discharged[period],
            ],
        ] = [
            1,
            -1,
            -eta,
            -eta,
            1 / eta,
        ]

        matrix[3 * periods + period, [gb[period], pb[period], mode[period]]] = [
            1,
            1,
            -power_limit,
        ]
        constraint_lower[3 * periods + period] = -np.inf

        matrix[4 * periods + period, [discharged[period], mode[period]]] = [
            1,
            power_limit,
        ]
        constraint_lower[4 * periods + period] = -np.inf
        constraint_upper[4 * periods + period] = power_limit

    matrix = matrix.tocsc()
    constraints = [LinearConstraint(matrix, constraint_lower, constraint_upper)]
    bounds = Bounds(lower, upper)
    cost = np.zeros(index.size)
    cost[gl] = cost[gb] = data["price"].to_numpy()

    def optimize(objective, stage):
        result = solver(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"mip_rel_gap": 0.0},
        )
        if not result.success:
            raise RuntimeError(f"第{stage}阶段未达到最优：{result.message}")
        return result

    first = optimize(cost, "一")

    epsilon = max(1e-6, abs(first.fun) * 1e-7)
    constraints.append(LinearConstraint(cost, -np.inf, first.fun + epsilon))
    curtailment_objective = np.zeros(index.size)
    curtailment_objective[curtailed] = 1
    second = optimize(curtailment_objective, "二")

    # 第二阶段只关心弃光，可能返回成本容差上界上的解。固定其最小弃光量后
    # 再次最小化成本，保证最终调度费用回到第一阶段的最低值。
    curtailment_epsilon = max(1e-8, abs(second.fun) * 1e-7)
    constraints.append(
        LinearConstraint(
            curtailment_objective,
            -np.inf,
            second.fun + curtailment_epsilon,
        )
    )
    third = optimize(cost, "三")

    solution = third.x
    schedule = data.reset_index(drop=True).copy()
    for name in FLOW_NAMES:
        schedule[name] = solution[index.flow[name]]
    schedule["grid_purchase"] = solution[gl] + solution[gb]
    schedule["battery_charge"] = solution[gb] + solution[pb]
    schedule["energy_start"] = solution[energy[:-1]]
    schedule["energy_end"] = solution[energy[1:]]
    schedule["charge_mode"] = np.rint(solution[mode]).astype(int)

    activity = matrix @ solution
    balance_error = np.max(
        np.abs(activity[: 3 * periods] - constraint_upper[: 3 * periods])
    )
    constraint_error = max(
        0.0,
        np.max(constraint_lower - activity),
        np.max(activity - constraint_upper),
        np.max(lower - solution),
        np.max(solution - upper),
        np.max(np.abs(solution[mode] - np.rint(solution[mode]))),
        cost @ solution - first.fun - epsilon,
        curtailment_objective @ solution - second.fun - curtailment_epsilon,
    )
    simultaneous = int(
        ((schedule.battery_charge > 1e-6) & (schedule.battery_discharge > 1e-6)).sum()
    )
    if constraint_error > 1e-5 or simultaneous:
        raise RuntimeError(f"解的约束校验未通过：最大违约 {constraint_error:.4e}")

    total_pv = data["pv"].sum()
    stats = {
        "model_revision": MODEL_REVISION,
        "optimal_cost": float(first.fun),
        "final_cost": float(cost @ solution),
        "epsilon": float(epsilon),
        "curtailment_epsilon": float(curtailment_epsilon),
        "grid_purchase": float(schedule.grid_purchase.sum()),
        "curtailment": float(schedule.pv_curtailment.sum()),
        "pv_utilization": (
            float(1 - schedule.pv_curtailment.sum() / total_pv) if total_pv else np.nan
        ),
        "energy_min": float(solution[energy].min()),
        "energy_max": float(solution[energy].max()),
        "energy_final": float(solution[energy[-1]]),
        "simultaneous_periods": simultaneous,
        "max_balance_residual": float(balance_error),
        "max_constraint_residual": float(constraint_error),
        "stage1_gap": float(first.mip_gap),
        "stage2_gap": float(second.mip_gap),
        "stage3_gap": float(third.mip_gap),
    }
    return schedule[list(SCHEDULE_COLUMNS)], stats
