"""复用的结果文件中英文列名转换与统一 Excel 样式。"""

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font


COLUMNS = {
    "model_revision": "模型版本",
    "time": "时间段", "load": "负荷电量（kWh）", "pv": "光伏电量（kWh）", "price": "电价（元/kWh）",
    "grid_to_load": "电网供负荷（kWh）", "grid_to_battery": "电网给电池充电（kWh）",
    "grid_purchase": "电网总购电量（kWh）", "pv_to_load": "光伏供负荷（kWh）",
    "pv_to_battery": "光伏给电池充电（kWh）", "pv_curtailment": "时段弃光量（kWh）",
    "battery_charge": "电池充电量（kWh）", "battery_discharge": "电池放电量（kWh）",
    "energy_start": "期初储能量（kWh）", "energy_end": "期末储能量（kWh）",
    "charge_mode": "充放电模式（1充电或空闲，0放电或空闲）",
    "optimal_cost": "第一阶段最优购电费（元）", "final_cost": "最终调度购电费（元）",
    "epsilon": "成本容差（元）", "curtailment_epsilon": "弃光容差（kWh）",
    "curtailment": "全天弃光量（kWh）", "pv_utilization": "光伏消纳率",
    "energy_min": "最低储能量（kWh）", "energy_max": "最高储能量（kWh）", "energy_final": "末端储能量（kWh）",
    "simultaneous_periods": "同时充放电时段数", "max_balance_residual": "最大能量平衡残差（kWh）",
    "max_constraint_residual": "最大约束违约量（各约束自身单位）",
    "stage1_gap": "第一阶段相对最优间隙", "stage2_gap": "第二阶段相对最优间隙",
    "stage3_gap": "第三阶段相对最优间隙",
    "parameter": "变化参数", "value": "参数取值（单位见变化参数）", "cost_change_pct": "费用变化率（%）",
    "E_max": "储能容量上限（kWh）", "eta": "单程充放电效率",
    "status": "求解状态", "message": "状态说明", "solve_seconds": "求解耗时（秒）",
    "total_grid_purchase": "全天累计购电量（kWh）", "total_curtailment": "全天累计弃光量（kWh）",
    "min_energy": "扫描方案最低储能量（kWh）", "max_energy": "扫描方案最高储能量（kWh）",
    "max_energy_reached": "实际达到的最大储能量（kWh）", "final_energy": "扫描方案末端储能量（kWh）",
    "next_E_max": "下一容量点（kWh）", "absolute_saving": "至下一容量点的日节费（元）",
    "marginal_saving": "容量边际节费（元/(kWh·天)）", "interval_midpoint": "容量区间中点（kWh）",
    "is_knee": "是否几何候选拐点", "normalized_distance": "归一化垂直距离",
    "point_type": "扫描点类型", "label": "参数名称", "x0": "基准参数值", "x_minus": "下扰动参数值",
    "x_plus": "上扰动参数值", "cost_minus": "下扰动最优费用（元）", "cost_plus": "上扰动最优费用（元）",
    "cost0": "基准最优费用（元）", "cost_column": "费用计算口径", "sensitivity": "归一化敏感性系数",
}
VALUES = {
    "parameter": {"e_max": "储能容量上限（kWh）", "p_max_kw": "充放电功率上限（kW）", "eta": "单程充放电效率"},
    "status": {"optimal": "最优", "infeasible": "不可行", "solver_error": "求解异常"},
    "point_type": {"grid": "参数网格点", "baseline_local": "基准附近补充点"},
    "cost_column": {"optimal_cost": "第一阶段最优购电费"},
    "is_knee": {True: "是", False: "否"},
}


# 结果只保存Excel一份，数值保留完整精度，显示四位小数
def write_result(data: pd.DataFrame, path) -> None:
    """使用统一中文列名和显示格式写入结果工作簿。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = data.copy()
    for column, mapping in VALUES.items():
        if column in shown:
            shown[column] = shown[column].map(lambda value: mapping.get(value, value))
    shown = shown.rename(columns=COLUMNS)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        shown.to_excel(writer, index=False, sheet_name="结果数据")
        sheet = writer.sheets["结果数据"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 44
        for cells in sheet.columns:
            sheet.column_dimensions[cells[0].column_letter].width = 26
            cells[0].font = Font(name="Microsoft YaHei", bold=True)
            cells[0].alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            for cell in cells[1:]:
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "0.0000%" if cells[0].value == "光伏消纳率" else "0.0000"


def read_result(path, sheet_name=0) -> pd.DataFrame:
    """读取 CSV/Excel 结果，并还原为程序内部使用的英文列名。"""

    path = Path(path)
    data = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=sheet_name)
    data = data.rename(columns={label: field for field, label in COLUMNS.items()})
    for column, mapping in VALUES.items():
        if column in data:
            reverse = {label: value for value, label in mapping.items()}
            data[column] = data[column].map(lambda value: reverse.get(value, value))
    return data
