import numpy as np
import pandas as pd

from q1_config import DEFAULTS, RESULT_DIR
from q1_model import load_data, solve_model
from q1_plotting import plt, save_figure, set_style
from q1_result_io import write_result


# 每次只改变一个参数，容量场景仅调整储能上限，其他边界保持不变
def sensitivity_analysis(data, baseline):
    cases = {
        "e_max": [9_720, DEFAULTS.capacity_kwh, 11_880],
        "p_max_kw": [4_000, DEFAULTS.power_kw, 6_000],
        "eta": [0.85, DEFAULTS.efficiency, 0.95],
    }
    defaults = {
        "e_max": DEFAULTS.capacity_kwh,
        "p_max_kw": DEFAULTS.power_kw,
        "eta": DEFAULTS.efficiency,
    }
    rows = []
    for parameter, values in cases.items():
        for value in values:
            stats = baseline if value == defaults[parameter] else solve_model(data, **{parameter: value})[1]
            rows.append({"parameter": parameter, "value": value, **stats})
            print(f"{parameter}={value:g}: 成本 {stats['optimal_cost']:.4f} 元，弃光 {stats['curtailment']:.4f} kWh")
    result = pd.DataFrame(rows)
    result["cost_change_pct"] = (result.optimal_cost / baseline["optimal_cost"] - 1) * 100
    write_result(result, RESULT_DIR / "q1_sensitivity.xlsx")
    return result

# 用相邻上下扰动点的中心差分计算无量纲敏感性，不重新求解
def normalized_sensitivity(source, cost0):
    set_style()
    settings = [
        ("e_max", DEFAULTS.capacity_kwh, "储能容量 $E_{max}$"),
        ("p_max_kw", DEFAULTS.power_kw, "功率上限 $P_{max}$"),
        ("eta", DEFAULTS.efficiency, r"充放电效率 $\eta$"),
    ]
    rows = []
    for parameter, x0, label in settings:
        part = source[source.parameter == parameter].sort_values("value")
        minus = part[part.value < x0].iloc[-1]
        plus = part[part.value > x0].iloc[0]
        if not np.isclose(x0 - minus.value, plus.value - x0):
            raise ValueError(f"{parameter} 的扰动点不对称，不能按指定中心差分计算。")
        coefficient = abs(((plus.optimal_cost - minus.optimal_cost) / cost0)
                          / ((plus.value - minus.value) / x0))
        rows.append({"parameter": parameter, "label": label, "x0": x0,
                     "x_minus": minus.value, "x_plus": plus.value,
                     "cost_minus": minus.optimal_cost, "cost_plus": plus.optimal_cost,
                     "cost0": cost0, "cost_column": "optimal_cost", "sensitivity": coefficient})
    result = pd.DataFrame(rows).sort_values("sensitivity", ascending=False).reset_index(drop=True)
    write_result(result, RESULT_DIR / "q1_normalized_sensitivity.xlsx")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    bars = ax.bar(result.label, result.sensitivity, width=0.52, color=["#36729a", "#6d96b0", "#a8c0cf"])
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in result.sensitivity], padding=6)
    ax.set(title="归一化参数敏感性（中心差分）", ylabel="无量纲敏感性系数 $S_x$",
           ylim=(0, result.sensitivity.max() * 1.2))
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    save_figure(fig, "q1_normalized_sensitivity")
    print(result[["parameter", "sensitivity"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    data = load_data()
    _, baseline = solve_model(data)
    result = sensitivity_analysis(data, baseline)
    normalized_sensitivity(result, baseline["optimal_cost"])


if __name__ == "__main__":
    main()
