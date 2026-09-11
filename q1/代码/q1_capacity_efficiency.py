from time import perf_counter

import numpy as np
import pandas as pd

from q1_config import DEFAULTS, PREPROCESSED_FILE, RESULT_DIR
from q1_model import load_data, solve_model
from q1_plotting import plt, save_figure, set_style
from q1_result_io import write_result

from matplotlib.ticker import StrMethodFormatter


CAPACITIES = [8000, 10000, 12000, 14000, 16000, 18000, 20000, 24000, 28000, 32000]
EFFICIENCIES = [0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95]


# 网格70点加基准容量附近3点，逐点调用统一分层模型
def solve_surface(data):
    combinations = [(e, eta, "grid") for eta in EFFICIENCIES for e in CAPACITIES]
    combinations += [
        (DEFAULTS.capacity_kwh, eta, "baseline_local")
        for eta in (0.875, DEFAULTS.efficiency, 0.925)
    ]
    rows = []
    for e_max, eta, kind in combinations:
        start = perf_counter()
        row = {"E_max": e_max, "eta": eta, "point_type": kind, "status": "optimal", "message": ""}
        try:
            _, stats = solve_model(data, e_max=e_max, eta=eta)
            row.update({"model_revision": stats["model_revision"],
                        "optimal_cost": stats["optimal_cost"], "final_cost": stats["final_cost"],
                        "curtailment": stats["curtailment"], "max_energy_reached": stats["energy_max"],
                        "final_energy": stats["energy_final"], "epsilon": stats["epsilon"],
                        "max_constraint_residual": stats["max_constraint_residual"],
                        "stage1_gap": stats["stage1_gap"], "stage2_gap": stats["stage2_gap"]})
            print(f"E_max={e_max}, eta={eta:.4f}：{stats['optimal_cost']:.4f} 元", flush=True)
        except RuntimeError as error:
            row.update(status="solver_error", message=str(error))
            print(f"E_max={e_max}, eta={eta:.4f}：求解失败，记录后继续。", flush=True)
        row["solve_seconds"] = perf_counter() - start
        rows.append(row)
    result = pd.DataFrame(rows)
    write_result(result, RESULT_DIR / "q1_capacity_efficiency_surface.xlsx")
    return result


# 等高线仅由规则网格绘制，基准点为独立求解结果，网格之间仅作绘图插值
def plot_surface(result):
    set_style()
    grid = result[result.point_type == "grid"].pivot(index="eta", columns="E_max", values="optimal_cost")
    z = np.ma.masked_invalid(grid.to_numpy())
    fig, ax = plt.subplots(figsize=(10.5, 6.3), layout="constrained")
    filled = ax.contourf(grid.columns, grid.index, z, levels=np.linspace(z.min(), z.max(), 81), cmap="viridis_r")
    levels = np.linspace(z.min(), z.max(), 8)[1:-1]
    contour = ax.contour(grid.columns, grid.index, z, levels=levels, colors="white", linewidths=0.85)
    ax.clabel(contour, fmt="%.4f", fontsize=8, inline_spacing=5)
    bar = fig.colorbar(filled, ax=ax, pad=0.025)
    bar.set_label("最优日电费/元")
    bar.set_ticks(np.linspace(z.min(), z.max(), 5))
    ax.scatter(DEFAULTS.capacity_kwh, DEFAULTS.efficiency, marker="*", s=190,
               color="#fbd34d", edgecolor="#222222", linewidth=0.9, zorder=5)
    ax.annotate("基准方案", (DEFAULTS.capacity_kwh, DEFAULTS.efficiency),
                xytext=(22, 14), textcoords="offset points",
                fontsize=10, color="black", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                arrowprops={"arrowstyle": "-", "color": "#222222"})
    ax.set(xlabel=r"储能上限 $E_{max}$ / kWh", ylabel=r"充放电效率 $\eta$",
           title="储能容量与充放电效率对最优日电费的影响", xlim=(8000, 32000), ylim=(0.80, 0.95))
    ax.set_xticks([8000, 14000, 20000, 26000, 32000])
    ax.set_yticks(EFFICIENCIES)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.4f}"))
    save_figure(fig, "q1_capacity_efficiency_contour")


# 用归一化局部变化和分区容量差分解释交互作用，避免直接比较不同单位的斜率
def analyze_surface(result):
    good = result[result.status == "optimal"]
    if len(good) != len(result):
        print(f"成功 {len(good)}/{len(result)}，存在缺失点，暂不输出完整交互结论。")
        return
    cost = good.set_index(["E_max", "eta"]).optimal_cost
    grid = good[good.point_type == "grid"].pivot(index="eta", columns="E_max", values="optimal_cost")
    mv = -np.diff(grid.to_numpy(), axis=1) / np.diff(grid.columns)
    capacity_increases = int((np.diff(grid.to_numpy(), axis=1) > 0.01).sum())
    efficiency_increases = int((np.diff(grid.to_numpy(), axis=0) > 0.01).sum())
    base = cost.loc[DEFAULTS.capacity_kwh, DEFAULTS.efficiency]
    se = abs(
        (cost.loc[12000, DEFAULTS.efficiency] - cost.loc[10000, DEFAULTS.efficiency])
        / base
        / (2000 / DEFAULTS.capacity_kwh)
    )
    seta = abs(
        (cost.loc[DEFAULTS.capacity_kwh, 0.925] - cost.loc[DEFAULTS.capacity_kwh, 0.875])
        / base
        / (0.05 / DEFAULTS.efficiency)
    )
    lowest = good.loc[good.optimal_cost.idxmin()]
    highest = good.loc[good.optimal_cost.idxmax()]
    lines = ["Q1 容量×效率双参数敏感性分析", f"成功 {len(good)}/{len(result)}：70个网格点，另有10800 kWh下3个局部点。",
             "每点均复用Q1分层MILP；最终调度在保持最小弃光的同时恢复第一阶段最低费用。",
             "等高线由离散扫描点插值得到，不能视作网格之间额外求解的结果。",
             f"最低日电费：{lowest.optimal_cost:.4f} 元，E_max={lowest.E_max:.4f} kWh，eta={lowest.eta:.4f}。",
             f"最高日电费：{highest.optimal_cost:.4f} 元，E_max={highest.E_max:.4f} kWh，eta={highest.eta:.4f}。",
             f"基准点成本：{base:.4f} 元。",
             f"容量增大/效率增大时出现明显成本上升的相邻网格对数：{capacity_increases}/{efficiency_increases}。",
             f"最大约束残差：{good.max_constraint_residual.max():.4e}。", "",
             f"基准附近归一化敏感性：容量 {se:.4f}，效率 {seta:.4f}。",
             "容量采用10000–12000 kWh割线并以10800归一化；效率采用基准容量下0.875–0.925中心差分。",
             f"该邻域{'效率' if seta > se else '容量'}对相对参数变化更敏感，不能将不同单位的原始斜率直接比较。",
             f"eta=0.90，容量10000→12000：{cost.loc[10000, 0.90]:.4f}→{cost.loc[12000, 0.90]:.4f} 元。",
             f"E_max=10800，eta=0.875→0.925：{cost.loc[10800, 0.875]:.4f}→{cost.loc[10800, 0.925]:.4f} 元。", "",
             "各效率下容量边际价值，单位元/(kWh·day)："]
    for eta in EFFICIENCIES:
        first = (cost.loc[8000, eta] - cost.loc[10000, eta]) / 2000
        last = (cost.loc[28000, eta] - cost.loc[32000, eta]) / 4000
        saving = cost.loc[8000, eta] - cost.loc[32000, eta]
        lines.append(f"eta={eta:.4f}：8000–10000为{first:.4f}，28000–32000为{last:.4f}，8000扩到32000共节省{saving:.4f}元。")
    difference = np.diff(mv, axis=0)
    lines += [f"提高效率后，容量边际价值增加/减少的相邻比较数：{(difference > 1e-5).sum()}/{(difference < -1e-5).sum()}。",
              "因此效率与扩容收益的关系需按容量区间判断，不预设替代或互补关系。", "",
              "高容量、高效率邻域："]
    for e_max in (24000, 28000, 32000):
        saving = cost.loc[e_max, 0.925] - cost.loc[e_max, 0.95]
        lines.append(f"E_max={e_max}，eta从0.925提高至0.95仍节省{saving:.4f}元/天。")
    high_mv = (cost.loc[28000, 0.95] - cost.loc[32000, 0.95]) / 4000
    low_mv = (cost.loc[8000, 0.95] - cost.loc[10000, 0.95]) / 2000
    lines += [f"eta=0.95的末段容量边际收益为初段的{high_mv / low_mv:.4%}。",
              "容量方向的收益递减与效率方向的节费空间是两件事；不能仅因容量曲线变平就断言二维区域整体进入平台。",
              "分析限于本日运行电费，不包含设备投资成本。"]
    report = "\n".join(lines)
    (RESULT_DIR / "q1_capacity_efficiency_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(PREPROCESSED_FILE)
    result = solve_surface(data)
    plot_surface(result)
    analyze_surface(result)


if __name__ == "__main__":
    main()
