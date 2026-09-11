from time import perf_counter

import numpy as np
import pandas as pd

from q1_config import DEFAULTS, MODEL_REVISION, PREPROCESSED_FILE, RESULT_DIR
from q1_model import load_data, solve_model
from q1_plotting import COLORS, plt, save_figure, set_style
from q1_result_io import read_result, write_result

from matplotlib.ticker import StrMethodFormatter


STEP = 250
SCAN_END = 35000
BASELINE = DEFAULTS.capacity_kwh
COST_TOL = 0.01


# 分段扫描并复用同一模型版本的成功容量点，只对新增或过期点重新求解
def scan_capacity(data, step=STEP, end=SCAN_END):
    capacities = sorted(set(range(6000, 20001, step)) | set(range(20500, end + 1, 500)) | {BASELINE, end})
    path = RESULT_DIR / "q1_capacity_dense_scan.xlsx"
    cached = read_result(path) if path.exists() else pd.DataFrame()
    if "model_revision" not in cached or not cached.model_revision.eq(MODEL_REVISION).all():
        cached = pd.DataFrame()
    elif not cached.empty:
        cached = cached.set_index("E_max")
    rows = []
    for capacity in capacities:
        if capacity in cached.index and cached.loc[capacity, "status"] == "optimal":
            rows.append({"E_max": capacity, **cached.loc[capacity].to_dict()})
            continue
        started = perf_counter()
        row = {
            "E_max": capacity,
            "model_revision": MODEL_REVISION,
            "status": "optimal",
            "message": "",
        }
        try:
            _, stats = solve_model(data, e_max=capacity)
            row.update({
                "optimal_cost": stats["optimal_cost"], "final_cost": stats["final_cost"],
                "total_grid_purchase": stats["grid_purchase"],
                "total_curtailment": stats["curtailment"],
                "pv_utilization": stats["pv_utilization"],
                "min_energy": stats["energy_min"], "max_energy": stats["energy_max"],
                "final_energy": stats["energy_final"], "epsilon": stats["epsilon"],
                "max_constraint_residual": stats["max_constraint_residual"],
                "stage1_gap": stats["stage1_gap"], "stage2_gap": stats["stage2_gap"],
            })
            print(f"E_max={capacity:5d} kWh，最优成本 {stats['optimal_cost']:.4f} 元", flush=True)
        except RuntimeError as error:
            row["status"] = "infeasible" if "infeasible" in str(error).lower() else "solver_error"
            row["message"] = str(error)
            print(f"E_max={capacity}: {row['status']}，继续扫描。", flush=True)
        row["solve_seconds"] = perf_counter() - started
        rows.append(row)
    return pd.DataFrame(rows)


# 边际价值归属当前容量到下一容量的区间，不跨越失败点计算差分
def analyze_scan(scan):
    scan["max_energy_reached"] = scan.max_energy
    scan["next_E_max"] = scan.E_max.shift(-1)
    scan["absolute_saving"] = scan.optimal_cost - scan.optimal_cost.shift(-1)
    scan["marginal_saving"] = scan.absolute_saving / (scan.next_E_max - scan.E_max)
    scan["interval_midpoint"] = (scan.E_max + scan.next_E_max) / 2
    scan["is_knee"] = False
    scan["normalized_distance"] = np.nan
    valid = scan.dropna(subset=["optimal_cost"])
    result = {"candidate": None, "knee": None, "slow_start": None, "flat_start": None}
    increases = scan.loc[scan.absolute_saving < -COST_TOL, ["E_max", "next_E_max", "absolute_saving"]]
    if len(increases):
        print("警告：成本出现超过 0.01 元的上升：\n" + increases.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(valid) < 3 or valid.optimal_cost.max() - valid.optimal_cost.min() <= COST_TOL:
        return scan, result
    x = valid.E_max.to_numpy()
    cost = valid.optimal_cost.to_numpy()
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (cost - cost.min()) / (cost.max() - cost.min())
    signed = ((yn[-1] - yn[0]) * xn - (yn - yn[0])) / np.hypot(1, yn[-1] - yn[0])
    distance = np.abs(signed)
    position = np.argmax(distance[1:-1]) + 1
    candidate = x[position]
    scan.loc[valid.index, "normalized_distance"] = distance
    result["candidate"] = candidate
    before = (cost[0] - cost[position]) / (candidate - x[0])
    after = (cost[position] - cost[-1]) / (x[-1] - candidate)
    mv = scan.marginal_saving.dropna().to_numpy()
    diminishing = np.mean(np.diff(mv) <= 1e-4) >= 0.9
    clear = (len(valid) == len(scan) and increases.empty and signed[position] > 0.03
             and diminishing and after < 0.7 * before)
    if clear:
        result.update(knee=candidate, cost=cost[position], before=before, after=after,
                      local_before=(cost[position - 1] - cost[position]) / (x[position] - x[position - 1]),
                      local_after=(cost[position] - cost[position + 1]) / (x[position + 1] - x[position]))
        scan.loc[scan.E_max == candidate, "is_knee"] = True
    initial_mv = scan.marginal_saving.iloc[:4].mean()
    result["slow_threshold"] = 0.1 * initial_mv
    for i in range(len(scan) - 4):
        tail = scan.marginal_saving.iloc[i:-1]
        if tail.isna().any():
            continue
        if result["slow_start"] is None and (tail <= result["slow_threshold"]).all():
            result["slow_start"] = scan.E_max.iloc[i]
        if result["flat_start"] is None and (tail.abs() <= 1e-4).all():
            result["flat_start"] = scan.E_max.iloc[i]
    return scan, result


# 仅绘制容量成本与边际价值，边际值位于相邻容量区间的中点
def plot_scan(scan, analysis):
    set_style()
    fig, left = plt.subplots(figsize=(9.2, 5.5), layout="constrained")
    right = left.twinx()
    left.plot(scan.E_max, scan.optimal_cost, color=COLORS["grid"], label="最优日成本")
    right.plot(scan.interval_midpoint, scan.marginal_saving, "--", color=COLORS["discharge"],
               label="边际节省（区间中点）")
    left.set(xlabel=r"储能上限 $E_{max}$ / kWh", ylabel="最优日成本 / 元",
             xlim=(scan.E_max.min(), scan.E_max.max()), title="储能容量成本与边际价值")
    right.set_ylabel("边际节省 / [元/(kWh·day)]")
    left.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    left.set_xticks([*np.arange(6000, scan.E_max.max() - 2000, 4000), scan.E_max.max()])
    if analysis["flat_start"] is not None:
        left.axvspan(analysis["flat_start"], scan.E_max.max(), color="#eeeeee", zorder=0)
        left.text((analysis["flat_start"] + scan.E_max.max()) / 2, 0.95, "数值平台区",
                  transform=left.get_xaxis_transform(), ha="center", fontsize=9)
    for value, text, height in [(BASELINE, "基准 10800 kWh", 0.97),
                                 (analysis["knee"], None, 0.83)]:
        if value is None:
            continue
        left.axvline(value, color="#777777", linestyle=":", linewidth=1)
        label = text or f"$E_{{knee}}$ = {value:g} kWh"
        left.text(value + 170, height, label, transform=left.get_xaxis_transform(), va="top", fontsize=9)
        point = scan.loc[scan.E_max == value, "optimal_cost"].iloc[0]
        left.scatter(value, point, color=COLORS["grid"], s=24, zorder=4)
    handles, labels = left.get_legend_handles_labels()
    rh, rl = right.get_legend_handles_labels()
    left.legend(handles + rh, labels + rl, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncols=2)
    left.grid(axis="y", color="#dddddd", linewidth=0.5)
    save_figure(fig, "q1_capacity_marginal_value")


# 区分曲线候选拐点、明显趋缓区间与数值平台，结论只讨论运行电费
def report_scan(scan, a):
    costs = scan.set_index("E_max").optimal_cost
    end = int(scan.E_max.max())
    lines = ["Q1 储能容量密集扫描", "复用 q1_model.solve_model；其他参数固定；每个点均完整执行分层优化。",
             "先求最低费用，再最小化弃光，最后在最小弃光方案中恢复最低费用；final_cost 与最终调度一致。",
             f"6000–20000 kWh 步长250，20000–{end} kWh 步长500，另含10800基准点；复用已有成功点，边际差分按真实间隔计算。",
             f"成功点数：{scan.status.eq('optimal').sum()} / {len(scan)}。"]
    for capacity in sorted({6000, BASELINE, 20000, 30000, end} & set(costs.index)):
        lines.append(f"{capacity} kWh 的最优成本：{costs.loc[capacity]:.4f} 元/天。")
    lines += [f"全扫描区间最大成本下降：{costs.max() - costs.min():.4f} 元/天。",
              f"明显成本上升区间数（超过 {COST_TOL} 元）：{(scan.absolute_saving < -COST_TOL).sum()}。",
              f"最大约束残差：{scan.max_constraint_residual.max():.4e}。",
              f"首尾连线最大距离的内部候选点：{a['candidate']} kWh。",
              "明确候选判据：完整扫描、成本非增、归一化有向距离>0.03、至少90%的相邻边际值非增，且拐点后平均边际值低于此前70%。"]
    if a["knee"] is not None:
        lines += [f"运行成本候选拐点：{a['knee']:g} kWh；对应成本 {a['cost']:.4f} 元/天。",
                  f"knee 前/后平均边际节省：{a['before']:.4f} / {a['after']:.4f} 元/(kWh·day)。",
                  f"knee 紧邻前/后区间边际节省：{a['local_before']:.4f} / {a['local_after']:.4f} 元/(kWh·day)。"]
    else:
        lines.append("未观察到满足判据的明显拐点；不将最大距离候选点标为明确拐点。")
    slow = f"{a['slow_start']:g} kWh" if a["slow_start"] is not None else "扫描范围内未达到"
    flat = f"{a['flat_start']:g} kWh" if a["flat_start"] is not None else "扫描范围内未观察到"
    lines += [f"持续趋缓判据：剩余至少4个区间的边际值始终≤初始4个区间平均值的10%（{a.get('slow_threshold', np.nan):.4f}）。",
              f"持续低边际收益起点：{slow}；数值平台起点（边际绝对值≤0.0001）：{flat}。",
              f"扫描末尾区间的边际节省：{scan.marginal_saving.iloc[-2]:.4f} 元/(kWh·day)。",
              "候选拐点是给定扫描范围内的几何折中位置，不代表成本在此突然转折或已经进入平台。"]
    saving = costs.loc[BASELINE] - costs.loc[end]
    near = a["slow_start"] is not None and BASELINE >= a["slow_start"]
    lines += [f"10800 kWh 是否进入上述平台邻近区间：{'是' if near else '否'}。",
              f"10800 扩至 {end} kWh 仍可节省：{saving:.4f} 元/天（{saving / costs.loc[BASELINE]:.4%}）。",
              f"20000 扩至 {end} kWh 可再节省：{costs.loc[20000] - costs.loc[end]:.4f} 元/天。", "",
              "简短结论：",
              f"10800 kWh 能满足原题运行约束，但{'已进入' if near else '尚未进入'}本次定义的低边际收益区间。",
              f"在其余参数固定时，增加至{end} kWh 可再降低日电费 {saving:.4f} 元，因此{'仍有运行电费改善空间' if saving > COST_TOL else '扩容的运行电费改善已很有限'}。"]
    if a["knee"] is not None:
        lines.append(f"扫描曲线的运行成本候选拐点为 {a['knee']:g} kWh，拐点后的全区间平均边际收益降至此前的 {a['after'] / a['before']:.4%}。")
    if a["slow_start"] is not None:
        lines.append(f"约 {a['slow_start']:g} kWh 起，边际收益持续低于初始水平的10%，可作为推荐关注容量区间的起点。")
    if a["flat_start"] is not None:
        flat_index = scan.index[scan.E_max == a["flat_start"]][0]
        previous = scan.E_max.iloc[max(0, flat_index - 1)]
        lines.append(f"从 {a['flat_start']:g} kWh 扫描点起成本维持 {costs.loc[a['flat_start']]:.4f} 元/天；平台起点的网格定位区间为 {previous:g}–{a['flat_start']:g} kWh。")
    lines.append("以上仅是本日运行电费分析；缺少扩容投资成本，不能把运行成本拐点称为投资最优容量。")
    report = "\n".join(lines)
    (RESULT_DIR / "q1_capacity_scan_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


def main():
    data = load_data(PREPROCESSED_FILE)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scan = scan_capacity(data)
    path = RESULT_DIR / "q1_capacity_dense_scan.xlsx"
    write_result(scan, path)
    if scan.status.eq("optimal").sum() < 3:
        print("成功点不足，已保存求解状态，暂不进行拐点分析。")
        return
    scan, analysis = analyze_scan(scan)
    write_result(scan, path)
    plot_scan(scan, analysis)
    report_scan(scan, analysis)


if __name__ == "__main__":
    main()
