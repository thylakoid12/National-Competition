from datetime import time

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font

from q1_config import (
    DEFAULTS,
    DT_HOURS,
    PERIODS_PER_DAY,
    RESULT_DIR,
    SOURCE_WORKBOOK,
    interval_labels,
    state_labels,
)
from q1_model import solve_model
from q1_plotting import plt, save_figure, set_style
from q1_result_io import COLUMNS, read_result


# 严格区分当天00:00与次日00:00，采样标签表示区间右端点
def minute_of_day(value):
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    text = str(value).strip()
    clock, _, day = text.partition("+")
    hour, minute = map(int, clock.split(":")[:2])
    return hour * 60 + minute + 1440 * int(day or 0)


# 原附件只读核验，梯形电量仅在本检验中构造，不改预处理数据
def make_trapezoid_data():
    raw = pd.read_excel(SOURCE_WORKBOOK)
    raw["minute"] = raw["时间"].map(minute_of_day)
    if raw.minute.duplicated().any():
        raise ValueError("原附件存在重复功率采样时刻。")
    points = raw.set_index("minute")
    ends = np.arange(10, 1441, 10)
    if not set(ends).issubset(points.index):
        raise ValueError("附件缺少当天00:10至次日00:00之间的采样点。")
    baseline = read_result(RESULT_DIR / "q1_schedule.xlsx")
    summary = read_result(RESULT_DIR / "q1_summary.xlsx").iloc[0].to_dict()
    labels = interval_labels()
    if baseline.time.tolist() != labels:
        raise ValueError("基准调度的时间标签与附件右端采样时刻不一致。")
    right = points.loc[ends, ["小区负载", "光伏发电预测功率"]].to_numpy(float)
    np.testing.assert_allclose(baseline[["load", "pv"]], right * DT_HOURS, atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(baseline.price, points.loc[ends, "电价"], atol=1e-12)
    np.testing.assert_allclose((baseline.price * baseline.grid_purchase).sum(), summary["final_cost"], atol=1e-8)
    left = np.vstack([right[0], right[:-1]])
    has_midnight = 0 in points.index
    if has_midnight:
        left[0] = points.loc[0, ["小区负载", "光伏发电预测功率"]].to_numpy(float)
    energy = (left + right) * DT_HOURS / 2
    if not has_midnight:
        energy[0] = right[0] * DT_HOURS
    alternative = baseline[["time", "load", "pv", "price"]].copy()
    alternative[["load", "pv"]] = energy
    note = ("附件存在当天00:00功率，全部144个区间采用梯形近似。" if has_midnight else
            "附件无当天00:00功率：首区间00:00–00:10仍用右端矩形法，后续143个区间采用梯形近似；0:00+1为次日，不作为当天左端点。")
    audit = pd.DataFrame({"时间段": labels, "右端采样时刻": points.loc[ends, "时间"].astype(str).to_numpy(),
                          "电价（元/kWh）": alternative.price,
                          "负荷右端功率（kW）": right[:, 0], "光伏右端功率（kW）": right[:, 1],
                          "负荷左端功率（kW）": left[:, 0], "光伏左端功率（kW）": left[:, 1],
                          "右端矩形负荷电量（kWh）": baseline.load,
                          "梯形近似负荷电量（kWh）": energy[:, 0],
                          "右端矩形光伏电量（kWh）": baseline.pv,
                          "梯形近似光伏电量（kWh）": energy[:, 1],
                          "积分方式": "梯形近似"})
    if not has_midnight:
        audit.loc[0, ["负荷左端功率（kW）", "光伏左端功率（kW）"]] = np.nan
        audit.loc[0, "积分方式"] = "右端矩形（左端采样缺失）"
    return baseline, summary, alternative, audit, note


# 费用统一对比第一阶段最优值，另列最终调度费用；零基准相对差不作除法
def compare_results(base, base_stats, trap, trap_stats):
    def metrics(s, stats):
        e = np.r_[s.energy_start, s.energy_end.iloc[-1]]
        return {"最优购电费用/元（第一阶段）": stats["optimal_cost"],
                "最终调度购电费用/元": float((s.price * s.grid_purchase).sum()),
                "总购电量/kWh": s.grid_purchase.sum(), "总光伏发电量/kWh": s.pv.sum(),
                "总负荷电量/kWh": s.load.sum(), "总弃光量/kWh": s.pv_curtailment.sum(),
                "光伏消纳率/%": 100 * (1 - s.pv_curtailment.sum() / s.pv.sum()),
                "最大储能量/kWh（SOC对应能量）": e.max(), "最小储能量/kWh（SOC对应能量）": e.min()}
    before, after = metrics(base, base_stats), metrics(trap, trap_stats)
    rows = []
    for name, a in before.items():
        b = after[name]
        rows.append({"指标": name, "原右端矩形法": a, "梯形近似": b, "绝对差": abs(b-a),
                     "相对差(%)": abs(b-a)/abs(a)*100 if abs(a)>1e-8 else "不适用（基准为零）",
                     "带符号差（梯形减矩形）": b-a})
    comparison = pd.DataFrame(rows)
    rows = []
    for field, label in [("grid_purchase", "电网购电量"), ("battery_charge", "充电量"),
                         ("battery_discharge", "放电量"), ("energy_start", "储能量（144个期初点）")]:
        delta = np.abs(trap[field].to_numpy() - base[field].to_numpy())
        rows.append({"比较对象": label, "点数": 144, "平均绝对差/kWh": delta.mean(),
                     "最大绝对差/kWh": delta.max(), "最大差对应时段": base.time.iloc[delta.argmax()]})
    a, b = np.r_[base.energy_start, base.energy_end.iloc[-1]], np.r_[trap.energy_start, trap.energy_end.iloc[-1]]
    delta = np.abs(a-b)
    index = int(delta.argmax())
    rows.append({"比较对象": "储能量（完整145个状态点）", "点数": 145, "平均绝对差/kWh": delta.mean(),
                 "最大绝对差/kWh": delta.max(), "最大差对应时段": f"{index//6:02d}:{index%6*10:02d}"})
    states = pd.DataFrame({"状态时刻": state_labels(),
                           "矩形法储能量（kWh）": a, "梯形近似储能量（kWh）": b,
                           "矩形法SOC（%）": a/12000*100, "梯形近似SOC（%）": b/12000*100})
    return comparison, pd.DataFrame(rows), states


# 仅展示购电、正负充放电和储能状态的差异
def plot_comparison(base, trap):
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9), sharex=True, layout="constrained")
    time = np.arange(PERIODS_PER_DAY + 1) * DT_HOURS
    for s, label, color, style in [(base, "右端矩形法", "#36729a", "-"),
                                    (trap, "梯形近似（首区间保留矩形）", "#c47753", "--")]:
        axes[0].stairs(s.grid_purchase, time, baseline=None, label=label, color=color, linestyle=style)
        axes[1].stairs(s.battery_charge-s.battery_discharge, time, baseline=None, label=label, color=color, linestyle=style)
        axes[2].plot(time, np.r_[s.energy_start, s.energy_end.iloc[-1]], label=label, color=color, linestyle=style)
    axes[0].set(title="(A) 电网购电量对比", ylabel="时段购电量/kWh")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncols=2)
    axes[1].set(title="(B) 充放电量对比（正值充电、负值放电）", ylabel="时段充放电量/kWh")
    axes[1].axhline(0, color="gray", linewidth=0.6)
    axes[2].set(title="(C) 储能状态对比（145个状态点）", ylabel="储能量/kWh", xlabel="时间")
    for boundary in (DEFAULTS.min_energy_kwh, DEFAULTS.capacity_kwh):
        axes[2].axhline(boundary, color="gray", linestyle=":", linewidth=0.7)
    axes[2].set_xticks(np.arange(0,25,2),[f"{h:02d}:00" for h in range(0,25,2)])
    for ax in axes:
        ax.set_xlim(0,24)
        ax.grid(axis="y", alpha=0.2)
    save_figure(fig,"q1_discretization_robustness")


# 所有检验材料放在一份中文工作簿内，保留数值精度且显示四位小数
def main():
    base, base_stats, data, audit, note = make_trapezoid_data()
    print(note)
    trap, trap_stats = solve_model(data)
    comparison, differences, states = compare_results(base, base_stats, trap, trap_stats)
    notes = [note, "电价直接保留原时段取值，不做平均或移动。附件为典型日数据，不额外假造带日期的00:00采样。",
             "仅新求解一次梯形方案（完整分层优化）；基准方案直接读取已有最终结果。",
             "额定容量12000kWh，储能1200–10800kWh，首末6000kWh，功率5000kW，单程效率0.9。",
             "绝对差为差值绝对值；相对差以原右端矩形法为基准；基准为零时相对差不适用。",
             "光伏消纳率行的绝对差单位为百分点。储能能量以kWh计，真正SOC=储能量/12000，另在状态表给出百分数。",
             "最终调度在保持最小弃光的同时重新最小化费用，因此费用与第一阶段最低值一致。",
             "逐时调度差异也可能包含多个等价最优解的选择差异，未添加平滑或最接近基准的第三目标。"]
    checks = pd.DataFrame([{"方法":name, **{COLUMNS.get(k,k):v for k,v in stats.items()}}
                            for name,stats in [("右端矩形法",base_stats),("梯形近似",trap_stats)]])
    sheets = {"指标对比":comparison, "调度差异":differences, "处理说明":pd.DataFrame({"说明":notes}),
              "时间与积分核验":audit, "基准调度":base.rename(columns=COLUMNS),
              "梯形调度":trap.rename(columns=COLUMNS), "储能状态145点":states, "求解校验":checks}
    path = RESULT_DIR / "q1_discretization_comparison.xlsx"
    with pd.ExcelWriter(path,engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer,index=False,sheet_name=name)
            ws=writer.sheets[name]; ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
            ws.row_dimensions[1].height=42
            for cells in ws.columns:
                ws.column_dimensions[cells[0].column_letter].width=30 if name!="处理说明" else 110
                cells[0].font=Font(name="Microsoft YaHei",bold=True)
                for cell in cells:
                    cell.alignment=Alignment(vertical="center",wrap_text=True)
                    if isinstance(cell.value,(int,float)): cell.number_format="0.0000"
            if name=="处理说明":
                for row in range(2,ws.max_row+1): ws.row_dimensions[row].height=38
    plot_comparison(base,trap)
    print(comparison.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(differences.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print(f"梯形方案最大约束残差：{trap_stats['max_constraint_residual']:.4e}；同时充放电时段数：{trap_stats['simultaneous_periods']}")
    print(f"已保存：{path}")


if __name__ == "__main__":
    main()
