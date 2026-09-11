"""读取附件 1，生成 10 分钟时段数据和基础曲线。"""

from datetime import datetime, time

import numpy as np
import pandas as pd

from q1_config import (
    DT_HOURS, MINUTES_PER_PERIOD, OUTPUT_DIR, PERIODS_PER_DAY,
    SOURCE_WORKBOOK, minute_label,
)
from q1_plotting import plt, set_style


def parse_time_minutes(value):
    """附件的时间为区间结束时刻，0:00+1 表示次日零点。"""
    if isinstance(value, (datetime, time)):
        return value.hour * 60 + value.minute
    if isinstance(value, (int, float, np.number)):
        return round(value * 1440)
    clock, _, day = str(value).strip().partition("+")
    hour, minute = map(int, clock.split(":")[:2])
    if not (0 <= hour <= 24 and 0 <= minute < 60) or (hour == 24 and minute):
        raise ValueError(f"无效时间：{value}")
    return hour * 60 + minute + 1440 * int(day or 0)


def load_data(path=SOURCE_WORKBOOK):
    columns = {"时间": "time_raw", "电价": "price", "小区负载": "load_kw", "光伏发电预测功率": "pv_kw"}
    data = pd.read_excel(path).loc[:, list(columns)].rename(columns=columns)
    ends = np.arange(1, PERIODS_PER_DAY + 1) * MINUTES_PER_PERIOD
    if not np.array_equal(data.time_raw.map(parse_time_minutes), ends):
        raise ValueError("附件应按顺序包含 00:10 至次日 00:00 的 144 个采样点。")
    numeric = ["price", "load_kw", "pv_kw"]
    data[numeric] = data[numeric].apply(pd.to_numeric)
    if not np.isfinite(data[numeric]).all().all() or (data[["load_kw", "pv_kw"]] < 0).any().any():
        raise ValueError("数值必须有限，负荷和光伏功率不能为负。")
    data["time_raw"] = data.time_raw.map(
        lambda value: value.strftime("%H:%M") if isinstance(value, (datetime, time)) else str(value).strip()
    )
    data.insert(0, "t", np.arange(1, PERIODS_PER_DAY + 1))
    data["time_start"] = [minute_label(end - MINUTES_PER_PERIOD) for end in ends]
    data["time_end"] = [minute_label(end) for end in ends]
    data["time_interval"] = data.time_start + "-" + data.time_end
    return add_derived_features(data)


# 增加净负荷和仅用于统计的电量字段
def add_derived_features(data):
    data["net_load_kw"] = data["load_kw"] - data["pv_kw"]
    data["load_kwh"] = data["load_kw"] * DT_HOURS
    data["pv_kwh"] = data["pv_kw"] * DT_HOURS
    data["net_load_kwh"] = data["net_load_kw"] * DT_HOURS
    columns = [
        "t", "time_raw", "time_start", "time_end", "time_interval",
        "price", "load_kw", "pv_kw", "net_load_kw",
        "load_kwh", "pv_kwh", "net_load_kwh",
    ]
    return data[columns]


def peak_text(data, column, kind="max"):
    series = data[column]
    index = series.idxmax() if kind == "max" else series.idxmin()
    return f"{data.loc[index, 'time_interval']}（{data.loc[index, column]:.4f}）"


# 输出问题一建模前需要的基础统计
def calculate_statistics(data):
    load_energy = data["load_kwh"].sum()
    pv_energy = data["pv_kwh"].sum()
    surplus = data.loc[data["pv_kw"] > data["load_kw"], "time_interval"].tolist()
    ratio = load_energy / pv_energy if pv_energy != 0 else np.nan

    print("\n========== 基础数据摘要 ==========")
    print(f"数据条数：{len(data)}")
    print(f"时间范围：{data['time_interval'].iloc[0]} 至 {data['time_interval'].iloc[-1]}")
    print(f"电价（元/kWh）：最小 {data['price'].min():.4f}，最大 {data['price'].max():.4f}，平均 {data['price'].mean():.4f}")
    print(f"最低电价时段：{peak_text(data, 'price', 'min')}")
    print(f"最高电价时段：{peak_text(data, 'price', 'max')}")
    print(f"负载（kW）：最小 {data['load_kw'].min():.4f}，最大 {data['load_kw'].max():.4f}，平均 {data['load_kw'].mean():.4f}")
    print(f"负载峰值时段：{peak_text(data, 'load_kw')}；全天负荷电量：{load_energy:.4f} kWh")
    print(f"光伏（kW）：最大 {data['pv_kw'].max():.4f}，平均 {data['pv_kw'].mean():.4f}")
    print(f"光伏峰值时段：{peak_text(data, 'pv_kw')}；全天预测发电量：{pv_energy:.4f} kWh")
    print(f"净负荷（kW）：最小 {data['net_load_kw'].min():.4f}，最大 {data['net_load_kw'].max():.4f}，平均 {data['net_load_kw'].mean():.4f}")
    print(f"净负荷最大时段：{peak_text(data, 'net_load_kw')}")
    print(f"净负荷小于 0 的时段数：{int((data['net_load_kw'] < 0).sum())}")
    print(f"光伏大于负载的时段：{', '.join(surplus) if surplus else '无'}")
    print(f"全天负荷总电量 / 全天光伏预测发电量：{ratio:.4f}" if pd.notna(ratio) else "该比值无法计算")


def draw_curve(data, columns, labels, title, ylabel, filename):
    x = data["t"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for column, label in zip(columns, labels):
        ax.plot(x, data[column], label=label, linewidth=1.5)
    ticks = list(range(1, len(data) + 1, 12))
    ax.set_xticks(ticks, data.loc[[tick - 1 for tick in ticks], "time_start"])
    ax.set_xlabel("时间")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(columns) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# 分别保存论文可用的基础曲线图
def plot_data(data):
    set_style()
    draw_curve(data, ["price"], ["电价"], "分时电价曲线", "电价（元/kWh）", "price_curve.png")
    draw_curve(data, ["load_kw", "pv_kw"], ["小区负载", "光伏预测功率"], "负载与光伏预测功率", "功率（kW）", "load_pv_curve.png")
    draw_curve(data, ["net_load_kw"], ["净负荷"], "净负荷曲线", "净负荷（kW）", "net_load_curve.png")
    draw_curve(data, ["load_kw", "pv_kw", "net_load_kw"], ["小区负载", "光伏预测功率", "净负荷"], "基础功率对比", "功率（kW）", "basic_energy_curve.png")


# 保存 CSV 和 Excel，保留功率字段并附加电量统计字段
def save_processed_data(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "preprocessed_q1.csv"
    xlsx_path = OUTPUT_DIR / "preprocessed_q1.xlsx"
    data.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="preprocessed_q1")
        sheet = writer.book["preprocessed_q1"]
        sheet.freeze_panes = "A2"
        widths = [8, 12, 12, 12, 18, 14, 14, 14, 16, 14, 14, 18]
        for column, width in zip(sheet.columns, widths):
            sheet.column_dimensions[column[0].column_letter].width = width
        for row in sheet.iter_rows(min_row=2, min_col=6, max_col=12):
            for cell in row:
                cell.number_format = "0.000000"
    return csv_path, xlsx_path


def main():
    data = load_data()
    calculate_statistics(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_data(data)
    csv_path, xlsx_path = save_processed_data(data)
    print("\n========== 输出文件 ==========")
    print(csv_path)
    print(xlsx_path)
    print(f"图像已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
