"""每个日期分别保存购电、储能和应急图，不拼接子图。"""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .storage import evaluation_choice
from .storage import read_json, read_arrays, write_json
from .config import Config, load_config

DEFAULT_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


def make_figures(day, date, cfg):
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )
    figures = {}
    for name, title, ylabel in (
        ("purchase", "负荷、光伏与计划购电", "十分钟电量 / kWh"),
        ("storage", "储能状态与参考轨迹", "储电量 / kWh"),
        ("emergency", "紧急购电", "十分钟电量 / kWh"),
    ):
        fig, ax = plt.subplots(figsize=(10, 3.7), layout="constrained")
        ax.set(title=f"{date}  {title}", xlabel="时间 / h", ylabel=ylabel, xlim=(0, 24))
        ax.set_xticks(np.arange(0, 25, 3))
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        figures[name] = fig
    ends = np.arange(1, 145) / 6
    ax = figures["purchase"].axes[0]
    for key, label, color in (
        ("load", "实际负荷", "#333333"),
        ("pv", "实际光伏", "#E6A01B"),
        ("G", "计划购电", "#2475B0"),
    ):
        ax.step(
            np.r_[0, ends],
            np.r_[day[key][0], day[key]],
            where="pre",
            label=label,
            color=color,
            linewidth=1.35,
        )
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=3)
    ax = figures["storage"].axes[0]
    states = np.arange(145) / 6
    ax.plot(states, day["E"], label="实际储电量", color="#2475B0", linewidth=1.6)
    ax.plot(
        states,
        day["reference"],
        label="参考储电量",
        color="#A266AC",
        linestyle="--",
        linewidth=1.2,
    )
    ax.axhline(cfg.e_min, color="#999999", linestyle=":", linewidth=1)
    ax.axhline(cfg.e_max, color="#999999", linestyle=":", linewidth=1)
    ax.set_ylim(0, cfg.e_max * 1.08)
    ax.legend(frameon=False, ncol=2)
    ax = figures["emergency"].axes[0]
    ax.bar(ends - 1 / 12, day["H"], width=1 / 6, color="#CA5346")
    ax.set_ylim(0, max(1.0, float(day["H"].max()) * 1.15))
    ax.text(
        0.99,
        0.94,
        f'合计 {day["H"].sum():.4f} kWh',
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    return figures


def plot(source, output, dates=DEFAULT_DATES, cfg=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(source / "protocol_frozen.json")
    cfg = cfg or (
        Config(**protocol["config"]) if "config" in protocol else load_config()
    )
    choice = evaluation_choice(source)["selected"]
    files = []
    for date in dates:
        day = read_arrays(source / "evaluation" / choice / (date + ".npz"))
        figures = make_figures(day, date, cfg)
        for name, fig in figures.items():
            for extension in ("png", "svg"):
                path = output / f"{date}_{name}.{extension}"
                fig.savefig(path, dpi=200)
                files.append(path.name)
            plt.close(fig)
    write_json(
        output / "figure_manifest.json",
        dict(dates=list(dates), panels_per_image=1, files=files),
    )
    return files


def combined_figure(series, title, dates=None):
    """四条能量序列共图；应急量用明确标注的红色右轴突出。"""
    import matplotlib.dates as mdates

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )
    fig, left = plt.subplots(figsize=(13.5, 5.4), layout="constrained")
    right = left.twinx()
    annual = dates is not None
    x = np.array(dates) if annual else np.arange(144) / 6
    lines = []
    for key, label, color in (
        ("load", "实际负荷", "#555D66"),
        ("pv", "实际光伏", "#D89A20"),
        ("G", "计划购电", "#2878B5"),
    ):
        if annual:
            (line,) = left.plot(
                x, series[key], label=label, color=color, linewidth=1.25, alpha=0.85
            )
        else:
            line = left.stairs(
                series[key],
                np.arange(145) / 6,
                label=label,
                color=color,
                linewidth=1.55,
            )
        lines.append(line)
    bars = right.bar(
        x if annual else x + 1 / 12,
        series["H"],
        width=0.8 if annual else 1 / 6,
        color="#D62728",
        alpha=0.7,
        label="紧急购电（右轴）",
        zorder=4,
    )
    active = np.flatnonzero(series["H"] > 1e-6)
    if len(active):
        right.scatter(
            x[active] if annual else x[active] + 1 / 12,
            series["H"][active],
            color="#B70016",
            s=13 if annual else 24,
            zorder=5,
        )
    maximum = float(np.max(series["H"]))
    right.set_ylim(0, max(1.0, maximum * 1.45))
    left.set_ylim(bottom=0)
    left.set_ylabel("日电量 / kWh" if annual else "十分钟电量 / kWh")
    right.set_ylabel("紧急购电量 / kWh（右轴）", color="#B70016", fontweight="bold")
    right.tick_params(axis="y", colors="#B70016")
    right.spines["right"].set_color("#B70016")
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    left.grid(axis="y", alpha=0.18)
    left.set_title(title, pad=40, fontweight="bold")
    left.legend(
        lines + [bars],
        ["实际负荷", "实际光伏", "计划购电", "紧急购电（右轴）"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
    )
    total = float(np.sum(series["H"]))
    note = f"紧急购电合计：{total:.4f} kWh"
    note += (
        f"  |  应急 {len(active)} 天"
        if annual
        else (
            "  |  当日无紧急购电"
            if not len(active)
            else f"  |  应急 {len(active)} 个十分钟时段"
        )
    )
    right.text(
        0.015,
        0.965,
        note,
        transform=right.transAxes,
        color="#B70016",
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#F1B6B6", alpha=0.94, pad=6),
    )
    if annual:
        left.xaxis.set_major_locator(mdates.MonthLocator())
        left.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
        left.set_xlim(x[0], x[-1])
        left.set_xlabel("2025年日期（日汇总；评价期为2—12月）")
    else:
        left.set_xlim(0, 24)
        left.set_xticks(np.arange(0, 25, 2))
        left.set_xlabel("时间 / h")
        if len(active):
            peak = int(np.argmax(series["H"]))
            right.annotate(
                f"最大时段：{maximum:.4f} kWh",
                xy=(x[peak] + 1 / 12, maximum),
                xytext=(0, 22),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="#B70016",
                arrowprops=dict(arrowstyle="-", color="#B70016"),
            )
    return fig


def plot_submission(source, output, dates=DEFAULT_DATES, cfg=None):
    """全年总览、四个日期四参数组合图，以及独立储能图。"""
    from datetime import datetime
    from .reporting import daily_results

    source, output = Path(source), Path(output)
    variant, directory, rows = daily_results(source)
    cfg = cfg or load_config()
    model = {"search_only": "A", "joint_reserve": "B", "adaptive_joint": "C"}[variant]
    output.mkdir(parents=True, exist_ok=True)
    keys = ("load", "pv", "G", "H")
    daily = {key: [] for key in keys}
    selected = {}
    for row in rows:
        day = read_arrays(directory / (row["date"] + ".npz"))
        for key in keys:
            daily[key].append(float(day[key].sum()))
        if row["date"] in dates:
            selected[row["date"]] = day
    daily = {key: np.array(values) for key, values in daily.items()}
    files = []

    def save(fig, stem):
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            path = stem.with_suffix("." + ext)
            fig.savefig(path, dpi=240)
            files.append(str(path.relative_to(output)))
        plt.close(fig)

    axis_dates = [datetime.fromisoformat(row["date"]) for row in rows]
    save(
        combined_figure(daily, f"模型{model}：全年购电与供需总览（334天）", axis_dates),
        output / "全年总览",
    )
    for date in dates:
        day = selected[date]
        save(
            combined_figure(day, f"模型{model}：{date} 购电与供需"),
            output / "四个指定日" / date,
        )
        # 储能状态仍单独保留，避免把不同物理含义挤进购电图。
        figures = make_figures(day, date, cfg)
        save(figures.pop("storage"), output / "储能图" / date)
        for fig in figures.values():
            plt.close(fig)
    write_json(
        output / "绘图核验.json",
        dict(
            model=model,
            days=len(rows),
            date_start=rows[0]["date"],
            date_end=rows[-1]["date"],
            annual_aggregation="Daily sums of 144 ten-minute energy values",
            annual_totals={k: float(v.sum()) for k, v in daily.items()},
            selected_dates=list(dates),
            files=files,
        ),
    )
    (output / "图件说明.md").write_text(
        "# 模型" + model + "图件\n\n"
        "全年总览按日汇总，覆盖正式评价期2025年2月1日至12月31日，共334天；一月用于选型，不混入评价图。\n\n"
        "四个指定日以十分钟为单位，实际负荷、实际光伏和计划购电使用左轴；紧急购电使用红色柱形和红点，读取右轴。左右轴刻度不同，不能直接按图形高度比较电量。红色数值标注保留四位小数。\n\n"
        "储能状态及参考轨迹另存于“储能图”。所有图提供PNG和SVG。\n",
        encoding="utf-8",
    )
    return files
