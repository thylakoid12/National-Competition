import numpy as np

from q1_config import DEFAULTS, DT_HOURS, MINUTES_PER_PERIOD, RESULT_DIR, state_labels
from q1_plotting import COLORS, plt, save_figure, set_style
from q1_result_io import read_result


def save_dispatch_figure(fig, ax, name):
    ax.set_xlabel(f"时间（每个调度时段为 {MINUTES_PER_PERIOD} min）")
    ax.set_xticks(np.arange(0, 25, 2), [f"{hour:02d}:00" for hour in range(0, 25, 2)])
    ax.set_xlim(0, 24)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    save_figure(fig, name)


# 分别输出供需、充放电和储能状态三张图。
def plot_dispatch(s):
    set_style()
    energy = np.r_[s.energy_start, s.energy_end.iloc[-1]]
    time = np.arange(len(s) + 1) * DT_HOURS
    fig, a = plt.subplots(figsize=(11.2, 4.5), layout="constrained")
    for field, label, color in [("load", "负荷", COLORS["load"]),
                                 ("pv", "光伏", COLORS["pv"]),
                                 ("grid_purchase", "购电量", COLORS["grid"])]:
        a.stairs(s[field], time, label=label, color=color, baseline=None)
    a.set(title="(A) 系统供需与最优购电", ylabel="时段电量 / kWh", ylim=(0, None))
    a.legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncols=3)

    save_dispatch_figure(fig, a, "q1_supply_demand")

    fig, b = plt.subplots(figsize=(11.2, 4.5), layout="constrained")
    b.bar(time[:-1], s.battery_charge, width=DT_HOURS, align="edge",
          color=COLORS["charge"], label="充电", linewidth=0)
    b.bar(time[:-1], -s.battery_discharge, width=DT_HOURS, align="edge",
          color=COLORS["discharge"], label="放电（负向）", linewidth=0)
    b.axhline(0, color="#555555", linewidth=0.7)
    b.set(title="(B) 电价与电池充放电", ylabel="充放电量 / kWh")
    price_axis = b.twinx()
    price_axis.stairs(s.price, time, color=COLORS["price"], label="电价", baseline=None)
    price_axis.set_ylabel("电价 / (元/kWh)", color=COLORS["price"])
    handles, labels = b.get_legend_handles_labels()
    ph, pl = price_axis.get_legend_handles_labels()
    b.legend(handles + ph, labels + pl, loc="lower right", bbox_to_anchor=(1, 1.01), ncols=3)

    save_dispatch_figure(fig, b, "q1_battery_dispatch")

    fig, c = plt.subplots(figsize=(11.2, 4.5), layout="constrained")
    c.plot(time, energy, color=COLORS["grid"], label="储能量 E")
    boundaries = [
        (DEFAULTS.min_energy_kwh, f"下限 {DEFAULTS.min_energy_kwh:g}", "--"),
        (DEFAULTS.capacity_kwh, f"上限 {DEFAULTS.capacity_kwh:g}", "--"),
        (DEFAULTS.initial_energy_kwh, f"首末状态 {DEFAULTS.initial_energy_kwh:g}", ":"),
    ]
    for value, label, style in boundaries:
        c.axhline(value, color="#888888", linestyle=style, linewidth=0.9)
        c.text(0.3, value + 180, f"{label} kWh", color="#666666", fontsize=9)
    c.scatter([0, 24], energy[[0, -1]], color=COLORS["grid"], s=28, zorder=4, clip_on=False)
    for value, offset in [(DEFAULTS.capacity_kwh, -44), (DEFAULTS.min_energy_kwh, 34)]:
        hits = np.flatnonzero(np.isclose(energy, value, atol=1e-5, rtol=0))
        if len(hits):
            i = hits[0]
            label = f"首次触及{'上' if value == DEFAULTS.capacity_kwh else '下'}限 {state_labels()[i]}"
            c.annotate(label, (time[i], energy[i]), xytext=(12, offset), textcoords="offset points",
                       fontsize=9, arrowprops={"arrowstyle": "-", "color": "#666666"})
    c.set(title=f"(C) 储能状态（{len(energy)} 个状态点）", ylabel="储能量 / kWh",
          xlabel=f"时间（每个调度时段为 {MINUTES_PER_PERIOD} min）",
          ylim=(0, DEFAULTS.capacity_kwh * 1.12))
    save_dispatch_figure(fig, c, "q1_energy_state")


def main():
    plot_dispatch(read_result(RESULT_DIR / "q1_schedule.xlsx"))


if __name__ == "__main__":
    main()
