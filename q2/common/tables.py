"""复用指定时段汇总；不包含result1模板导出。"""
from common.config import PERIODS_PER_HOUR,interval_labels

def build_tables(schedule):
    """把调度结果汇总为题目要求的两张表。"""
    labels = interval_labels()
    if schedule.time.tolist() != labels:
        raise ValueError("调度应按时间顺序包含全天的连续时段。")
    s = schedule.set_index("time")
    selected = ["10:00-10:10", "12:00-12:10", "14:00-14:10",
                "16:00-16:10", "18:00-18:10", "20:00-20:10"]
    table1 = [["时间段", "购电量", "时间段", "购电量", "时间段", "购电量"]]
    for group in (selected[:3], selected[3:]):
        table1.append([item for label in group for item in (label, float(s.loc[label, "grid_purchase"]))])
    daily_grid = float(s.grid_purchase.sum())
    daily_cost = float((s.price * s.grid_purchase).sum())
    table1.append(["全天购电量", None, daily_grid, "全天购电费", None, daily_cost])
    blocks = {}
    for hour in range(0, 24, 4):
        part = s.iloc[hour * PERIODS_PER_HOUR:(hour + 4) * PERIODS_PER_HOUR]
        blocks[f"{hour}:00-{hour + 4}:00"] = [
            float(part.battery_charge.sum()), float(part.battery_discharge.sum())
        ]
    table2 = [["时间段", "充电量", "放电量", "时间段", "充电量", "放电量"]]
    keys = list(blocks)
    for i in range(0, 6, 2):
        table2.append([keys[i], *blocks[keys[i]], keys[i + 1], *blocks[keys[i + 1]]])
    first, last = float(s.loc[labels[0], "energy_start"]), float(s.loc[labels[-1], "energy_end"])
    table2.append(["0:00 储电量", None, first, "24:00 储电量", None, last])
    return s, table1, table2, blocks, (first, last)

