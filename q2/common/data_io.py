"""复用附件1读取与右端点时间解析；不含原预处理输出入口。"""
from datetime import datetime,time
from pathlib import Path
import numpy as np
import pandas as pd
from common.config import DT_HOURS,MINUTES_PER_PERIOD,PERIODS_PER_DAY,minute_label
SOURCE_WORKBOOK=Path(__file__).resolve().parents[2]/"附件"/"附件1.xlsx"

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

