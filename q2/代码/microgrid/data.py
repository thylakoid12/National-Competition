"""读取附件并将功率转换为十分钟电量。"""

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ATTACHMENTS, DT_HOURS, SLOTS, resolve_attachments


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


@dataclass(frozen=True)
class Data:
    dates: tuple
    values: np.ndarray
    price: np.ndarray


def load_data(attachments=ATTACHMENTS):
    attachments = resolve_attachments(attachments)
    tariff = pd.read_excel(attachments / "附件1.xlsx")
    # 附件末尾存在无时间的合计行；仍严格检查真实的 144 个时段。
    tariff = tariff.loc[tariff["时间"].notna()]
    if len(tariff) != SLOTS:
        raise ValueError("附件1必须恰好包含144个有时间的时段")
    expected_ends = np.arange(10, 1441, 10)
    if not np.array_equal(tariff["时间"].map(parse_time_minutes), expected_ends):
        raise ValueError("附件1的时段顺序不正确")
    price = tariff["电价"].to_numpy(float)
    values = []
    expected_dates = pd.date_range("2025-01-01", "2025-12-31")
    for name in ("小区负载", "光伏发电实际功率"):
        sheet = pd.read_excel(attachments / "附件2.xlsx", sheet_name=name)
        if sheet.shape != (365, 145) or not pd.DatetimeIndex(sheet.iloc[:, 0]).equals(
            expected_dates
        ):
            raise ValueError(f"{name}的日期或维度不正确")
        if not np.array_equal(
            [parse_time_minutes(v) for v in sheet.columns[1:]], expected_ends
        ):
            raise ValueError(f"{name}的时段顺序不正确")
        values.append(sheet.iloc[:, 1:].to_numpy(float) * DT_HOURS)
    actual = np.stack(values, axis=-1)
    if (
        not np.isfinite(actual).all()
        or (actual < 0).any()
        or not np.isfinite(price).all()
        or (price <= 0).any()
    ):
        raise ValueError("电量必须有限且非负，电价必须有限且大于零")
    return Data(tuple(d.date().isoformat() for d in expected_dates), actual, price)
