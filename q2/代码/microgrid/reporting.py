"""从日结果生成汇总和明细，不重新求解优化模型。"""

from pathlib import Path
import numpy as np
import pandas as pd
from .storage import evaluation_choice
from .storage import read_json, read_arrays, write_json, write_arrays
from .metrics import totals

METRICS = (
    ("计划购电量 / kWh", "G"),
    ("紧急购电量 / kWh", "H"),
    ("计划购电费 / 元", "plan_cost"),
    ("紧急购电费 / 元", "emergency_cost"),
    ("实际总购电费 / 元", "cash_cost"),
    ("发生应急的天数 / 日", "emergency_days"),
    ("应急时段数 / 个", "emergency_slots"),
    ("未利用计划额度 / kWh", "unused"),
    ("弃光量 / kWh", "curtailment"),
    ("储能转换损耗 / kWh", "storage_loss"),
    ("光伏消纳率", "pv_utilization"),
    ("12月31日24:00储电量 / kWh", "Eend"),
)


def daily_results(source, choice=None):
    source = Path(source)
    choice = choice or evaluation_choice(source)["selected"]
    directory = source / "evaluation" / choice
    paths = sorted(directory.glob("2025-??-??.json"))
    rows = [read_json(p) for p in paths]
    expected = pd.date_range("2025-02-01", "2025-12-31").strftime("%Y-%m-%d").tolist()
    if [r["date"] for r in rows] != expected:
        raise ValueError(
            "Full evaluation requires all 334 days, February through December."
        )
    return choice, directory, rows


def report(source, output):
    output = Path(output)
    choice, directory, rows = daily_results(source)
    summary = dict(selected=choice, **totals(rows))
    write_json(output / "evaluation_summary.json", summary)
    scalar_keys = [k for k, v in rows[0].items() if isinstance(v, (str, int, float))]
    pd.DataFrame(rows)[scalar_keys].to_csv(
        output / "daily_summary.csv", index=False, encoding="utf-8-sig"
    )
    arrays = [read_arrays(directory / (r["date"] + ".npz")) for r in rows]
    keys = ("G", "H", "charge", "discharge", "U", "W", "load", "pv", "E", "reference")
    write_arrays(
        output / "annual_selected.npz",
        dates=np.array([r["date"] for r in rows]),
        **{key: np.stack([a[key] for a in arrays]) for key in keys},
    )
    lines = [
        "# 表2-5 主模型评价期总结",
        "",
        f"方案：{choice}；评价期：2025年2月1日至12月31日，共334天。",
        "",
        "| 指标 | 数值 |",
        "| :--- | ---: |",
    ]
    for label, key in METRICS:
        v = summary[key]
        value = (
            f"{v:.4%}"
            if key == "pv_utilization"
            else (
                f"{v:.0f}"
                if key in ("emergency_days", "emergency_slots")
                else f"{v:.4f}"
            )
        )
        lines.append(f"| {label} | {value} |")
    lines += [
        "",
        "说明：方案标识与选型记录来自本次结果目录。模型属于事后重新设计，不能声称为未见全年数据时的预注册实验。",
    ]
    (output / "评价期汇总.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

