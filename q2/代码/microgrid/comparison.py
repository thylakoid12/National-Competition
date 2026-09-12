"""固定 A/B 定义的完整评价。A 是用户指定对照，不重新选择模型。"""

from pathlib import Path
import shutil
import numpy as np
from .config import CODE_DIR
from .data import load_data
from .storage import read_json, read_arrays, write_json, file_hash
from .experiment import run_variant, totals

MODELS = {"A": "search_only", "B": "joint_reserve"}


def run_fixed_model(source, output, cfg, model="A"):
    variant = MODELS[model]
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("对照评价必须使用独立结果目录")
    if cfg.to_dict() != read_json(CODE_DIR / "inputs/forecast_archive_config.json"):
        raise ValueError("A/B对照必须使用当前模型B的原配置")
    source_choice = read_json(source / "selection_frozen.json")
    if source_choice["selected"] != MODELS["B"]:
        raise ValueError("对照来源必须是模型B")
    bank = read_arrays(source / "evaluation_bank.npz")
    data = load_data()
    np.testing.assert_array_equal(bank["dates"], np.array(data.dates))
    np.testing.assert_array_equal(bank["actual"], data.values)
    np.testing.assert_array_equal(bank["price"], data.price)
    output.mkdir(parents=True, exist_ok=True)
    choice = dict(
        selected=variant,
        model=model,
        selection_rule="User-requested fixed-model comparison; not a January selection winner",
        january_selection_winner=MODELS["B"],
        evaluation_dates=["2025-02-01", "2025-12-31"],
        january_bank_sha256=file_hash(source / "january_bank.npz"),
        evaluation_bank_sha256=file_hash(source / "evaluation_bank.npz"),
        initial_states_sha256=file_hash(source / "initial_states.json"),
        config=cfg.to_dict(),
        reference_factor=1.0 if model == "A" else "optimized",
    )
    path = output / "evaluation_choice.json"
    if path.exists() and read_json(path) != choice:
        raise ValueError("续跑配置或输入改变，请使用新目录")
    write_json(path, choice)
    write_json(
        output / "protocol_frozen.json",
        dict(config=cfg.to_dict(), experiment="Fixed A/B comparison"),
    )
    for name in ("january_bank.npz", "evaluation_bank.npz", "initial_states.json"):
        target = output / name
        if target.exists():
            if file_hash(target) != file_hash(source / name):
                raise ValueError(f"续跑输入改变: {name}")
        else:
            shutil.copy2(source / name, target)
    rows = run_variant(output, variant, "evaluation", cfg)
    if model == "A" and any(row["alpha"] != 1.0 for row in rows):
        raise ValueError("模型A参考系数必须固定为1")
    summary = dict(selected=variant, model=model, **totals(rows))
    write_json(output / "evaluation_summary.json", summary)
    return summary
