"""统一读写：按日落盘支持断点续跑。"""

import hashlib
import json
import os
from pathlib import Path
from time import sleep
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_arrays(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _replace(temp, path):
    # Windows预览程序可能短暂占用目标文件，重试仅放在写入边界。
    for attempt in range(5):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            sleep(0.1 * (attempt + 1))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _replace(temp, path)


def write_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    _replace(temp, path)


def evaluation_choice(source):
    """显式对照实验与一月选型结果使用不同记录，避免混淆选型结论。"""
    source = Path(source)
    comparison = source / "evaluation_choice.json"
    return read_json(
        comparison if comparison.exists() else source / "selection_frozen.json"
    )
