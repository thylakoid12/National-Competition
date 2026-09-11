"""重跑第一问，将三阶段记录和原始求解日志按运行时间归档。"""

from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import scipy

from q1_config import DEFAULTS, MODEL_REVISION, PREPROCESSED_FILE, RESULT_DIR
from q1_model import load_data


def worker(run_dir):
    from q1 import main

    records = []
    try:
        main(stage_records=records, solver_log=True)
    finally:
        (run_dir / "stages.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main():
    started = datetime.now().astimezone()
    run_dir = RESULT_DIR / "logs" / started.strftime("q1_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    source_dir = run_dir / "source"
    source_dir.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, source_dir / source.name)
    shutil.copy2(PREPROCESSED_FILE, run_dir / PREPROCESSED_FILE.name)
    data = load_data()
    data.to_csv(run_dir / "model_input.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "started_at": started.isoformat(),
        "model_revision": MODEL_REVISION,
        "parameters": asdict(DEFAULTS),
        "python": sys.version,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "input_sha256": sha256(PREPROCESSED_FILE.read_bytes()).hexdigest(),
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in source_dir.glob("*.py")},
        "timing_definition": "perf_counter 测量每次 scipy.optimize.milp 调用的墙钟时间，含接口开销，不含模型构建、绘图和文件导出。",
    }
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with (run_dir / "solver.log").open("wb") as log:
        result = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(run_dir)],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )
    metadata.update(finished_at=datetime.now().astimezone().isoformat(), exit_code=result.returncode)
    (run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"运行记录：{run_dir}", flush=True)
    result.check_returncode()
    records = json.loads((run_dir / "stages.json").read_text(encoding="utf-8"))
    assert len(records) == 3
    for record in records:
        np.testing.assert_allclose(data.price @ np.array(record["grid_purchase_by_period"]), record["cost"], atol=1e-8)
        np.testing.assert_allclose(sum(record["curtailment_by_period"]), record["curtailment"], atol=1e-8)
    for name in ("q1_schedule.xlsx", "q1_summary.xlsx", "q1_table1.xlsx", "q1_table2.xlsx", "result1.xlsx"):
        shutil.copy2(RESULT_DIR / name, run_dir / name)
    labels = ["第一阶段：费用优化", "第二阶段：弃光优化", "第三阶段：费用回调"]
    lines = ["# 第一问三阶段求解记录", "", f"运行时间：{started.isoformat()}", "",
             "| 求解阶段 | 全天购电费用/元 | 全天弃光量/kWh | 本阶段目标的相对最优间隙 | 求解时间/s |",
             "|---|---:|---:|---:|---:|"]
    for label, record in zip(labels, records):
        curtailed = record["curtailment"]
        shown_curtailment = f"{curtailed:.4e}" if 0 < abs(curtailed) < 1e-4 else f"{curtailed:.4f}"
        lines.append(f"| {label} | {record['cost']:.4f} | {shown_curtailment} | {record['mip_gap']:.8g} | {record['solve_seconds']:.6f} |")
    lines += ["", metadata["timing_definition"], "", "最优间隙直接取各阶段的 mip_gap；第二阶段对应弃光目标，第一、三阶段对应费用目标。", ""]
    for label, record in zip(labels, records):
        lines.append(f"- {label}：status={record['status']}，success={record['success']}，{record['message']}")
    lines += ["", "详细记录：stages.json；HiGHS 原始输出：solver.log；运行环境、输入和代码校验值：run.json。",
              "本记录来自上述时间的重新运行，不是此前运行的历史日志。"]
    (run_dir / "三阶段求解记录.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
    else:
        main()
