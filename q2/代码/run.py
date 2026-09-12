"""问题二统一入口。运行 python run.py --help 查看用法。"""

import argparse
import json
from pathlib import Path
from microgrid.config import CODE_DIR, RESULT_DIR, CURRENT_RESULT, load_config


def main():
    parser = argparse.ArgumentParser(description="问题二：一月选型，二月至十二月评价")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "select", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--output", type=Path, default=RESULT_DIR / "m3_current")
        p.add_argument("--config", type=Path, default=CODE_DIR / "config.json")
        p.add_argument(
            "--initial-states",
            type=Path,
            default=CODE_DIR / "inputs/initial_states.json",
        )
        if name != "select":
            p.add_argument(
                "--forecast-archive",
                type=Path,
                help="可选：复用 formal_v2 历史预测缓存",
            )
    for name in ("report", "plot"):
        p = commands.add_parser(name)
        p.add_argument("--source", type=Path, default=CURRENT_RESULT)
        p.add_argument("--output", type=Path)
        if name == "plot":
            p.add_argument("--dates", nargs="+")

    p = commands.add_parser("run-model", help="运行固定模型，完成后自动导出Excel")
    p.add_argument("--model", choices=["A", "B"], required=True)
    p.add_argument("--source", type=Path, default=CURRENT_RESULT)
    p.add_argument("--output", type=Path)
    p = commands.add_parser(
        "export-excel", help="从完成的结果生成全年提交表和论文Excel"
    )
    p.add_argument("--model", choices=["A", "B", "all"], default="all")
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run-model":
        from microgrid.comparison import run_fixed_model
        from microgrid.reporting import report
        from microgrid.submission import export_with_runtime

        output = args.output or RESULT_DIR / f"模型{args.model}_优化后"
        run_fixed_model(args.source, output, load_config(), args.model)
        report(output, output / "exports")
        export_with_runtime(output, RESULT_DIR / f"模型{args.model}_提交材料")
        return
    if args.command == "export-excel":
        from microgrid.submission import export_with_runtime

        if args.model == "all" and args.source:
            parser.error("--source只能用于单个模型")
        models = ("A", "B") if args.model == "all" else (args.model,)
        for model in models:
            source = args.source or (
                RESULT_DIR / "模型A_优化后" if model == "A" else CURRENT_RESULT
            )
            output = (
                args.output / f"模型{model}_提交材料"
                if args.model == "all" and args.output
                else args.output or RESULT_DIR / f"模型{model}_提交材料"
            )
            export_with_runtime(source, output)
        return

    if args.command in ("prepare", "select", "evaluate"):
        from microgrid.experiment import lock_protocol, prepare_bank, select, evaluate

        cfg = load_config(args.config)
        lock_protocol(args.output, cfg, args.initial_states)
        if args.command == "prepare":
            prepare_bank(args.output, cfg, forecast_archive=args.forecast_archive)
            result = {"january_bank": str(args.output / "january_bank.npz")}
        elif args.command == "select":
            result = select(args.output, cfg)
        else:
            result = evaluate(args.output, cfg, args.forecast_archive)
            from microgrid.submission import export_with_runtime

            export_with_runtime(args.output, args.output / "提交材料")
    elif args.command == "report":
        from microgrid.reporting import report

        result = report(args.source, args.output or args.source / "exports")
    else:
        from microgrid.plotting import plot_submission, DEFAULT_DATES
        from microgrid.storage import evaluation_choice

        model = {"search_only": "A", "joint_reserve": "B", "adaptive_joint": "C"}[
            evaluation_choice(args.source)["selected"]
        ]
        result = plot_submission(
            args.source,
            args.output or RESULT_DIR / f"模型{model}_提交材料" / "图",
            args.dates or DEFAULT_DATES,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
