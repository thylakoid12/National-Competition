"""问题一入口：求解基准方案，生成结果表和图。"""

import pandas as pd

from plot_q1 import plot_dispatch
from q1_config import RESULT_DIR
from q1_export_tables import export_tables
from q1_model import load_data, solve_model
from q1_result_io import write_result

def main(*, stage_records=None, solver_log=False):
    """执行问题一的基准求解与标准输出流程。"""

    schedule, stats = solve_model(
        load_data(), stage_records=stage_records, solver_log=solver_log
    )
    write_result(schedule, RESULT_DIR / "q1_schedule.xlsx")
    write_result(pd.DataFrame([stats]), RESULT_DIR / "q1_summary.xlsx")

    print(f"第一阶段最优成本：{stats['optimal_cost']:.4f} 元")
    print(f"最终调度成本：{stats['final_cost']:.4f} 元")
    print(f"购电量：{stats['grid_purchase']:.4f} kWh；弃光：{stats['curtailment']:.4f} kWh")
    print(f"光伏消纳率：{stats['pv_utilization']:.4%}")
    print(
        f"储能范围：{stats['energy_min']:.4f}–{stats['energy_max']:.4f} kWh；"
        f"末端：{stats['energy_final']:.4f} kWh"
    )
    print(f"最大约束残差：{stats['max_constraint_residual']:.4e}")

    plot_dispatch(schedule)
    export_tables(schedule)


if __name__ == "__main__":
    main()
