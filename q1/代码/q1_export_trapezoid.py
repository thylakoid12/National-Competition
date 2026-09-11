"""将已有梯形近似调度导出到独立目录。"""

from q1_config import RESULT_DIR
from q1_export_tables import export_tables
from q1_result_io import read_result


def main():
    schedule = read_result(
        RESULT_DIR / "q1_discretization_comparison.xlsx", sheet_name="梯形调度"
    )
    export_tables(schedule, RESULT_DIR / "梯形近似", prefix="梯形近似方案；")


if __name__ == "__main__":
    main()
