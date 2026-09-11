# C 题问题一代码说明

代码按配置、预处理、模型、输出和扩展分析组织，保留独立脚本入口。

## 运行

在本目录执行：

```powershell
python -m pip install -r requirements.txt
python preprocess.py
python q1.py
```

原始附件位于仓库根目录的 `../../附件/`，预处理数据和基础图写入 `../outputs/`，模型结果、论文图和提交表格写入 `../results/`。

扩展分析按需要运行：

```powershell
python q1_sensitivity.py
python q1_capacity_scan.py
python q1_capacity_efficiency.py
python q1_discretization.py
python q1_export_trapezoid.py
```

容量和敏感性分析依赖预处理数据；离散化比较还需要先运行 `q1.py`，梯形表格导出需要先运行 `q1_discretization.py`。`plot_q1.py` 和 `q1_export_tables.py` 可以单独运行，重新生成已有基准调度的图和表。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `q1_config.py` | 路径、时间尺度、基准参数 |
| `preprocess.py` | 读取附件 1，检查时间和数值，生成标准数据及基础图 |
| `q1_model.py` | 读取标准数据、构建和求解 MILP、检查约束 |
| `q1.py` | 基准求解入口，串联结果保存、绘图和表格导出 |
| `q1_result_io.py` | 结果读写、中英文列名转换、Excel 样式 |
| `q1_plotting.py` | 绘图环境、配色和公共样式 |
| `plot_q1.py` | 基准调度图 |
| `q1_export_tables.py` | 论文表格和官方结果模板导出 |
| `q1_sensitivity.py` | 单参数分析、归一化敏感性系数及对应图表 |
| `q1_capacity_scan.py` | 容量扫描和边际价值分析 |
| `q1_capacity_efficiency.py` | 容量与效率二维分析 |
| `q1_discretization.py` | 右端矩形法与梯形近似比较 |
| `q1_export_trapezoid.py` | 已有梯形调度的表格导出 |

## 维护约定

- 改基准参数和路径：修改 `q1_config.py`。
- 改目标函数或约束：修改 `q1_model.py`；所有分析都从这里导入模型。
- 增加分析：调用 `load_data()` 和 `solve_model(data, 参数名=取值)`，复用结果读写和绘图模块。
- 改结果字段：同步更新 `q1_result_io.py` 的中英文列名映射。
- 附件按本题固定列名读取；更换来源时修改 `preprocess.py` 的列名映射。

主流程直接把调度传给 `plot_dispatch(schedule)` 和 `export_tables(schedule)`，不通过重新读取文件传递数据。数据入口检查时间和数值，模型检查求解状态及约束，图表层负责呈现。单参数分析每次重新计算基准，避免引用旧汇总表。

容量扫描会复用相同模型版本的历史成功点。更改输入数据或基准参数后，应删除 `../results/q1_capacity_dense_scan.xlsx` 再运行；更改模型算法后更新 `MODEL_REVISION`。

## 模型约定

- 一天划分为 144 个 10 分钟时段，调度电量单位为 kWh。
- 储能状态含首末时刻，共 145 个点。
- 三阶段依次最小化购电费、在成本容差内最小化弃光、在弃光容差内重新最小化购电费。
- 本次整理不改变目标函数、约束或求解容差。

## 版本与验证

项目根目录已建立本地 Git 仓库，代码、说明和原始附件纳入版本管理；生成结果、图片、缓存及 `tmp/` 不纳入版本管理。

- `before-cleanup`：本次整理前的代码。
- `after-cleanup`：整理并验证后的代码。
- 在项目根目录执行 `git log --oneline` 查看历史；执行 `git revert after-cleanup` 撤销本次整理并保留版本记录。

本次已对照备份验证：预处理数据完全一致；基准、容量变化、效率变化、梯形近似四种方案的调度和全部求解指标一致。另已运行预处理、主流程、单参数分析、梯形比较和表格导出，检查容量扫描及缓存复用，并验证二维分析的绘图、报告和小规模求解。验证产物位于项目根目录的 `tmp/cleanup-verification/`。

## 保留三阶段求解日志

运行 `python q1_run_audit.py` 可重跑基准流程，按运行时间归档到 `../results/logs/`。记录包括各阶段费用、弃光量、相对最优间隙、实测耗时、状态码、求解器消息及逐时段购电与弃光量。

`solver.log` 保存 HiGHS 原始输出，`stages.json` 保存完整精度的阶段结果，`三阶段求解记录.md` 为可读表格，`run.json` 保存运行环境和输入、代码校验值。每次求解调用单独计时，不含模型构建、绘图和导出。该入口依赖先运行 `preprocess.py`。
