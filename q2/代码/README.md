# 统计风险模型运行说明

本目录的 `run.py` 是新主线。使用本机已配置的 Anaconda 环境 **CUMCM**（Python 3.12）。

论文主消融为 M0—M3。`--models all` 另外包含 M3-R、M3-S、M3-RS 三个风险保护扩展；本次实验保存了这七项真实对照，但报告中主模型与扩展实验分别展示。仅做四模型的新实验可用 `run-all --models M0 M1 M2 M3 --output ../结果/main_four_v1`；应使用新输出目录，不能改写本次七候选冻结记录。

```powershell
conda activate CUMCM
cd D:/数模/2026国赛/Code/q2/代码
python -X utf8 run.py --help
```

如果终端尚未初始化 conda，可直接调用：

```powershell
& C:/Users/CUIShuijia/.conda/envs/CUMCM/python.exe -X utf8 run.py --help
```

## 真实数据

默认按顺序寻找 `Code/附件/`、`Code/`、`Code` 上一级中的附件 1 和附件 2。本机实际读取 **`D:\数模\2026国赛`**；附件 1 为分时电价，附件 2 为全年真实负载与光伏功率，十分钟电量由 kW 除以 6 得到。详见 [数据位置说明](../../数据位置说明.md)。

模型不会在缺少原始附件时自动生成模拟数据替代正式输入。

## 完整实验与分步续跑

```powershell
python -X utf8 run.py run-all --workers 7
```

分步执行等价流程：

```powershell
python -X utf8 run.py prepare
python -X utf8 run.py calibrate --models all --workers 7
python -X utf8 run.py evaluate --models all --workers 7
python -X utf8 run.py stress --workers 2
python -X utf8 run.py report
python -X utf8 run.py plot
python -X utf8 run.py export --model selected
```

`prepare` 保存输入和代码哈希、因果预测缓存及从题设 6000 kWh 起算的一月热身库存。`calibrate` 在 1 月 15—31 日比较七个候选；费用不超过 M2 的 1.03 倍后，优先降低实测紧急费用与开发压力费用的尾部风险。`evaluate` 在 2 月 1 日至 12 月 31 日进行 334 天连续回测。`stress` 默认只测试冻结所选模型与 M2，共 4 个季节起点 × 6 种留出冲击 × 2 个模型。`export` 只接受完整结果，不修改冻结选型。

每个模型单独逐日落盘。中断后重复同一条命令会验证哈希和库存连续性并续跑。代码、输入或参数变动时，旧实验拒绝继续，应保留旧目录并指定新的 `--output`：

```powershell
python -X utf8 run.py run-all --config my_config.json --attachments D:/数模/2026国赛 --output ../结果/statistical_risk_v2 --workers 7
```

`evaluate --max-days 2` 仅供冒烟检查，生成的部分结果不能提交。当前默认每个模型每天最多 3000 次候选评估、3 个起点，属于局部搜索，不认证全局最优。

## 结果目录

默认输出 `../结果/statistical_risk_v1/`：

| 路径 | 内容 |
| --- | --- |
| `protocol_frozen.json` | 数据路径、原始附件哈希、参数、计算代码哈希与实验规则 |
| `january_bank.npz`、`evaluation_bank.npz` | 真实附件和因果统计预测缓存 |
| `initial_states.json`、`warmup/` | 共同热身过程与评价初始库存 |
| `selection_frozen.json` | 一月冻结的模型选择 |
| `validation/模型/`、`evaluation/模型/` | 逐日真实结算、优化日志、事前风险和完整期汇总 |
| `*/模型/locked_plans/` | 看到当日真实数据前已确定的购电计划与参考库存 |
| `stress/`、`stress_summary.json` | 留出压力案例及恢复过程 |
| `report/模型与风险结果.md` | 同一来源的费用、风险与压力结果 |
| `figures/` | M3/M2 全年、指定日、储能、风险与压力对照图；PNG/SVG |
| `figures/绘图数据/` | 各模型 48096 行十分钟 CSV，含真实曲线、预测、计划和结算 |
| `submission/所选模型/` | 官方 `result2.xlsx`、`论文结果表.xlsx` 与导出核验记录 |

常规费用由实际附件结算。压力情景是对真实曲线的人工扰动，未假定其发生概率。目前允许无限紧急补购，压力结果刻画费用与应急依赖，不能解释为停电可靠性。

`python -X utf8 run.py plot` 只读取已保存结果，不重新求解，默认绘制所选模型与 M2。用 `--models all` 可为所有已完成模型导出十分钟明细与图件。流量电量乘 6 可转为功率 kW；储电量是库存，不能乘 6。字段解释见 `figures/绘图说明.md`。

## 模型与计算

[模型说明.md](模型说明.md) 给出统计预测、联合残差、TV、CVaR、购电搜索、日内控制与选型的公式和顺序。

`requirements.txt` 是统计主线依赖。`requirements-legacy-ml.txt` 仅供历史机器学习入口。当前 CPU 按模型并行，每个 HiGHS 实例单线程；Numba 编译场景回放，缓存重复参考计划。`--workers` 最高 8；本机 24 个逻辑处理器使用 7 个模型进程。没有启用 GPU。

Excel 通过本机缓存的 Node 与 `@oai/artifact-tool` 导出，openpyxl 只负责读取核验。换电脑需配置对应运行环境；Node 可由 `CODEX_NODE` 指定。官方模板原件不改动。

## 验证

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m tests.audit_real_results
```

新主线测试使用构造数据独立核对 CVaR、物理控制、因果顺序、检查点、防篡改、风险筛选与多日压力。它们不是正式实验数据。依赖历史年度结果的旧 A/B 测试在缺少对应历史附件时明确跳过；当前实际年度数据另做独立重放审计。

历史入口保存在 `legacy_run.py`，历史 A/B/C 的名称、冻结结果和新 M0—M3 不作等同。原论文需要按新实验方法与实际结果同步修改。
