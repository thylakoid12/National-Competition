# C题问题二

本目录包含问题二源码、测试、配置及实际使用的公共内核。下文的 `result/` 描述运行生成的结果目录，结果文件未随源码上传。原问题一位于 `../q1/代码/`，整理没有改动原问题一源码、附件或已有result1。问题一的独立入口、扫描程序、模板导出备份及旧outputs/results已从q2删除。

## 目录

| 位置 | 内容 |
|---|---|
| `common/` | 从问题一提取、第二问实际使用的时间解析、设备参数、储能求解、表格汇总和绘图逻辑 |
| `q2_*.py` | 第二问数据、预测、场景、控制、优化、回测、导出和审计 |
| `test_q2*.py`、`save_acceptance.py` | 15项自动验收 |
| `config_formal_v2.json` | 已冻结的正式配置 |
| `run_formal_v2.py` | 正式运行和检查点恢复入口 |
| `benchmark_bounded.py` | 1月历史日期的统一预算性能检查 |
| `result/result2.xlsx` | 指定单一策略star_M3的正式提交文件 |
| `result/selected_days/` | 主提交策略四个指定日期图表、合并表3、逐槽明细及导出校验 |
| `result/paper/` | 经济、预测、概率、压力、预热与稳定性汇总，以及交付结果说明 |
| `result/formal_v2/` | 六套334天正式回测、预测/残差/场景/锁定计划存档、检查点及审计；六套独立导出在其exports中 |
| `result/formal_v2_stress/` | 负载上冲、光伏下降、多日联合冲击及独立审计 |
| `result/formal_v2_sensitivity/` | 104组独立诊断与完整轨迹traces_v2 |
| `result/verification/` | 测试、原问题一回归证据、环境和性能记录、整理完整性检查 |

旧冒烟实验、未完成formal_v1、失败诊断轨迹和一次性开发脚本已清理。本仓库不包含本地虚拟环境，首次运行请创建 `.venv` 并安装依赖。

## 运行命令

在q2目录执行，使用已有Python 3.12环境；新环境安装`requirements.txt`。原问题一环境不需修改。

```powershell
# 自动验收
.\.venv\Scripts\python.exe -X utf8 save_acceptance.py

# 完整正式运行；检测到本程序检查点时继续
.\.venv\Scripts\python.exe -X utf8 -u run_formal_v2.py

# 全年实际记录与场景风险独立审计
.\.venv\Scripts\python.exe -X utf8 q2_audit.py --run result/formal_v2 --config config_formal_v2.json
.\.venv\Scripts\python.exe -X utf8 q2_risk_audit.py

# 导出六套实验与指定提交策略
.\.venv\Scripts\python.exe -X utf8 q2_pipeline.py export

# 压力实验及其预测重建验收
.\.venv\Scripts\python.exe -X utf8 q2_pipeline.py stress
.\.venv\Scripts\python.exe -X utf8 q2_stress_audit.py

# 论文汇总
.\.venv\Scripts\python.exe -X utf8 q2_postprocess.py tables --run result/formal_v2 --config config_formal_v2.json
```

正式存档不可覆盖。变更数学参数或算法后，应使用新的输出目录；已有完整结果无需重跑全年。重复运行压力或敏感性求解可能产生不同耗时日志，重新实验应指定新的独立目录。不要读取来源不明的pickle检查点。

## 复用及口径

`common.config`保留原时间、区间标签和设备常量；`common.data_io`保留附件1读取和时间解析；`common.optimization.solve_model`保留原储能MILP，只为第二问提供基于日前预测的额外购电起点；`common.tables.build_tables`保留指定时段和四小时汇总；`common.result_io`与`common.plotting`复用原导出及绘图基础。这里没有问题一运行入口或result1导出逻辑。提取前后关键函数的语法树完全一致，证据在`result/verification/common_extraction.json`。

附件2原始365天×144槽为kW，仅在数据层转换一次为kWh。00:10对应00:00—00:10，末槽23:50—24:00；144槽对应145个状态。额定容量12000 kWh，内部电量范围1200—10800，单槽微网侧充放上限5000/6 kWh，双向效率分别0.9。

问题二状态跨日连续，不每日重置或强制日末等于日初。共同V1控制器只读取当前及此前观测；所有场景共用候选G对应的一条日前参考轨迹。紧急时不充电是V1策略限制。现金费用为`price @ G + 5 * price @ H`，终端项仅用于优化，12月31日为0。

F0候选和F0/F1逐目标组成由1月已发预测评分冻结；主系统本次选中F0两分支，互补系统使用CatBoost。预测和历史特征按原发出日期永久存档，日末才生成残差；使用最近60个完整联合日残差。M1—M3共享整日场景顺序，M2名义权重用于概率评分，M3最坏权重不用于CRPS。

工程配置、正式提交策略与历史验证选择分开标明。六套策略使用相同1156次评价预算、144维分量和四起点。每起点预算可完成一轮正负坐标扫描，可能尚未进入细步长；结果是预算内可行近似解，不能宣称全局最优。104组敏感性和稳定性检查只覆盖四个预设日期，未回写正式计划。

## 导出与验证

result2使用真正附件5模板，按用户授权只修正输出副本的144个区间表头。计划量严格为G，充放电和储电量为实际执行值；紧急电量按同日连续正值区间汇总。“全天购电费”按用户确认写计划费加紧急费。各项电量、费用、日期及指定时段均重新读取校验。

15项自动测试、六套334天独立重算、2004个计划的场景费用和TV审计、压力预测重建及七份工作簿重读均已通过。结果比较和实际费用见`result/paper/交付结果说明.md`。原问题一任务期间曾有外部绘图更新，已单独记录；本次目录整理核对原问题一及result1共18个文件哈希完全不变。
