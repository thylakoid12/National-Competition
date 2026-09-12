# National-Competition

2026 国赛 C 题代码。当前本机使用 Anaconda 环境 **CUMCM**。

**论文取图与提交入口：[论文用结果](论文用结果/00_从这里开始.html)。** 已集中为两个结果表、五张正文图和一个重绘资料包；入口说明每张图放在哪里，并列明与附件 5 示例的时间表头差异。

HTML 顶部新增第二问建模正文示例，依次解释统计预测、误差场景、计划执行、鲁棒目标、滚动更新与风险选型。下载后用浏览器打开 HTML 即可阅读；GitHub 文件页显示的是源码。

仓库提供源码、论文图表、重绘资料及[精简实验记录](q2/结果/statistical_risk_v1/README.md)。约 457 MB 的完整逐日记录保留在本机，不随 Git 上传。换电脑重跑时请提供原始附件并使用新的输出目录，例如 `python -X utf8 run.py run-all --workers 7 --output ../结果/statistical_risk_v2`。

**真实附件在 `D:\数模\2026国赛`，即 `Code` 上一级。** [数据位置说明](数据位置说明.md) 列出了原始文件、运行缓存与核验记录的对应关系。附件不随源码仓库提供，当前模型读取本机已有 Excel。

| 目录 | 内容 | 说明 |
| --- | --- | --- |
| [q1/代码](q1/代码/) | 问题一：预处理、储能优化、绘图与导出 | [原运行说明](q1/代码/README.md) |
| [q2/代码](q2/代码/) | 问题二：统计预测、M0—M3 消融、风险限额、压力测试 | [新运行说明](q2/代码/README.md) |
| [q2/结果/statistical_risk_v1](q2/结果/statistical_risk_v1/) | 当前统计风险实验，逐日落盘、支持续跑 | [模型与风险结果](q2/结果/statistical_risk_v1/report/模型与风险结果.md) |
| [output/review](output/review/) | 原论文核查、调整方案、代码备份与数据溯源 | [调整方案](output/review/问题二统计风险模型调整方案.md) |

在仓库根目录打开 Anaconda Prompt 或已初始化 conda 的 PowerShell：

```powershell
conda activate CUMCM
cd q2/代码
python -X utf8 run.py run-all --workers 7
python -X utf8 run.py plot
python -X utf8 run.py export --model selected
```

`run-all` 依次进行一月验证选型、二月至十二月连续回测、留出压力测试和报告生成；已有且哈希一致的检查点会复用。费用增幅上限是相对 M2 的 **3%**，在一月验证期筛选，并在后续评价期独立核验。提交 Excel 需完成全部 334 天后单独导出。

主线没有机器学习预测器。CPU 按模型并行，Numba 加速场景回放；当前求解器没有启用 GPU。历史 A/B/C 入口保留在 `q2/代码/legacy_run.py`，与新的 M0—M3 实验不混用。

换电脑可将附件放在 `Code/附件/`，或通过 `--attachments` 显式指定含附件 1、附件 2 的目录；导出还需要其中的 `附件5/result2.xlsx`。不要将缺少数据的单元测试误当成真实年度回测。
