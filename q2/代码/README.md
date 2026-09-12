# 第二问：运行与生成 Excel

在本目录（q2/代码）运行。

## 一条命令：模型 A 全年计算、断点续跑、生成 Excel

~~~powershell
..\.venv\Scripts\python.exe run.py run-model --model A
~~~

模型 A 使用与现有 B 一致的预测、历史场景与共同初始储电量，参考系数 α 固定为 1。
逐日优化 2025 年 2 月 1 日至 12 月 31 日，共334天。
每天结果落盘，中断后重复命令即可接着算。完成后自动生成两份 Excel。

- 计算过程：q2/结果/模型A_优化后/
- 提交文件：q2/结果/模型A_提交材料/result2.xlsx
- 论文表格：q2/结果/模型A_提交材料/论文结果表.xlsx

## 已有结果直接导出 A、B 全部 Excel

~~~powershell
..\.venv\Scripts\python.exe run.py export-excel --model all
~~~

也可以单独导出：

~~~powershell
..\.venv\Scripts\python.exe run.py export-excel --model A
..\.venv\Scripts\python.exe run.py export-excel --model B
~~~

模型 B 默认使用已完成的一月选型版，不因导出重复计算。
若确实需要从头运行固定模型 B：

~~~powershell
..\.venv\Scripts\python.exe run.py run-model --model B
~~~

完成的模型 B 新跑结果用下面的命令明确指定来源导出：

~~~powershell
..\.venv\Scripts\python.exe run.py export-excel --model B --source ../结果/模型B_优化后
~~~

## 每个模型生成哪些表

| 文件 | 内容 |
| --- | --- |
| result2.xlsx | 按官方模板保留三个工作表：全年计划购电量、充放电量、紧急购电量 |
| 论文结果表.xlsx | 评价期汇总；3月20日、6月21日、9月23日、12月21日的表1、表2、表3 |
| 使用说明.md | 模型定义、单位、费用口径和模板时间标题修正说明 |
| 核验记录/ | 逐值核验结果、来源记录及可选预览 |

导出时会检查全年334天是否完整，重新读入Excel核对数值，并核对应急区间合计。
未算完不会生成看似完整的提交表。官方模板原文件不改动。

## 代码位置

- run.py：统一命令入口。
- microgrid/comparison.py：固定 A/B 定义的全年对照运行。
- microgrid/experiment.py：逐日求解、结算与断点续跑。
- microgrid/submission.py：构建所有提交数据、调用Excel导出、核验导出值。
- 交表工具/export_submission.mjs：官方模板填报及论文结果表排版。
- microgrid/forecast.py、optimizer.py、controller.py：预测、优化与储能反馈。
- config.json：参数；inputs/：共同初始状态等固定输入。

Excel导出使用本机已有的文档运行环境，无需手工复制数据。
换电脑时需配置文档Python和Node运行环境；可通过 CODEX_ARTIFACT_PYTHON、CODEX_NODE 指定路径，
并为交表工具安装对应的 @oai/artifact-tool 依赖。

## 保留原一月选型流程

~~~powershell
..\.venv\Scripts\python.exe run.py prepare
..\.venv\Scripts\python.exe run.py select
..\.venv\Scripts\python.exe run.py evaluate
~~~

该流程默认保存到 q2/结果/m3_current/，完整评价后也会自动输出该目录下的提交材料。
一月选型的胜出方案仍为 B；A 的全年运行是用户指定的固定方案对照，不能改写为一月选型胜出者。


## 生成模型 B 提交图件

在代码目录运行：

~~~powershell
..\.venv\Scripts\python.exe run.py plot
~~~

图片保存到 ../结果/模型B_提交材料/图/。全年总览为2—12月334天的日汇总；四个指定日为3月20日、6月21日、9月23日、12月21日的十分钟结果。实际负荷、实际光伏、计划购电与紧急购电合在一张图中，紧急购电以红色突出并使用右轴；左右轴刻度不同。储能图单独保存。每张图同时输出PNG和SVG，数值标注保留四位小数。
