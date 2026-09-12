# 第二问

代码统一放在“代码/”，计算结果和提交材料统一放在“结果/”。

| 模型 | 计算结果 | 全部提交Excel |
| --- | --- | --- |
| A：固定α=1 | 结果/模型A_优化后/ | 结果/模型A_提交材料/ |
| B：联合优化α | 结果/m3_january_only_20260911/ | 结果/模型B_提交材料/ |

每个提交目录都有 result2.xlsx（官方全年三表）和论文结果表.xlsx（汇总及四个日期表1—表3）。电量、费用显示四位小数，天数和时段数显示整数，原始精度保留。

在 q2 目录运行模型A，完成后自动生成Excel：

~~~powershell
.\.venv\Scripts\python.exe 代码/run.py run-model --model A
~~~

只从已有结果重新导出A、B全部Excel：

~~~powershell
.\.venv\Scripts\python.exe 代码/run.py export-excel --model all
~~~

详细参数见 代码/README.md。旧提交表保存在 结果/旧版提交表/。

模型 B 的图片与 Excel 一起保存在 结果/模型B_提交材料/，其中 图/全年总览.png 为正式评价期2—12月的日汇总，图/四个指定日/ 为四个日期的十分钟组合图，图/储能图/ 为独立储能图。重新生成图片可运行：

~~~powershell
.\.venv\Scripts\python.exe 代码/run.py plot
~~~
