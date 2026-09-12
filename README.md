# National-Competition

2026 国赛 C 题代码，第一题和第二题分别存放。

| 目录 | 内容 | 入口 |
| --- | --- | --- |
| [q1/代码](q1/代码/) | 第一题：预处理、储能调度模型、绘图、表格导出与扩展分析 | `preprocess.py` → `q1.py` |
| [q2/代码](q2/代码/) | 第二题：A/B 模型、预测优化、滚动控制、Excel 导出与绘图 | `run.py` |

第一题说明见 [q1/代码/README.md](q1/代码/README.md)，第二题说明见 [q2/README.md](q2/README.md)。根目录的 `preprocess.py` 是仓库原有脚本；本次两题代码以各自目录为准。

## 数据位置

代码需要题目原始附件，附件、运行结果和虚拟环境未随此次源码上传。运行前，将 C 题的完整 `附件` 文件夹放在仓库根目录：

```text
National-Competition/
├── 附件/
│   ├── 附件1.xlsx
│   ├── 附件2.xlsx
│   ├── 附件3.xlsx
│   ├── 附件4.xlsx
│   └── 附件5/            # 官方结果模板
├── q1/
│   └── 代码/
└── q2/
    └── 代码/
```

## 第一题

在仓库根目录打开终端：

```powershell
cd q1/代码
python -m pip install -r requirements.txt
python preprocess.py
python q1.py
```

预处理输出位于 `q1/outputs/`，模型结果位于 `q1/results/`。

## 第二题

在仓库根目录打开终端，推荐 Python 3.12：

```powershell
cd q2
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 代码/requirements.txt
.\.venv\Scripts\python.exe -X utf8 代码/run.py run-model --model A
```

结果及检查点位于 `q2/结果/`。模型 B 使用 `run-model --model B`；已有结果可用 `export-excel --model all` 导出，`plot` 生成模型 B 图件。完整年度计算可能耗时较长，支持断点续跑。测试依赖本地已有计算结果；Excel 导出另需配置文档 Python、Node 和 @oai/artifact-tool，详见 [第二题运行说明](q2/代码/README.md)。
