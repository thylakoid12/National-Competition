# National-Competition

2026 国赛 C 题代码，第一题和第二题分别存放。

| 目录 | 内容 | 入口 |
| --- | --- | --- |
| [q1/代码](q1/代码/) | 第一题：预处理、储能调度模型、绘图、表格导出与扩展分析 | `preprocess.py` → `q1.py` |
| [q2](q2/) | 第二题：预测、场景、滚动控制、优化、回测、审计与测试 | `run_formal_v2.py`；M3 续跑入口 `q2_m3_resume.py` |

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
    └── common/
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
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s . -p "test_q2*.py"
.\.venv\Scripts\python.exe -X utf8 -u run_formal_v2.py
```

结果及检查点位于 `q2/result/`。完整年度回测可能耗时较长，运行与恢复方式见第二题说明。
