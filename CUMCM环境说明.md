# CUMCM 环境

已在本机 Anaconda 下创建并完成运行检查。

**当前进展：**问题二已切换到 `run.py` 统计风险主线，自动找到上一级真实附件，生成自己的热身库存和预测缓存。M0—M3 与风险保护模型按 CPU 进程并行；CUMCM 中虽保留 CatBoost 供历史入口使用，新主线不调用机器学习预测器。当前结果见 [模型报告](q2/结果/statistical_risk_v1/report/模型与风险结果.md) 与 [绘图数据](q2/结果/statistical_risk_v1/figures/绘图数据/)。

- Anaconda：`D:\Program Files\Anaconda`
- 环境名：`CUMCM`
- Python：3.12.14
- 解释器：`C:\Users\CUIShuijia\.conda\envs\CUMCM\python.exe`
- 完整 Python 依赖版本：`requirements-CUMCM.lock.txt`

## 使用

在 Anaconda Prompt 中执行：

```bat
conda activate CUMCM
cd /d D:\数模\2026国赛\Code
python -m pip check
```

PyCharm 当前项目仍指向原来的 `ml_env`。在项目的 Python Interpreter 设置中添加已有 Conda 环境，选择上面的 `CUMCM\python.exe`。

需要在普通 PowerShell 中直接运行时，可以指定解释器：

```powershell
& 'C:\Users\CUIShuijia\.conda\envs\CUMCM\python.exe' -X utf8 'q2/代码/run.py' --help
```

## Excel 导出

激活 CUMCM 时会自动设置：

- `PYTHONUTF8=1`
- `CODEX_ARTIFACT_PYTHON=C:\Users\CUIShuijia\.conda\envs\CUMCM\python.exe`
- `CODEX_NODE=C:\Users\CUIShuijia\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe`

项目中的 `node_modules/@oai/artifact-tool` 是指向本机已有工具包的目录联接，版本为 2.8.59；来源为 `C:\Users\CUIShuijia\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\@oai\artifact-tool`。该联接和检查产物已加入 Git 忽略规则。换电脑后需要重新配置本机导出运行时和联接。

## 验证结果

- 两题所有 requirements 依赖已安装；CatBoost 1.2.10、HiGHS Python 接口 1.15.1、Numba 0.67.0 与仓库要求一致。
- `pip check` 通过。
- 第一题：合成数据的 144 时段三阶段 MILP 通过，成本 14400 元，最大约束残差为 0。
- 第二题：合成场景的 A/B 初始方案、参考轨迹 LP、Numba 加速、完整优化搜索与反馈回放通过；场景费用与逐时段回放一致。
- 第二题：用附件 2 前 14 天数据调用实际 CatBoost 预测实现通过。
- Excel 读写、Matplotlib 绘图通过。
- 官方 result2.xlsx 模板导出副本后，由 openpyxl 独立回读核对通过，原始模板未修改。
- Conda 激活后的解释器、UTF-8 和导出路径配置通过。

以上列表记录环境初建时的小规模验证。后续已在真实附件上运行统计主线，并独立重放已完成结果，见 `q2/结果/statistical_risk_v1/audit/`。

## 初始检查问题的处理状态

1. 附件放在项目上一级 `D:\数模\2026国赛\`，问题二新主线已支持自动识别，也支持 `--attachments` 显式路径。
2. 附件 1 末尾有额外合计行，第一题预处理需要识别并排除该非时段行。
3. 历史 `run-model` 依赖缺失的 B 实验记录；新主线通过 `prepare → calibrate → evaluate` 独立生成全部所需状态，无需那些旧文件。原入口保留为 `legacy_run.py`。

原始附件保持不变。后续问题二代码调整与运行方式见 [运行说明](q2/代码/README.md)。

## 重建 Python 环境

在尚无同名环境的机器上，从项目根目录运行 `conda env create -f environment.yml`。此文件锁定 Python 依赖；上述本机 Excel 导出路径及工具包联接需另行设置。
