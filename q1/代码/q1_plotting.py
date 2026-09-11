"""问题一所有图表共用的无界面绘图环境与论文样式。"""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "q1_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import StrMethodFormatter

from q1_config import FIGURE_DIR


COLORS = {
    "load": "#333333",
    "pv": "#d69c32",
    "grid": "#36729a",
    "charge": "#36729a",
    "discharge": "#c47753",
    "price": "#934c70",
}


def set_style() -> None:
    """应用统一论文图样式。"""

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next(
        (
            name
            for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
            if name in fonts
        ),
        None,
    )
    if chinese is None:
        raise RuntimeError("未找到中文字体，请安装 Microsoft YaHei、SimHei 或 Noto Sans CJK SC。")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese, "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "savefig.dpi": 350,
        }
    )


def save_figure(fig, name: str) -> None:
    """以统一精度保存 PNG，并及时释放图对象。"""

    for axis in fig.axes:
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.4f}"))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=350, bbox_inches="tight")
    print(f"已生成：{path}")
    plt.close(fig)
