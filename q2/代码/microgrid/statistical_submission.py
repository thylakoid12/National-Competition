"""完整统计风险结果的官方模板导出；不会改写一月选型。"""

from pathlib import Path
from .models import get_model
from .statistical_experiment import load_run
from .submission import export_excel


def export(source, model, output):
    source=Path(source)
    load_run(source)
    get_model(model)
    return export_excel(source,output,model=model,render=True)
