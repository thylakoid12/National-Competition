
"""完整模型结果的独立回放核验。"""
from pathlib import Path
import numpy as np
from microgrid.config import CURRENT_RESULT, RESULT_DIR, load_config
from microgrid.storage import read_arrays, write_json
from microgrid.reporting import daily_results
from microgrid.metrics import totals
from tests.oracle import scalar_replay

def audit(source):
    cfg=load_config()
    variant,directory,rows=daily_results(source)
    price=read_arrays(Path(source)/"evaluation_bank.npz")["price"]
    previous=rows[0]["E0"]
    worst=0.
    for row in rows:
        a=read_arrays(directory/(row["date"]+".npz"))
        if abs(row["E0"]-previous)>1e-6:
            raise ValueError("跨日初始状态错误")
        out=scalar_replay(a["G"],a["reference"],row["E0"],
                         np.column_stack([a["load"],a["pv"]]),price,cfg)
        np.testing.assert_allclose(out["E"],a["E"],atol=1e-6,rtol=0)
        np.testing.assert_allclose(out["H"],a["H"],atol=1e-6,rtol=0)
        if abs(out["cash_cost"]-row["cash_cost"])>1e-5:
            raise ValueError("费用与独立回放不符")
        if row["forecast_train_end"]>=row["date"] or row["scene_end"]>=row["date"]:
            raise ValueError("预测或场景用了当日信息")
        if variant=="search_only" and row["alpha"]!=1:
            raise ValueError("模型A参考系数错误")
        worst=max(worst,float(row["max_residual"]))
        previous=row["Eend"]
    return dict(variant=variant,independent_replay_days=len(rows),max_physical_residual=worst,**totals(rows))

if __name__=="__main__":
    result={"A":audit(RESULT_DIR/"模型A_优化后"),"B":audit(CURRENT_RESULT)}
    result["A_minus_B"]={k:result["A"][k]-result["B"][k] for k in ("cash_cost","H","emergency_days")}
    write_json(RESULT_DIR/"整理记录/A与B完整回放核验.json",result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))

