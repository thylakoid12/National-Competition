"""Immutable JSON/NPZ artifacts; separate directories isolate experiment branches."""
import json
from pathlib import Path
import numpy as np

def json_value(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def save_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,default=json_value,sort_keys=True)
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:
            raise FileExistsError('Refusing to overwrite immutable record: '+str(path))
        return
    with path.open('x',encoding='utf-8') as f:
        f.write(text)

def save_npz(filename,**arrays):
    path=Path(filename)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays):
                raise FileExistsError(path)
            for k in arrays:
                if not np.array_equal(old[k],arrays[k],equal_nan=True):
                    raise FileExistsError('Immutable array mismatch: '+str(path)+' '+k)
        return
    with path.open('xb') as f:
        np.savez_compressed(f,**arrays)

def save_issue(root,issued):
    for f in issued['issued']:
        dest=Path(root)/'forecasts'/f['meta']['predictor_id']/issued['date']
        save_json(dest.with_suffix('.json'),f['meta'])
        save_npz(dest.with_suffix('.npz'),path=f['path'],context=f['context'])
    save_npz(Path(root)/'candidates'/(issued['date']+'.npz'),
             load=issued['candidates'][0],pv=issued['candidates'][1])

def save_scenarios(root,system,date,bundle):
    dest=Path(root)/'scenarios'/system/date
    save_json(dest.with_suffix('.json'),{k:v for k,v in bundle.items() if k not in ['paths','weights','clipping']})
    save_npz(dest.with_suffix('.npz'),paths=bundle['paths'],weights=bundle['weights'],clipping=bundle['clipping'])

def save_rollout(root,strategy,date,plan,out):
    dest=Path(root)/'runs'/strategy/date
    save_json(dest.with_suffix('.json'),dict(plan=plan['log'],
              settlement={k:out[k] for k in ['plan_cost','emergency_cost','cash_cost','residuals']}))
    save_npz(dest.with_suffix('.npz'),**{k:v for k,v in out.items() if isinstance(v,np.ndarray)},
             reference=plan['reference'],worst_q=plan['worst_q'],scenario_costs=plan['Q'])
