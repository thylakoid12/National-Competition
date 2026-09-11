"""Paired full-day residual scenarios and stable conditional nominal weights."""
import numpy as np
from q2_config import Config,digest

def compute_weights(current,historical,h=1.,shrinkage=.25):
    historical=np.asarray(historical,float)
    if not len(historical):
        return np.empty(0),dict(effective_n=0)
    if h<=0 or not 0<=shrinkage<=1:
        raise ValueError('Invalid kernel settings')
    # First seven fields are weekday one-hot, not numeric weekday distance.
    mu=historical[:,7:].mean(axis=0)
    scale=historical[:,7:].std(axis=0)
    numeric=np.divide(historical[:,7:]-current[7:],scale,
                      out=np.zeros_like(historical[:,7:]),where=scale>0)
    delta=np.c_[historical[:,:7]-current[:7],numeric]
    distance=(delta**2).mean(axis=1)
    logw=-distance/(2*h*h)
    w=np.exp(logw-logw.max())
    w/=w.sum()
    w=(1-shrinkage)*w+shrinkage/len(w)
    return w,dict(effective_n=float(1/(w@w)),mean=mu.tolist(),scale=scale.tolist(),
                  distance=distance.tolist(),bandwidth=h,shrinkage=shrinkage)

def build_scenarios(forecast,records,composition,cfg=Config()):
    date=forecast['meta']['date']
    if any(r['date']>=date for r in records):
        raise ValueError('Future residual supplied')
    records=records[-cfg.residual_window:]
    dates=[r['date'] for r in records]
    if not len(records):
        return dict(paths=np.empty((0,144,2)),weights=np.empty(0),clipping=np.empty((0,144,2)),
                    dates=[],version=digest([date,forecast['meta']['version'],'empty']),
                    sources=[],weight_info=dict(effective_n=0),fallback='M0_no_scenarios')
    residual=np.stack([np.column_stack([r['residual'][composition[t],:,t] for t in (0,1)])
                       for r in records])
    raw=forecast['path'][None]+residual
    paths=np.maximum(raw,0)
    weights,info=compute_weights(forecast['context'],np.array([r['context'] for r in records]),
                                 cfg.bandwidth,cfg.shrinkage)
    sources=[[dict(date=r['date'],predictor='F'+str(composition[t]),
                   version=r['issued'][composition[t]]['meta']['version'],
                   fallback=r['issued'][composition[t]]['meta']['fallback'][t],
                   actual_source=r['issued'][composition[t]]['meta']['actual_branch_sources'][t])
              for t in (0,1)] for r in records]
    version=digest([date,forecast['meta']['version'],sources,weights.tolist(),cfg.version])
    return dict(paths=paths,weights=weights,clipping=paths-raw,dates=dates,sources=sources,
                version=version,weight_info=info,fallback='')
