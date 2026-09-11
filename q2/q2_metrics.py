"""Nominal probabilistic scores and independent cash/tail accounting."""
import numpy as np
from q2_risk import cvar

def distribution_metrics(samples,observed,weights):
    samples=np.asarray(samples,float)
    if samples.ndim==1:
        samples=samples[:,None]
    observed=np.atleast_1d(observed)
    order=np.argsort(samples,axis=0,kind='stable')
    x=np.take_along_axis(samples,order,axis=0)
    w=np.take_along_axis(np.broadcast_to(weights[:,None],samples.shape),order,axis=0)
    cumulative=np.cumsum(w,axis=0)
    # 0.5 E|X-X'| computed in O(S log S), not using worst-case weights.
    half_pair=np.sum(w*x*(2*cumulative-w-1),axis=0)
    crps=np.sum(weights[:,None]*np.abs(samples-observed),axis=0)-half_pair
    lo=x[np.argmax(cumulative>=.05,axis=0),np.arange(x.shape[1])]
    hi=x[np.argmax(cumulative>=.95,axis=0),np.arange(x.shape[1])]
    return dict(crps=float(crps.mean()),coverage=float(((observed>=lo)&(observed<=hi)).mean()),
                width=float((hi-lo).mean()))

def probability_scores(bundle,actual,price):
    if not len(bundle['paths']):
        return {}
    net=bundle['paths'][:,:,0]-bundle['paths'][:,:,1]
    observed=actual[:,0]-actual[:,1]
    high=price>=np.quantile(price,.75)
    out={}
    for model,w in [('M1',np.ones(len(net))/len(net)),('M2',bundle['weights'])]:
        out[model]=dict(slot=distribution_metrics(net,observed,w),
                        high_price_daily=distribution_metrics(net[:,high].sum(axis=1),observed[high].sum(),w))
    return out

def summarize(rows):
    keys=['plan_cost','emergency_cost','cash_cost','emergency_kwh','curtailment_kwh','unused_kwh']
    result={key:float(sum(r[key] for r in rows)) for key in keys}
    result.update(days=len(rows),emergency_days=sum(r['emergency_kwh']>1e-6 for r in rows),
                  emergency_slots=sum(r['emergency_slots'] for r in rows),
                  cash_cvar95=cvar([r['cash_cost'] for r in rows]),
                  emergency_cvar95=cvar([r['emergency_cost'] for r in rows]))
    return result

def point_scores(records):
    out={}
    for name,subset in [('all',records),('common_ml_valid',[r for r in records if r['ml_valid']])]:
        if not subset:
            out[name]=dict(days=0)
            continue
        out[name]=dict(days=len(subset),mae=np.mean([r['mae'] for r in subset],axis=0).tolist(),
                       rmse=np.sqrt(np.mean([r['mse'] for r in subset],axis=0)).tolist(),
                       pv_daylight_mae=np.mean([r['pv_daylight_mae'] for r in subset],axis=0).tolist(),
                       pv_daylight_rmse=np.sqrt(np.mean([r['pv_daylight_mse'] for r in subset],axis=0)).tolist(),
                       daylight_window='06:00-18:00, predeclared, slots 36:108')
    return out
