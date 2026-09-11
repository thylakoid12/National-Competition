"""Issued forecasts, historical-date features and January-only model selection."""
from datetime import date as Date, timedelta
from dataclasses import asdict
import warnings
import numpy as np
from q2_config import Config, digest

LOAD_METHODS = ('previous_day','previous_week','weekday_2','weekday_3','mean_7')
PV_METHODS = ('mean_1','mean_3','mean_7','mean_14','linear_7','linear_14')

def context(history, date):
    weekday = np.eye(7)[Date.fromisoformat(date).weekday()]
    if not len(history):
        return np.r_[weekday, np.zeros(5)]
    totals = history[-7:].sum(axis=1)
    return np.r_[weekday, totals[-1,0], totals[:,0].mean(), totals[-1,1],
                 totals[:,1].mean(), totals[:,1].std()]

def candidates(history, target):
    if not len(history):
        raise ValueError('F0 requires at least one completed day')
    y = history[:,:,target]
    result, flags = [], []
    for method in (LOAD_METHODS if target==0 else PV_METHODS):
        fallback = False
        if method=='previous_day':
            out=y[-1]
        elif method=='previous_week' or method.startswith('weekday_'):
            weeks=1 if method=='previous_week' else int(method.split('_')[1])
            indices=[len(y)-7*k for k in range(1,weeks+1) if len(y)>=7*k]
            fallback=not indices
            out=y[indices].mean(axis=0) if indices else y.mean(axis=0)
        else:
            kind,k=method.split('_')
            part=y[-int(k):]
            if kind=='linear' and len(part)>=2:
                x=np.arange(len(part),dtype=float)
                slope=(x-x.mean())@part/np.sum((x-x.mean())**2)
                out=part.mean(axis=0)+slope*(len(part)-x.mean())
            else:
                fallback=kind=='linear'
                out=y.mean(axis=0) if fallback else part.mean(axis=0)
        result.append(np.maximum(out,0))
        flags.append(fallback)
    return np.array(result),flags

def features(history, date):
    """144 rows, using only days strictly before date, including for training rows."""
    n=144
    slot=np.arange(n)
    columns=[slot,np.sin(2*np.pi*slot/n),np.cos(2*np.pi*slot/n)]
    columns += [np.full(n,float(Date.fromisoformat(date).weekday()==w)) for w in range(7)]
    for lag in (1,7):
        for target in (0,1):
            columns.append(history[-lag,:,target] if len(history)>=lag else np.full(n,np.nan))
    for window in (3,7,14):
        for target in (0,1):
            part=history[-window:,:,target]
            columns += [part.mean(axis=0),part.std(axis=0)] if len(part) else [np.full(n,np.nan)]*2
    if len(history):
        totals=history.sum(axis=1)
        constants=np.r_[totals[-1],totals[-7:].mean(axis=0)]
    else:
        constants=np.full(4,np.nan)
    columns += [np.full(n,v) for v in constants]
    return np.column_stack(columns)

class Forecaster:
    def __init__(self,cfg=Config()):
        self.cfg=cfg
        self.scores=[np.zeros(len(LOAD_METHODS)),np.zeros(len(PV_METHODS))]
        self.scored_days=0
        self.records=[]
        self.frozen=None
        self.composition=None
        self.comparison=[]
        self.freeze_report=None

    def freeze(self):
        if self.frozen is not None:
            return
        self.frozen=[int(np.argmin(x)) for x in self.scores]
        effective=[r for r in self.comparison if r['ml_valid']]
        if effective:
            mae=np.mean([r['mae'] for r in effective],axis=0)
            rmse=np.sqrt(np.mean([r['mse'] for r in effective],axis=0))
            self.composition=[int(mae[1,t]<mae[0,t]) for t in (0,1)]
        else:
            mae=rmse=np.full((2,2),np.nan)
            self.composition=[0,0]
        self.freeze_report=dict(date='2025-02-01',train_end='2025-01-31',
                                f0_methods=[LOAD_METHODS[self.frozen[0]],PV_METHODS[self.frozen[1]]],
                                candidate_mae=[(s/max(self.scored_days,1)).tolist() for s in self.scores],
                                star=self.composition,dagger=[1-x for x in self.composition],
                                effective_ml_days=len(effective),mae=mae.tolist(),rmse=rmse.tolist(),
                                config=asdict(self.cfg),parameter_basis=self.cfg.parameter_basis)

    def issue(self,history,date):
        if date>='2025-02-01':
            self.freeze()
        arrays,flags=zip(*(candidates(history,t) for t in (0,1)))
        chosen=self.frozen if self.frozen is not None else [int(np.argmin(x)) for x in self.scores]
        f0=np.column_stack([arrays[t][chosen[t]] for t in (0,1)])
        methods=[LOAD_METHODS[chosen[0]],PV_METHODS[chosen[1]]]
        ml_reason='insufficient_history' if len(history)<self.cfg.ml_min_days else ''
        f1=f0.copy()
        rows=0
        if not ml_reason:
            try:
                from catboost import CatBoostRegressor
                first=max(0,len(history)-self.cfg.ml_window)
                start_date=Date.fromisoformat(date)-timedelta(days=len(history))
                X=np.concatenate([features(history[:j],(start_date+timedelta(days=j)).isoformat())
                                  for j in range(first,len(history))])
                Y=history[first:].reshape(-1,2)
                rows=len(Y)
                query=features(history,date)
                predictions=[]
                for t in (0,1):
                    model=CatBoostRegressor(depth=self.cfg.ml_depth,iterations=self.cfg.ml_iterations,
                         learning_rate=self.cfg.ml_learning_rate,loss_function='RMSE',
                         random_seed=self.cfg.seed,thread_count=1,verbose=False,allow_writing_files=False)
                    model.fit(X,Y[:,t])
                    predictions.append(np.maximum(model.predict(query),0))
                f1=np.column_stack(predictions)
            except ImportError as exc:
                ml_reason='catboost_unavailable: '+str(exc)
            except Exception as exc:
                ml_reason='catboost_training_failed: '+repr(exc)
                warnings.warn(ml_reason)
        z=context(history,date)
        issued=[]
        for i,path in enumerate([f0,f1]):
            meta=dict(date=date,issued_at=date+'T00:00:00',
                      train_end=(Date.fromisoformat(date)-timedelta(days=1)).isoformat(),
                      predictor_id='F'+str(i),config_version=self.cfg.version,
                      model_version=self.cfg.forecast_version,
                      methods=methods if i==0 else ['CatBoost','CatBoost'],
                      fallback=[flags[t][chosen[t]] for t in (0,1)] if i==0 else [bool(ml_reason)]*2,
                      fallback_reason='' if i==0 else ml_reason,
                      actual_branch_sources=methods if i==0 or ml_reason else ['CatBoost','CatBoost'],
                      historical_days=len(history),training_rows=rows if i==1 else 0)
            meta['version']=digest([meta,path.tolist(),z.tolist()])
            path=path.copy()
            path.flags.writeable=False
            issued.append(dict(meta=meta,path=path,context=z.copy()))
        return dict(date=date,issued=issued,candidates=arrays,candidate_fallback=flags)

    def settle(self,issued,actual):
        """Only call after that day's control and settlement, never overwrite forecasts."""
        date=issued['date']
        if self.records and date<=self.records[-1]['date']:
            raise ValueError('Residual dates must increase strictly')
        residual=np.stack([actual-r['path'] for r in issued['issued']])
        entry=dict(date=date,residual=residual,issued=issued['issued'],
                   context=issued['issued'][0]['context'])
        self.records.append(entry)
        error=np.stack([actual-r['path'] for r in issued['issued']])
        comparison=dict(date=date,ml_valid=not any(issued['issued'][1]['meta']['fallback']),
                        mae=np.abs(error).mean(axis=1).tolist(),mse=(error**2).mean(axis=1).tolist())
        if date<'2025-02-01':
            self.comparison.append(comparison)
            for t in (0,1):
                self.scores[t]+=np.abs(issued['candidates'][t]-actual[:,t]).mean(axis=1)
            self.scored_days+=1
        return entry,comparison

def combine(issued,composition,system):
    branches=[issued['issued'][composition[t]] for t in (0,1)]
    meta=dict(date=issued['date'],predictor_id=system,composition=composition,
              branch_versions=[b['meta']['version'] for b in branches],
              branch_sources=[b['meta']['actual_branch_sources'][t] for t,b in enumerate(branches)],
              fallback=[b['meta']['fallback'][t] for t,b in enumerate(branches)])
    meta['version']=digest(meta)
    return dict(meta=meta,path=np.column_stack([b['path'][:,t] for t,b in enumerate(branches)]),
                context=branches[0]['context'])
