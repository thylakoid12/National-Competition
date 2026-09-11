"""Bounded q1 auxiliary solve; never relabel a time-limited incumbent optimal."""
import warnings
import numpy as np
from common import optimization as q1_model
from scipy.optimize import milp as scipy_milp

def _feasible(x,kwargs,n,tolerance=1e-6):
    if x is None or not np.isfinite(x).all():
        return False
    bounds=kwargs['bounds']
    if np.max(bounds.lb-x)>tolerance or np.max(x-bounds.ub)>tolerance:
        return False
    constraints=kwargs['constraints']
    for c in constraints:
        activity=c.A@x
        if np.max(c.lb-activity)>tolerance or np.max(activity-c.ub)>tolerance:
            return False
    index=q1_model._make_variable_index(n)
    C=x[index.flow['grid_to_battery']]+x[index.flow['pv_to_battery']]
    D=x[index.flow['battery_discharge']]
    if np.minimum(C,D).max()>tolerance:
        return False
    return True

def deterministic_seed(data,cfg,E0):
    original=q1_model.milp
    candidates=[]
    statuses=[]
    n=len(data)
    index=q1_model._make_variable_index(n)
    def bounded(*args,**kwargs):
        options=dict(kwargs.pop('options',{}))
        options.update(time_limit=cfg.seed_time_limit_seconds,mip_rel_gap=cfg.seed_mip_gap,
                       mip_feasibility_tolerance=1e-9,threads=1,mip_max_nodes=cfg.seed_node_limit)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            result=scipy_milp(*args,options=options,**kwargs)
        valid=_feasible(result.x,kwargs,n,cfg.physical_tolerance)
        statuses.append(dict(status=int(result.status),message=result.message,
                             success=bool(result.success),incumbent_checked=valid,
                             node_limit=cfg.seed_node_limit,node_count=int(result.mip_node_count) if result.x is not None else None,
                             mip_gap=float(result.mip_gap) if result.x is not None else None))
        if valid:
            G=result.x[index.flow['grid_to_load']]+result.x[index.flow['grid_to_battery']]
            candidates.append((float(data.price.to_numpy()@G),np.maximum(G,0)))
        return result
    q1_model.milp=bounded
    try:
        schedule,stats=q1_model.solve_model(data,e_max=cfg.e_max,p_max_kw=cfg.power,
                         eta=cfg.eta_c,e_min=cfg.e_min,e_initial=E0)
        return schedule.grid_purchase.to_numpy(),dict(status='bounded_q1_checked',stages=statuses,
                    time_limit_per_stage_seconds=cfg.seed_time_limit_seconds,mip_rel_gap=cfg.seed_mip_gap)
    except RuntimeError as error:
        if candidates:
            _,G=min(candidates,key=lambda row:row[0])
            return G,dict(status='checked_q1_incumbent_not_converged',stages=statuses,error=str(error),
                           time_limit_per_stage_seconds=cfg.seed_time_limit_seconds,optimality_claim=False)
        return None,dict(status='optional_q1_seed_failed',stages=statuses,error=str(error),
                          time_limit_per_stage_seconds=cfg.seed_time_limit_seconds,optimality_claim=False)
    finally:
        q1_model.milp=original
