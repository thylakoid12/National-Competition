"""Reuse q1's checked solver; tighten MIP tolerance only if its own check fails."""
import warnings
from common import optimization as q1_model
from scipy.optimize import milp as scipy_milp

def tight_milp(*args,**kwargs):
    options=dict(kwargs.pop('options',{}))
    options['mip_feasibility_tolerance']=1e-9
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
        return scipy_milp(*args,options=options,**kwargs)

def deterministic_seed(data,cfg,E0):
    kwargs=dict(e_max=cfg.e_max,p_max_kw=cfg.power,eta=cfg.eta_c,e_min=cfg.e_min,e_initial=E0)
    try:
        schedule,stats=q1_model.solve_model(data,**kwargs)
        return schedule.grid_purchase.to_numpy(),dict(status='q1_checked',attempts=1)
    except RuntimeError as first:
        original=q1_model.milp
        q1_model.milp=tight_milp
        try:
            schedule,stats=q1_model.solve_model(data,**kwargs)
            return schedule.grid_purchase.to_numpy(),dict(status='q1_checked_after_tighter_mip_retry',
                        attempts=2,initial_error=str(first),mip_feasibility_tolerance=1e-9)
        except RuntimeError as second:
            # G>=0 is always feasible under V1 emergency supply. Preserve the same
            # fourth-start budget if this optional seed fails, and report failure.
            return None,dict(status='optional_q1_seed_failed',attempts=2,
                              initial_error=str(first),retry_error=str(second))
        finally:
            q1_model.milp=original
