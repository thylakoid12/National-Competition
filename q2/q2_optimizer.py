"""144-dimensional best-improvement direct search with a fixed shared budget."""
from time import perf_counter
import hashlib
import numpy as np
from q2_config import Config,digest
from q2_controller import solve_reference,batch_costs,rollout
from q2_risk import aggregate

class Evaluator:
    def __init__(self,forecast,scenarios,E0,price,cfg=Config()):
        self.forecast,self.scenarios,self.E0,self.price,self.cfg=forecast,scenarios,E0,price,cfg
        self.nu=cfg.terminal(forecast['meta']['date'],price)
        self.prefix=digest([forecast['meta'],E0,price.tolist(),cfg.version,self.nu,
                            cfg.controller_version,scenarios['version']])
        self.cache={}
        self.lp_count=0
        self.reference_solver=None
        self.path_costs=batch_costs
        if cfg.reference_backend=='highspy':
            from q2_fast import FastReference
            self.reference_solver=FastReference(forecast['path'],E0,price,self.nu,cfg)
        if cfg.accelerated_paths:
            from q2_fast import fast_batch
            self.path_costs=fast_batch

    def key(self,G):
        return self.prefix+hashlib.sha256(np.asarray(G,dtype='<f8').tobytes()).hexdigest()

    def evaluate(self,G,model):
        key=self.key(G)
        if key not in self.cache:
            ref,log=(self.reference_solver.solve(G) if self.reference_solver is not None else
                     solve_reference(G,self.forecast['path'],self.E0,self.price,self.nu,self.cfg))
            self.lp_count+=2
            self.cache[key]=dict(reference=ref,lp=log,Q0=None,Q=None)
        item=self.cache[key]
        effective=model if len(self.scenarios['paths']) else 'M0'
        if effective=='M0':
            if item['Q0'] is None:
                item['Q0']=self.path_costs(G,item['reference'],self.E0,self.forecast['path'][None],
                                       self.price,self.nu,self.cfg)
            Q=item['Q0']
            weights=np.ones(1)
        else:
            if item['Q'] is None:
                item['Q']=self.path_costs(G,item['reference'],self.E0,self.scenarios['paths'],
                                      self.price,self.nu,self.cfg)
            Q=item['Q']
            weights=self.scenarios['weights']
        risk,q=aggregate(effective,Q,weights,self.cfg.radius)
        return float(self.price@G+risk),item,q,Q

def optimize_plan(model,forecast,scenarios,E0,price,cfg=Config()):
    started=perf_counter()
    evaluator=Evaluator(forecast,scenarios,E0,price,cfg)
    net=np.maximum(forecast['path'][:,0]-forecast['path'][:,1],0)
    starts=[net,np.zeros(144),1.1*net]
    names=['positive_net','zero','positive_net_110pct']
    seed_log=dict(status='disabled')
    if cfg.deterministic_start:
        import pandas as pd
        if cfg.bounded_seed:
            from q2_seed_bounded import deterministic_seed
        else:
            from q2_seed import deterministic_seed
        from common.config import interval_labels
        source=pd.DataFrame(dict(time=interval_labels(),load=forecast['path'][:,0],
                                 pv=forecast['path'][:,1],price=price))
        seed,seed_log=deterministic_seed(source,cfg,E0)
        starts.append(seed if seed is not None else net.copy())
        names.append('q1_forecast_only_equal_terminal_seed' if seed is not None else 'q1_seed_failed_positive_net_fallback')
    nstarts=len(starts)
    if cfg.budget<2*nstarts:
        raise ValueError('Budget must support all starts')
    best=None
    logs=[]
    evaluations=0
    allocation=[cfg.budget//nstarts+(i<cfg.budget%nstarts) for i in range(nstarts)]
    for si,start in enumerate(starts):
        G=start.copy()
        value,_,_,_=evaluator.evaluate(G,model)
        evaluations+=1
        initial=value
        used=1
        rounds=[]
        ki=0
        while used<allocation[si] and ki<len(cfg.steps):
            step=cfg.steps[ki]*cfg.B
            candidate_best=(value,G)
            attempted=0
            # Candidates in a round ALL share the current base G.
            # Accept largest objective decrease after the round; ties retain first.
            for t in range(144):
                for sign in (1,-1):
                    if used>=allocation[si]:
                        break
                    candidate=G.copy()
                    candidate[t]=max(0.,candidate[t]+sign*step)
                    v,_,_,_=evaluator.evaluate(candidate,model)
                    used+=1
                    evaluations+=1
                    attempted+=1
                    if v<candidate_best[0]-cfg.objective_tolerance:
                        candidate_best=(v,candidate)
                if used>=allocation[si]:
                    break
            improved=candidate_best[0]<value-cfg.objective_tolerance
            rounds.append(dict(step_kwh=step,attempts=attempted,accepted=improved,
                               complete_sweep=attempted==288))
            if improved:
                value,G=candidate_best
            elif attempted==288:
                ki+=1
            else:
                break
        converged=ki==len(cfg.steps)
        logs.append(dict(start=names[si],initial_objective=initial,final_objective=value,
                         rounds=rounds,evaluations=used,
                         stop='all_scales_full_sweep_no_improvement' if converged else 'budget_exhausted'))
        if best is None or value<best[0]-cfg.objective_tolerance:
            best=(value,G.copy(),si)
    value,G,si=best
    value,item,q,Q=evaluator.evaluate(G,model)
    ref=item['reference']
    check=rollout(G,ref,E0,iter(forecast['path']),price,cfg)
    G.flags.writeable=False
    return dict(G=G,reference=ref,objective=value,worst_q=q,Q=Q,
                log=dict(model=model,date=forecast['meta']['date'],E0=E0,nu=evaluator.nu,
                         evaluations=evaluations,lp_count=evaluator.lp_count,start=names[si],starts=logs,
                         elapsed_seconds=perf_counter()-started,best_objective=value,
                         stop_reason=('local_search_stopped' if all(x['stop'].startswith('all_scales') for x in logs)
                                      else 'budget_limited_feasible_approximation'),
                         global_optimum_claim=False,lp=item['lp'],residuals=check['residuals'],
                         config_hash=cfg.version,forecast_version=forecast['meta']['version'],
                         scenario_version=scenarios['version'],cache_prefix=evaluator.prefix,
                         algorithm=cfg.optimizer_version,seed_solver=seed_log,
                         fallback='' if len(scenarios['paths']) or model=='M0' else 'M0_no_scenarios'))
