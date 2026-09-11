"""Whole-day TV ambiguity, with a small LP as correctness oracle."""
import numpy as np
from scipy.optimize import linprog

def check(Q,w,rho):
    Q,w=np.asarray(Q,float),np.asarray(w,float)
    if Q.ndim!=1 or not len(Q) or Q.shape!=w.shape or not np.isfinite(Q).all():
        raise ValueError('Invalid cost vector')
    if not np.isfinite(w).all() or (w<0).any() or abs(w.sum()-1)>1e-8 or not 0<=rho<=1:
        raise ValueError('Invalid nominal probabilities or radius')
    return Q,w

def worst_case_tv_lp(Q,w,rho):
    Q,w=check(Q,w,rho)
    n=len(Q)
    A=np.vstack([np.c_[np.eye(n),-np.eye(n)],np.c_[-np.eye(n),-np.eye(n)],
                 np.r_[np.zeros(n),np.ones(n)][None]])
    r=linprog(np.r_[-Q,np.zeros(n)],A_ub=A,b_ub=np.r_[w,-w,2*rho],
              A_eq=np.r_[np.ones(n),np.zeros(n)][None],b_eq=[1],bounds=(0,None),method='highs-ds')
    if not r.success:
        raise RuntimeError(r.message)
    return float(Q@r.x[:n]),r.x[:n]

def worst_case_tv(Q,w,rho):
    Q,w=check(Q,w,rho)
    q=w.copy()
    order=np.argsort(Q,kind='stable')
    receiver=order[-1]
    remaining=min(rho,1-q[receiver])
    for donor in order:
        if remaining<=0 or Q[donor]>=Q[receiver]:
            break
        move=min(remaining,q[donor])
        q[donor]-=move
        q[receiver]+=move
        remaining-=move
    assert .5*np.abs(q-w).sum()<=rho+1e-8
    return float(Q@q),q

def aggregate(model,Q,w,rho):
    if model in ['M0','M1']:
        return float(np.mean(Q)),np.ones(len(Q))/len(Q)
    if model=='M2':
        return float(np.dot(w,Q)),np.asarray(w)
    if model=='M3':
        return worst_case_tv(Q,w,rho)
    raise ValueError(model)

def cvar(values,alpha=.95):
    z=np.sort(np.asarray(values,float))[::-1]
    if not len(z) or not 0<=alpha<1:
        raise ValueError('Invalid CVaR input')
    mass=(1-alpha)*len(z)
    whole=int(np.floor(mass))
    numerator=z[:whole].sum()
    if whole<len(z):
        numerator+=(mass-whole)*z[whole]
    return float(numerator/mass)
