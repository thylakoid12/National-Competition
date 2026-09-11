"""Run with python -m unittest -v test_q2; no additional test dependency."""
from dataclasses import replace
import unittest
import numpy as np
from q2_config import Config
from q2_controller import solve_reference,rollout,batch_costs
from q2_forecast import Forecaster,features,combine
from q2_scenarios import compute_weights,build_scenarios
from q2_risk import worst_case_tv,worst_case_tv_lp,cvar
from q2_optimizer import Evaluator,optimize_plan
from q2_data import load_data

class Q2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=replace(Config(),budget=30,ml_min_days=10000)
        cls.data=load_data()

    def test_data_units_and_endpoints(self):
        d=self.data
        self.assertEqual(d.values.shape,(365,144,2))
        self.assertEqual(len(d.dates[31:]),334)
        self.assertEqual(d.interval_start[60],'10:00')
        self.assertEqual(d.interval_end[60],'10:10')
        self.assertEqual(d.interval_end[-1],'24:00')
        self.assertAlmostEqual(d.values[0,0,0],3529.7296/6)
        self.assertFalse(d.history(3).flags.writeable)

    def test_controller_physics_prefix_cash_and_batch(self):
        cfg=self.cfg
        rng=np.random.default_rng(2)
        forecast=self.data.values[1].copy()
        G=np.maximum(forecast[:,0]-forecast[:,1],0)*.8
        nu=cfg.terminal('2025-01-02',self.data.price)
        ref,log=solve_reference(G,forecast,6000,self.data.price,nu,cfg)
        ref2,_=solve_reference(G,forecast,6000,self.data.price,nu,cfg)
        np.testing.assert_array_equal(ref,ref2)
        paths=np.maximum(forecast[None]+rng.normal(0,200,(5,144,2)),0)
        for path,q in zip(paths,batch_costs(G,ref,6000,paths,self.data.price,nu,cfg)):
            o=rollout(G,ref,6000,iter(path),self.data.price,cfg)
            self.assertAlmostEqual(q,o['emergency_cost']-nu*(o['E'][-1]-cfg.e_min),places=6)
            changed=path.copy()
            changed[73:]*=10
            other=rollout(G,ref,6000,iter(changed),self.data.price,cfg)
            for key in ['charge','discharge','H','U','W','E']:
                np.testing.assert_array_equal(o[key][:73],other[key][:73])
            np.testing.assert_array_equal(G,o['G'])
            self.assertAlmostEqual(o['cash_cost'],self.data.price@G+5*self.data.price@o['H'])
        self.assertLess(log['residual'],1e-6)

    def test_tv_oracle_monotonicity_and_zero_mass(self):
        rng=np.random.default_rng(17)
        for n in [1,2,5,17]:
            for _ in range(6):
                Q=rng.normal(0,100,n)
                w=rng.dirichlet(np.ones(n))
                if n>1:
                    w[0]=0
                    w/=w.sum()
                last=float(Q@w)
                for rho in [0,.02,.05,.3,1]:
                    v,q=worst_case_tv(Q,w,rho)
                    lp,_=worst_case_tv_lp(Q,w,rho)
                    self.assertAlmostEqual(v,lp,places=6)
                    self.assertGreaterEqual(v+1e-8,last)
                    self.assertLessEqual(.5*np.abs(q-w).sum(),rho+1e-8)
                    self.assertAlmostEqual(q.sum(),1)
                    self.assertTrue((q>=0).all())
                    last=v

    def prepare(self,history):
        engine=Forecaster(self.cfg)
        for i in range(1,len(history)):
            issued=engine.issue(history[:i],self.data.dates[i])
            engine.settle(issued,history[i])
        current=engine.issue(history,self.data.dates[len(history)])
        f=combine(current,[0,0],'test')
        s=build_scenarios(f,engine.records,[0,0],self.cfg)
        return engine,current,f,s

    def test_no_future_leakage_and_cache_keys(self):
        i=9
        future=self.data.values.copy()
        future[i:]*=30
        a=self.prepare(self.data.history(i))
        b=self.prepare(future[:i].copy())
        for key in ['path','context']:
            np.testing.assert_array_equal(a[2][key],b[2][key])
        for key in ['paths','weights','clipping']:
            np.testing.assert_array_equal(a[3][key],b[3][key])
        G=[]
        for prepared in (a,b):
            G.append(optimize_plan('M3',prepared[2],prepared[3],6000,self.data.price,self.cfg)['G'])
        np.testing.assert_array_equal(*G)
        eva=Evaluator(a[2],a[3],6000,self.data.price,self.cfg)
        self.assertNotEqual(eva.key(G[0]),Evaluator(a[2],a[3],6001,self.data.price,self.cfg).key(G[0]))
        changed=G[0].copy(); changed[-1]+=1e-9
        self.assertNotEqual(eva.key(G[0]),eva.key(changed))
        self.assertTrue(all(date<self.data.dates[i] for date in a[3]['dates']))
        for sources in a[3]['sources']:
            self.assertEqual(sources[0]['date'],sources[1]['date'])

    def test_degeneracy_and_weight_order(self):
        _,_,f,s=self.prepare(self.data.history(8))
        G=np.maximum(f['path'][:,0]-f['path'][:,1],0)
        equal=dict(s,weights=np.ones(len(s['paths']))/len(s['paths']))
        e=Evaluator(f,equal,6000,self.data.price,replace(self.cfg,radius=0))
        values=[e.evaluate(G,m)[0] for m in ['M1','M2','M3']]
        np.testing.assert_allclose(values,[values[0]]*3,atol=1e-7)
        single=dict(s,paths=f['path'][None],weights=np.ones(1),version='single')
        e=Evaluator(f,single,6000,self.data.price,self.cfg)
        values=[e.evaluate(G,m)[0] for m in ['M0','M1','M2','M3']]
        np.testing.assert_allclose(values,[values[0]]*4,atol=1e-7)
        z=np.array([r['context'] for r in self.prepare(self.data.history(8))[0].records])
        w,_=compute_weights(f['context'],z,shrinkage=1)
        np.testing.assert_allclose(w,np.ones(len(z))/len(z))
        w,_=compute_weights(f['context'],z)
        reversed_w,_=compute_weights(f['context'],z[::-1])
        np.testing.assert_allclose(w,reversed_w[::-1])

    def test_fractional_cvar_and_training_features(self):
        x=np.arange(334,dtype=float)
        expected=(x[-16:].sum()+.7*x[-17])/16.7
        self.assertAlmostEqual(cvar(x),expected)
        h=self.data.history(7)
        X=features(h,'2025-01-08')
        np.testing.assert_array_equal(X[:,10],h[-1,:,0])
        np.testing.assert_array_equal(X[:,12],h[-7,:,0])
        self.assertTrue(np.isnan(features(h[:0],'2025-01-01')[:,4:]).any())
        self.assertEqual(Config().terminal('2025-12-31',self.data.price),0)

if __name__=='__main__':
    unittest.main()
