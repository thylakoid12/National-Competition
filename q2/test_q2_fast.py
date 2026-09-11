from dataclasses import replace
from time import perf_counter
import unittest
import numpy as np
from q2_config import Config
from q2_data import load_data
from q2_controller import solve_reference,batch_costs
from q2_fast import FastReference,fast_batch

class AccelerationTests(unittest.TestCase):
    def test_native_lp_and_compiled_paths(self):
        d=load_data()
        cfg=Config()
        rng=np.random.default_rng(47)
        for day in [1,14,30]:
            f=d.values[day].copy()
            p=d.price
            nu=cfg.terminal(d.dates[day],p)
            native=FastReference(f,6000,p,nu,cfg)
            G=np.maximum(f[:,0]-f[:,1],0)*.8
            expected,log=solve_reference(G,f,6000,p,nu,cfg)
            actual,fastlog=native.solve(G)
            self.assertAlmostEqual(log['primary_objective'],fastlog['primary_objective'],places=5)
            self.assertAlmostEqual(expected.sum(),actual.sum(),places=3)
            native.solve(G+15)
            again,_=native.solve(G)
            np.testing.assert_array_equal(actual,again)
            paths=np.maximum(f[None]+rng.normal(0,200,(60,144,2)),0)
            python=batch_costs(G,actual,6000,paths,p,nu,cfg)
            compiled=fast_batch(G,actual,6000,paths,p,nu,cfg)
            np.testing.assert_allclose(python,compiled,rtol=0,atol=1e-7)

if __name__=='__main__':
    unittest.main()
