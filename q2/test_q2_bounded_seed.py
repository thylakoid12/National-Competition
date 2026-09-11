import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from q2_config import Config,interval_labels
from q2_data import load_data
from q2_seed_bounded import deterministic_seed
from q2_controller import solve_reference,rollout

class BoundedSeedTests(unittest.TestCase):
    def test_bounded_seed_returns_checked_plan_or_explicit_failure(self):
        d=load_data()
        cfg=replace(Config(),bounded_seed=True,seed_time_limit_seconds=5.)
        f=d.values[21]
        frame=pd.DataFrame(dict(time=interval_labels(),load=f[:,0],pv=f[:,1],price=d.price))
        G,log=deterministic_seed(frame,cfg,6000)
        self.assertTrue(log['stages'])
        self.assertTrue(all(r['node_limit']==16 for r in log['stages']))
        if G is None:
            self.assertEqual(log['status'],'optional_q1_seed_failed')
        else:
            self.assertEqual(G.shape,(144,))
            self.assertTrue((G>=0).all())
            self.assertTrue(any(r['incumbent_checked'] for r in log['stages']))
            ref,_=solve_reference(G,f,6000,d.price,cfg.terminal('2025-01-22',d.price),cfg)
            out=rollout(G,ref,6000,iter(f),d.price,cfg)
            self.assertLessEqual(max(out['residuals'].values()),1e-6)

if __name__=='__main__':
    unittest.main()
