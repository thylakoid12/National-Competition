"""Archive, export semantics, historical feature and branch-isolation tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
from datetime import date,timedelta
import copy
import unittest
import numpy as np
from q2_archive import save_npz,save_json
from q2_config import Config
from q2_data import load_data
from q2_forecast import Forecaster,features
from q2_export import emergency_intervals
from q2_metrics import distribution_metrics

class IntegrationTests(unittest.TestCase):
    def test_immutable_archive(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            save_npz(root/'x.npz',x=np.arange(3.))
            save_npz(root/'x.npz',x=np.arange(3.))
            with self.assertRaises(FileExistsError):
                save_npz(root/'x.npz',x=np.arange(3.)+1)
            save_json(root/'x.json',{'version':1})
            with self.assertRaises(FileExistsError):
                save_json(root/'x.json',{'version':2})

    def test_emergency_gaps_prices_and_midnight(self):
        H=np.zeros(144)
        H[60:62]=[10,20]
        H[63]=30
        H[-1]=40
        price=np.arange(144,dtype=float)+1
        rows=emergency_intervals(H,price)
        self.assertEqual([r[0] for r in rows],['10:00-10:20','10:30-10:40','23:50-24:00'])
        self.assertEqual(sum(r[1] for r in rows),H.sum())
        self.assertEqual(sum(r[2] for r in rows),5*price@H)

    def test_nominal_crps_matches_pairwise(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(13,8))
        w=rng.dirichlet(np.ones(13))
        obs=rng.normal(size=8)
        expected=(w[:,None]*np.abs(x-obs)).sum(axis=0)-.5*np.einsum('i,j,ijt->t',w,w,np.abs(x[:,None]-x[None]))
        self.assertAlmostEqual(distribution_metrics(x,obs,w)['crps'],expected.mean())

    def test_january_freeze_branch_isolation_and_retrain_no_rewrite(self):
        data=load_data()
        engine=Forecaster(replace(Config(),ml_min_days=10000))
        for i in range(1,31):
            issue=engine.issue(data.history(i),data.dates[i])
            engine.settle(issue,data.values[i])
        previous=engine.records[3]['issued'][0]['path'].copy()
        issue=engine.issue(data.history(31),'2025-02-01')
        frozen=copy.deepcopy(engine.frozen)
        composition=copy.deepcopy(engine.composition)
        branch=copy.deepcopy(engine)
        altered=data.values[31].copy()*2
        branch.settle(issue,altered)
        history=np.concatenate([data.history(31),altered[None]])
        branch.issue(history,'2025-02-02')
        self.assertEqual(engine.frozen,frozen)
        self.assertEqual(engine.composition,composition)
        self.assertEqual(len(engine.records),30)
        self.assertEqual(len(branch.records),31)
        np.testing.assert_array_equal(engine.records[3]['issued'][0]['path'],previous)
        self.assertEqual(branch.frozen,frozen)

    def test_training_each_date_uses_its_own_prefix(self):
        d=load_data()
        day=15
        for j in [0,1,7,14]:
            X=features(d.history(day)[:j],d.dates[j])
            modified=d.history(day).copy()
            modified[j:]*=100
            other=features(modified[:j],d.dates[j])
            np.testing.assert_array_equal(X,other)

if __name__=='__main__':
    unittest.main()
