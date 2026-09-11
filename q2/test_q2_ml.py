"""Actual CatBoost causality check, never count a fallback as an ML success."""
import unittest
from dataclasses import replace
import numpy as np
from q2_config import Config
from q2_data import load_data
from q2_forecast import Forecaster

class MLTests(unittest.TestCase):
    def test_actual_ml_day_ahead_invariance(self):
        from catboost import CatBoostRegressor
        data=load_data()
        future=data.values.copy()
        future[14:]*=100
        cfg=replace(Config(),ml_iterations=20)
        first=Forecaster(cfg).issue(data.history(14),'2025-01-15')['issued'][1]
        second=Forecaster(cfg).issue(future[:14].copy(),'2025-01-15')['issued'][1]
        self.assertEqual(first['meta']['fallback'],[False,False])
        self.assertEqual(first['meta']['training_rows'],2016)
        self.assertEqual(first['meta']['train_end'],'2025-01-14')
        np.testing.assert_array_equal(first['path'],second['path'])

if __name__=='__main__':
    unittest.main()
