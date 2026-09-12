"""B 模型防泄漏审计；仅在临时副本中修改实测数据。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from microgrid.config import CURRENT_RESULT, RESULT_DIR, load_config
from microgrid.data import Data, load_data
from microgrid.forecast import Forecaster
from microgrid.experiment import prepare_bank
from microgrid.scenarios import inputs
from microgrid.seed import seed_plan
from microgrid.optimizer import optimize
from microgrid.controller import rollout
from microgrid.storage import read_arrays, read_json, write_json

class BCausalityAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (CURRENT_RESULT / 'evaluation_bank.npz').is_file():
            raise unittest.SkipTest('缺少历史 B 年度结果；新统计风险主线由 test_statistical_risk 验证')
        cls.cfg = load_config()
        cls.bank = read_arrays(CURRENT_RESULT / 'evaluation_bank.npz')
        cls.days = CURRENT_RESULT / 'evaluation/joint_reserve'
        cls.scratch = RESULT_DIR / 'b_causality_audit_20260912'
        cls.scratch.mkdir(parents=True, exist_ok=True)

    def test_all_saved_forecasts_rebuild_from_past_only(self):
        raw = load_data()
        np.testing.assert_array_equal(raw.values, self.bank['actual'])
        engine = Forecaster(self.cfg)
        for i in range(1, 365):
            f = engine.issue(raw.values[:i], str(self.bank['dates'][i]), use_ml=False)
            for key, saved in [('f0', 'f0'), ('context', 'contexts')]:
                np.testing.assert_allclose(f[key], self.bank[saved][i], atol=1e-9, rtol=0)
            if i >= 31:
                independent = np.column_stack([raw.values[[i-7, i-14, i-21], :, 0].mean(axis=0), raw.values[i-7:i, :, 1].mean(axis=0)])
                np.testing.assert_allclose(f['f0'], independent, atol=1e-9, rtol=0)
            engine.settle(f, raw.values[i], str(self.bank['dates'][i]))
        self.assertEqual(engine.report()['methods'], ['weekday_3', 'mean_7'])

    def test_all_334_scenarios_ignore_current_and_future_actuals(self):
        for day in range(31, 365):
            changed = dict(self.bank)
            changed['actual'] = self.bank['actual'].copy()
            changed['actual'][day:] = np.nan
            a, sa = inputs(self.bank, day, 'joint_reserve', self.cfg)
            b, sb = inputs(changed, day, 'joint_reserve', self.cfg)
            for key in ('path', 'context'):
                np.testing.assert_array_equal(a[key], b[key])
            for key in ('paths', 'weights'):
                np.testing.assert_array_equal(sa[key], sb[key])
                self.assertTrue(np.isfinite(sb[key]).all())
            self.assertLess(max(sb['dates']), b['meta']['date'])
            self.assertLess(b['meta']['train_end'], b['meta']['date'])

    def test_rebuilt_predictions_and_reoptimized_plans_ignore_future(self):
        records = []
        for target in ('2025-02-01', '2025-06-21', '2025-12-31'):
            day = list(self.bank['dates']).index(target)
            altered = self.bank['actual'].copy()
            altered[day:, :, 0], altered[day:, :, 1] = 1e6, 0.0
            data = Data(tuple(self.bank['dates']), altered, self.bank['price'])
            with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
                write_json(Path(temp) / 'selection_frozen.json', {'selected': 'joint_reserve'})
                with patch('microgrid.experiment.load_data', return_value=data):
                    rebuilt = prepare_bank(temp, self.cfg, january=False)
            for key in ('f0', 'contexts'):
                np.testing.assert_allclose(rebuilt[key][1:day+1], self.bank[key][1:day+1], atol=1e-9, rtol=0)
            E0 = read_json(self.days / (target + '.json'))['E0']
            plans = []
            for bank in (self.bank, rebuilt):
                f, scenes = inputs(bank, day, 'joint_reserve', self.cfg)
                seed, _ = seed_plan(f, E0, bank['price'], self.cfg)
                plans.append(optimize(f, scenes, E0, bank['price'], seed, self.cfg, True))
            for key in ('G', 'reference', 'alpha', 'objective', 'Q', 'worst_q'):
                np.testing.assert_allclose(plans[0][key], plans[1][key], atol=1e-7, rtol=0)
            locked = read_arrays(self.days / 'locked_plans' / (target + '.npz'))
            for key in ('G', 'reference', 'alpha'):
                np.testing.assert_allclose(plans[0][key], locked[key], atol=1e-7, rtol=0)
            records.append({'date': target, 'max_plan_delta': float(np.max(np.abs(plans[0]['G']-plans[1]['G']))), 'saved_plan_matches': True})
            print('Rebuilt and reoptimized without target/future observations:', target, flush=True)
        write_json(self.scratch / 'perturbation_results.json', records)

    def test_locked_plans_and_intraday_no_future_access_all_334_days(self):
        previous = read_json(CURRENT_RESULT / 'initial_states.json')['evaluation_initial']
        count = 0
        for day in range(31, 365):
            date = str(self.bank['dates'][day])
            old = read_arrays(self.days / (date + '.npz'))
            row = read_json(self.days / (date + '.json'))
            locked = read_arrays(self.days / 'locked_plans' / (date + '.npz'))
            self.assertAlmostEqual(previous, row['E0'], places=6)
            for key in ('G', 'reference'):
                np.testing.assert_array_equal(locked[key], old[key])
            self.assertLess(row['forecast_train_end'], date)
            self.assertLess(row['scene_end'], date)
            actual = self.bank['actual'][day]
            base = rollout(locked['G'], locked['reference'], previous, iter(actual), self.bank['price'], self.cfg)
            changed = actual.copy()
            changed[72:, 0], changed[72:, 1] = 1e6, 0.0
            other = rollout(locked['G'], locked['reference'], previous, iter(changed), self.bank['price'], self.cfg)
            for key in ('G', 'H', 'charge', 'discharge', 'U', 'W'):
                np.testing.assert_allclose(base[key], old[key], atol=1e-7, rtol=0)
                np.testing.assert_array_equal(base[key][:72], other[key][:72])
            np.testing.assert_array_equal(base['G'], other['G'])
            np.testing.assert_array_equal(base['E'][:73], other['E'][:73])
            np.testing.assert_allclose(base['E'], old['E'], atol=1e-7, rtol=0)
            previous = row['Eend']
            count += 1
        self.assertEqual(count, 334)

    def test_initial_states_are_previous_january_day_end(self):
        states = read_json(CURRENT_RESULT / 'initial_states.json')
        warmup = RESULT_DIR / 'formal_v2/runs/warmup'
        for day, field in [('2025-01-14', 'validation_initial'), ('2025-01-31', 'evaluation_initial')]:
            old = read_arrays(warmup / (day + '.npz'))
            self.assertAlmostEqual(states[field], float(old['E'][-1]), places=8)
            row = read_json(warmup / (day + '.json'))
            replay = rollout(old['G'], old['reference'], row['plan']['E0'], iter(np.column_stack([old['load'], old['pv']])), self.bank['price'], self.cfg)
            np.testing.assert_allclose(replay['E'], old['E'], atol=1e-7, rtol=0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
