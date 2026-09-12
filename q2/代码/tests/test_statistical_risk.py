"""独立数学核验、物理回放、防泄漏和选型边界；无需旧年度结果。"""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog

from microgrid.config import Config
from microgrid.controller import rollout
from microgrid.data import Data
from microgrid.forecast import StatisticalForecaster
from microgrid.models import get_model
from microgrid.optimizer import RiskPolicyEvaluator, optimize_risk
from microgrid.risk import weighted_var_cvar, worst_case_cvar, worst_case_tv
from microgrid.scenarios import build_scenarios
from microgrid.solver import batch_metrics, fast_batch
from microgrid.statistical_experiment import build_bank, scenes_at, choose_model
from microgrid.stress import HELDOUT, DEVELOPMENT, apply_shock, fixed_plan_stress
from tests.oracle import scalar_replay


def synthetic_data(days=365):
    rng = np.random.default_rng(713)
    t = np.arange(144)
    paths = []
    for d in range(days):
        load = 650+120*np.sin(t/144*2*np.pi)+35*(d % 7)+rng.normal(0, 10, 144)
        pv = np.maximum(0, np.sin((t-36)/72*np.pi))*400
        pv[(t < 36) | (t > 108)] = 0
        paths.append(np.column_stack([load, pv*(0.7+0.3*rng.random())]))
    dates = tuple((date(2025,1,1)+timedelta(days=d)).isoformat() for d in range(days))
    price = np.where((t >= 102) & (t < 126), 1.3, 0.5)
    return Data(dates, np.array(paths), price)


def cvar_lp(values, weights, rho, beta):
    """独立概率-尾部密度 LP，而非被测排序算法。"""
    n = len(values)
    A, b = [], []
    for i in range(n):
        row = np.zeros(3*n); row[i] = -1; row[n+i] = 1-beta
        A.append(row); b.append(0)
        row = np.zeros(3*n); row[i] = 1; row[2*n+i] = -1
        A.append(row); b.append(weights[i])
        row = np.zeros(3*n); row[i] = -1; row[2*n+i] = -1
        A.append(row); b.append(-weights[i])
    row = np.zeros(3*n); row[2*n:] = 1
    A.append(row); b.append(2*rho)
    eq = np.zeros((2,3*n)); eq[0,:n] = 1; eq[1,n:2*n] = 1
    result = linprog(np.r_[np.zeros(n), -np.asarray(values), np.zeros(n)],
        A_ub=np.array(A), b_ub=np.array(b), A_eq=eq, b_eq=np.ones(2), bounds=(0,None), method="highs")
    if not result.success:
        raise AssertionError(result.message)
    return -result.fun


class RiskMathematics(unittest.TestCase):
    def test_discrete_atom_and_zero_weights(self):
        self.assertEqual(weighted_var_cvar([1,2,10], [.5,.45,.05], .9)[0], 2)
        self.assertAlmostEqual(weighted_var_cvar([1,2,10], [.5,.45,.05], .9)[1], 6)
        self.assertAlmostEqual(weighted_var_cvar([1,2,10], [.5,.45,.05], 0)[1], 1.9)
        self.assertEqual(weighted_var_cvar([-100,4,20], [0,1,0], .95), (4,4))

    def test_robust_cvar_against_independent_lp(self):
        rng = np.random.default_rng(947)
        for n in (1,2,3,8,15):
            for _ in range(8):
                values = rng.normal(20, 70, n).round(1)
                weights = rng.dirichlet(np.ones(n))
                rho, beta = float(rng.choice([0,.01,.05,.2,1])), float(rng.choice([0,.5,.9,.99]))
                got, q = worst_case_cvar(values, weights, rho, beta)
                self.assertAlmostEqual(got, cvar_lp(values, weights, rho, beta), places=6)
                self.assertLessEqual(.5*np.abs(q-weights).sum(), rho+1e-8)

    def test_probability_perturbation_does_not_extend_support(self):
        self.assertAlmostEqual(worst_case_cvar([2,10], [.99,.01], 1, .9)[0], 10)
        self.assertAlmostEqual(worst_case_tv([2,10], [.99,.01], 0)[0], 2.08)

    def test_invalid_values_fail(self):
        for values, weights, beta in [([1], [1], 1), ([np.nan],[1],.9), ([1,2],[.2,.3],.9)]:
            with self.assertRaises(ValueError):
                weighted_var_cvar(values,weights,beta)


class PhysicsAndCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_data(40)
        cls.cfg = replace(Config(), search_evaluations=180, search_starts=3, bootstrap_repetitions=0)
        cls.bank, _ = build_bank(cls.data, cls.cfg, 40)

    def test_batch_matches_scalar_oracle_with_asymmetric_efficiencies(self):
        cfg = replace(self.cfg, eta_c=.85, eta_d=.93)
        G = np.linspace(200,800,144)
        R = np.linspace(6000,1200,145)
        paths = self.data.values[30:33]
        got = batch_metrics(G,R,6000,paths,self.data.price,cfg)
        Q = fast_batch(G,R,6000,paths,self.data.price,.4,cfg)
        for i,path in enumerate(paths):
            oracle = scalar_replay(G,R,6000,path,self.data.price,cfg)
            real = rollout(G,R,6000,iter(path),self.data.price,cfg)
            np.testing.assert_allclose(real['E'],oracle['E'],atol=1e-8)
            self.assertAlmostEqual(got['emergency_kwh'][i],real['H'].sum(),places=7)
            self.assertAlmostEqual(got['emergency_cost'][i],real['emergency_cost'],places=7)
            self.assertAlmostEqual(got['end_energy'][i],real['E'][-1],places=7)
            self.assertAlmostEqual(got['min_energy'][i],real['E'].min(),places=7)
            self.assertAlmostEqual(got['unused'][i],real['U'].sum(),places=7)
            self.assertAlmostEqual(got['curtailment'][i],real['W'].sum(),places=7)
            self.assertAlmostEqual(Q[i],got['emergency_cost'][i]-.4*(got['end_energy'][i]-cfg.e_min),places=7)

    def test_forecasts_and_scenarios_ignore_future(self):
        changed = self.data.values.copy()
        changed[31:] = 1e6
        bank, _ = build_bank(Data(self.data.dates,changed,self.data.price),self.cfg,40)
        np.testing.assert_allclose(bank['f0'][:32],self.bank['f0'][:32],atol=1e-9)
        for mode in ('point','uniform','conditional'):
            f,a = scenes_at(self.bank,31,mode,self.cfg)
            _,b = scenes_at(bank,31,mode,self.cfg)
            np.testing.assert_allclose(a['paths'],b['paths'],atol=1e-9)
            np.testing.assert_allclose(a['weights'],b['weights'],atol=1e-12)
            self.assertLess(f['meta']['train_end'],f['meta']['date'])
        self.assertTrue((self.bank['f0'][:, :30, 1] == 0).all())

    def test_reoptimized_plans_ignore_future(self):
        altered = dict(self.bank)
        altered['actual'] = self.bank['actual'].copy()
        altered['actual'][31:] = np.nan
        plans=[]
        for bank in (self.bank,altered):
            f,s=scenes_at(bank,31,'conditional',self.cfg)
            G=np.maximum(f['path'][:,0]-f['path'][:,1],0)
            plans.append(optimize_risk(f,s,6000,self.data.price,G,self.cfg,'M3-RS'))
        for key in ('G','reference','alpha','objective'):
            np.testing.assert_allclose(plans[0][key],plans[1][key],atol=1e-8)

    def test_intraday_feedback_cannot_see_later_shock(self):
        G=np.full(144,500.0); R=np.linspace(6000,1200,145)
        path=self.data.values[31].copy(); changed=path.copy(); changed[72:,0]*=3
        a=rollout(G,R,6000,iter(path),self.data.price,self.cfg)
        b=rollout(G,R,6000,iter(changed),self.data.price,self.cfg)
        np.testing.assert_array_equal(a['G'],b['G'])
        np.testing.assert_array_equal(a['H'][:72],b['H'][:72])
        np.testing.assert_array_equal(a['E'][:73],b['E'][:73])

    def test_model_reductions_share_controller(self):
        cfg=replace(self.cfg,radius=0,shrinkage=1)
        f,s1=scenes_at(self.bank,31,'uniform',cfg)
        _,s2=scenes_at(self.bank,31,'conditional',cfg)
        np.testing.assert_allclose(s1['weights'],s2['weights'],atol=1e-14)
        G=np.maximum(f['path'][:,0]-f['path'][:,1],0)
        results=[RiskPolicyEvaluator(f,s,6000,self.data.price,cfg,m).evaluate(G,.5)
                 for s,m in ((s1,'M1'),(s2,'M2'),(s2,'M3'))]
        for other in results[1:]:
            self.assertAlmostEqual(results[0]['objective'],other['objective'],places=8)
            np.testing.assert_array_equal(results[0]['reference'],other['reference'])

    def test_multistart_budget_and_feasible_repair(self):
        f,s=scenes_at(self.bank,31,'conditional',self.cfg)
        G=np.maximum(f['path'][:,0]-f['path'][:,1],0)
        result=optimize_risk(f,s,6000,self.data.price,G,self.cfg,'M3-RS')
        self.assertEqual(len(result['log']['starts']),3)
        self.assertLessEqual(result['log']['evaluations'],self.cfg.search_evaluations)
        self.assertTrue(result['feasible'])
        self.assertFalse(result['log']['global_optimum_claim'])
        self.assertLessEqual(result['risk_cvar'],result['risk_budget']+1e-6)
        self.assertLessEqual(result['stress_cost'],result['stress_budget']+1e-6)

    def test_risk_budget_is_independent_of_candidate_purchase(self):
        f,s=scenes_at(self.bank,31,'conditional',self.cfg)
        ev=RiskPolicyEvaluator(f,s,6000,self.data.price,self.cfg,'M3-RS')
        a=ev.evaluate(np.zeros(144),1)
        b=ev.evaluate(np.full(144,2000.0),1)
        self.assertEqual(a['risk_budget'],b['risk_budget'])
        self.assertEqual(a['stress_budget'],b['stress_budget'])
        self.assertFalse(a['feasible'])
        self.assertTrue(b['feasible'])

    def test_stress_is_disjoint_and_plan_stays_locked(self):
        self.assertFalse(set(x.name for x in HELDOUT)&set(x.name for x in DEVELOPMENT))
        G=np.full(144,500.0); saved=G.copy(); R=np.linspace(6000,1200,145)
        path=self.data.values[31].copy(); original=path.copy()
        with patch('microgrid.optimizer.optimize_risk',side_effect=AssertionError('Replanning forbidden')):
            result=fixed_plan_stress(G,R,6000,path,self.data.price,self.cfg)
        self.assertEqual(set(result),set(x.name for x in DEVELOPMENT))
        np.testing.assert_array_equal(G,saved)
        np.testing.assert_array_equal(path,original)
        shocked=apply_shock(path,HELDOUT[1])
        self.assertTrue((shocked[path[:,1]==0,1]==0).all())


class SelectionTests(unittest.TestCase):
    def test_three_percent_cap_and_risk_first(self):
        def row(cost,tail,stress,fail=0):
            return dict(cash_cost=cost,emergency_cvar=tail,development_stress_cvar=stress,
                        risk_budget_unmet_days=fail,max_physical_residual=0)
        summaries={'M2':row(100,10,20),'M3':row(102,8,16),'M3-R':row(104,1,1),
                   'M3-RS':row(101,2,2,1)}
        choice=choose_model(summaries,Config())
        self.assertEqual(choice['selected'],'M3')
        self.assertAlmostEqual(choice['validation_cash_cap'],103)
        self.assertFalse(choice['comparison']['M3-R']['cost_eligible'])
        self.assertFalse(choice['comparison']['M3-RS']['feasible'])

    def test_no_eligible_plan_is_not_silently_approved(self):
        summary=dict(cash_cost=100,emergency_cvar=0,development_stress_cvar=0,
                     risk_budget_unmet_days=1,max_physical_residual=0)
        choice=choose_model({'M2':summary},Config())
        self.assertIsNone(choice['selected'])
        self.assertEqual(choice['status'],'no_eligible_candidate')


class PipelineTests(unittest.TestCase):
    def test_fresh_warmup_checkpoint_freeze_and_evaluation(self):
        from microgrid.statistical_experiment import prepare,run_candidate,calibrate,evaluate,setup
        from microgrid.storage import read_json,read_arrays
        data=synthetic_data()
        cfg=replace(Config(),search_evaluations=90,bootstrap_repetitions=0)
        def cheap_seed(f,E0,price,cfg):
            return np.maximum(f['path'][:,0]-f['path'][:,1],0),dict(status='test_seed')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); attachments=root/'attachments'; attachments.mkdir()
            for name in ('附件1.xlsx','附件2.xlsx'):
                (attachments/name).write_bytes(b'fixture only, loader mocked')
            out=root/'run'
            with patch('microgrid.statistical_experiment.load_data',return_value=data), patch(
                    'microgrid.statistical_experiment.seed_plan',side_effect=cheap_seed):
                prepare(out,cfg,attachments)
                states=read_json(out/'initial_states.json')
                self.assertAlmostEqual(states['validation_initial'],float(read_arrays(out/'warmup/2025-01-14.npz')['E'][-1]))
                self.assertAlmostEqual(states['evaluation_initial'],float(read_arrays(out/'warmup/2025-01-31.npz')['E'][-1]))
                self.assertTrue((read_arrays(out/'warmup/2025-01-01.npz')['G']==0).all())
                first=run_candidate(out,'M2','validation',2)
                self.assertFalse(first['complete'])
                with patch('microgrid.statistical_experiment.optimize_risk',side_effect=AssertionError('Should resume')):
                    resumed=run_candidate(out,'M2','validation',2)
                self.assertEqual(first['cash_cost'],resumed['cash_cost'])
                with self.assertRaises(ValueError):
                    run_candidate(out,'M2','validation',1)
                choice=calibrate(out,['M2','M3-RS'])
                self.assertFalse(choice['february_december_outcomes_used'])
                self.assertFalse(choice['heldout_stress_used'])
                self.assertEqual(choice['selection_days'],17)
                after=evaluate(out,['M2'],max_days=2)
                self.assertFalse(after['complete'])
                with self.assertRaises(ValueError):
                    setup(out,replace(cfg,cost_premium=.05),attachments)
                target=out/'evaluation/M2/2025-02-01.npz'
                target.write_bytes(target.read_bytes()+b'edited')
                with self.assertRaises(ValueError):
                    run_candidate(out,'M2','evaluation',2)

    def test_multiday_stress_updates_from_revealed_history_only(self):
        from microgrid.statistical_experiment import build_bank
        from microgrid.stress_testing import simulate_episode
        from microgrid.stress import Shock
        from microgrid.storage import read_arrays
        data=synthetic_data(40)
        cfg=replace(Config(),search_evaluations=90,bootstrap_repetitions=0)
        bank,_=build_bank(data,cfg,40)
        def cheap_seed(f,E0,price,cfg):
            return np.maximum(f['path'][:,0]-f['path'][:,1],0),{}
        with tempfile.TemporaryDirectory() as temp, patch('microgrid.stress_testing.seed_plan',side_effect=cheap_seed):
            normal=Path(temp)/'normal'; stressed=Path(temp)/'stressed'; cache={}
            a=simulate_episode(bank,31,6000,Shock('normal',duration=3),cfg,'M3-RS',normal,cache)
            b=simulate_episode(bank,31,6000,Shock('compound',1.2,.5,duration=2),cfg,'M3-RS',stressed,cache,recovery_days=1)
            np.testing.assert_array_equal(read_arrays(normal/'locked_plans/2025-02-01.npz')['G'],
                                          read_arrays(stressed/'locked_plans/2025-02-01.npz')['G'])
            self.assertEqual(b[1]['E0'],b[0]['Eend'])
            self.assertEqual(b[2]['E0'],b[1]['Eend'])
            self.assertEqual(b[2]['phase'],'recovery')
            self.assertGreater(b[0]['cash_cost'],a[0]['cash_cost'])


if __name__=='__main__':
    unittest.main(verbosity=2)
