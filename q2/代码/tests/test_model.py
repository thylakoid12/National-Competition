"""重构回归：保存的旧解、全年独立回放、因果预测与断点连续性。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from microgrid.config import RESULT_DIR, CURRENT_RESULT, load_config
from microgrid.storage import read_arrays, read_json, write_arrays, write_json
from microgrid.forecast import Forecaster
from microgrid.scenarios import inputs
from microgrid.optimizer import optimize
from microgrid.controller import rollout
from microgrid.seed import seed_plan
from microgrid.experiment import run_variant, select_january, totals, prepare_bank
from microgrid.plotting import make_figures
from tests.oracle import scalar_replay

FIXTURES = RESULT_DIR / "refactor_20260911"


class ModelRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (CURRENT_RESULT / "evaluation_bank.npz").is_file():
            raise unittest.SkipTest("缺少历史 A/B 年度结果；新统计风险主线由 test_statistical_risk 验证")
        cls.cfg = load_config()
        cls.bank = read_arrays(CURRENT_RESULT / "evaluation_bank.npz")

    def test_old_optimizer_fixtures(self):
        if not list(FIXTURES.glob("fixture_*.npz")):
            self.skipTest("缺少历史优化器固定样本")
        for path in sorted(FIXTURES.glob("fixture_*.npz")):
            with self.subTest(fixture=path.name):
                z = read_arrays(path)
                forecast = dict(
                    path=z["forecast"],
                    context=z["context"],
                    meta={"date": str(z["date"])},
                )
                scenes = dict(
                    paths=z["paths"],
                    weights=z["weights"],
                    dates=z["scene_dates"].tolist(),
                )
                out = optimize(
                    forecast,
                    scenes,
                    float(z["E0"]),
                    z["price"],
                    z["initial_G"],
                    self.cfg,
                    "search_only" not in path.name,
                )
                for key in ("G", "reference", "Q", "worst_q", "objective", "alpha"):
                    np.testing.assert_allclose(out[key], z[key], atol=1e-7, rtol=0)
                print("optimizer matched:", path.name, flush=True)

    def test_seed_solver_injection(self):
        if not list(FIXTURES.glob("fixture_search_only*.npz")):
            self.skipTest("缺少历史种子求解固定样本")
        z = read_arrays(next(FIXTURES.glob("fixture_search_only*.npz")))
        out, log = seed_plan(
            {"path": z["forecast"]}, float(z["E0"]), z["price"], self.cfg
        )
        np.testing.assert_allclose(out, z["initial_G"], atol=1e-6, rtol=0)
        self.assertTrue(log["status"].startswith(("bounded", "checked")))

    def test_annual_settlement_and_oracle(self):
        rows = []
        previous = read_json(CURRENT_RESULT / "initial_states.json")[
            "evaluation_initial"
        ]
        directory = CURRENT_RESULT / "evaluation/joint_reserve"
        for path in sorted(directory.glob("2025-??-??.npz")):
            old = read_arrays(path)
            row = read_json(path.with_suffix(".json"))
            self.assertAlmostEqual(previous, row["E0"], places=6)
            actual = np.column_stack([old["load"], old["pv"]])
            out = rollout(
                old["G"],
                old["reference"],
                row["E0"],
                iter(actual),
                self.bank["price"],
                self.cfg,
            )
            oracle = scalar_replay(
                old["G"],
                old["reference"],
                row["E0"],
                actual,
                self.bank["price"],
                self.cfg,
            )
            for key in ("G", "H", "charge", "discharge", "E", "U", "W"):
                np.testing.assert_allclose(out[key], old[key], atol=1e-7, rtol=0)
            np.testing.assert_allclose(out["E"], oracle["E"], atol=1e-7, rtol=0)
            self.assertAlmostEqual(out["cash_cost"], row["cash_cost"], places=6)
            self.assertAlmostEqual(out["cash_cost"], oracle["cash_cost"], places=6)
            rows.append(row)
            previous = row["Eend"]
        self.assertEqual(len(rows), 334)
        self.assertAlmostEqual(totals(rows)["cash_cost"], 14003414.740197701, places=5)

    def test_forecast_freeze_and_causality(self):
        engine = Forecaster(self.cfg)
        for i in range(1, 365):
            date = str(self.bank["dates"][i])
            issue = engine.issue(self.bank["actual"][:i], date, use_ml=False)
            np.testing.assert_allclose(
                issue["f0"], self.bank["f0"][i], atol=1e-9, rtol=0
            )
            engine.settle(issue, self.bank["actual"][i], date)
        self.assertEqual(engine.report()["methods"], ["weekday_3", "mean_7"])
        altered = {k: v.copy() for k, v in self.bank.items()}
        altered["actual"][31:] += 1e6
        for variant in ("search_only", "joint_reserve", "adaptive_joint"):
            a, sa = inputs(self.bank, 31, variant, self.cfg)
            b, sb = inputs(altered, 31, variant, self.cfg)
            np.testing.assert_array_equal(a["path"], b["path"])
            np.testing.assert_array_equal(sa["paths"], sb["paths"])

    def test_ml_prediction_matches_archive(self):
        i = 14
        engine = Forecaster(self.cfg)
        prediction = engine.predict_ml(
            self.bank["actual"][:i], str(self.bank["dates"][i])
        )
        np.testing.assert_allclose(prediction, self.bank["f1"][i], atol=1e-7, rtol=0)

    def test_january_bank_archive_import(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as temp:
            bank = prepare_bank(
                temp, self.cfg, forecast_archive=RESULT_DIR / "formal_v2"
            )
            old = read_arrays(CURRENT_RESULT / "january_bank.npz")
            for key in old:
                np.testing.assert_array_equal(bank[key], old[key])

    def test_selection_uses_only_january(self):
        groups = {}
        for variant in ("search_only", "joint_reserve", "adaptive_joint"):
            groups[variant] = [
                read_json(p)
                for p in sorted(
                    (CURRENT_RESULT / "january" / variant).glob("2025-??-??.json")
                )
            ]
        self.assertEqual(select_january(groups), "joint_reserve")
        groups["joint_reserve"][-1]["date"] = "2025-02-01"
        with self.assertRaises((ValueError, AssertionError)):
            select_january(groups)

    def test_resume_skips_complete_and_rejects_broken_state(self):
        import shutil

        with tempfile.TemporaryDirectory(dir=FIXTURES) as temp:
            dest = Path(temp)
            shutil.copy(CURRENT_RESULT / "january_bank.npz", dest)
            shutil.copy(CURRENT_RESULT / "initial_states.json", dest)
            shutil.copytree(
                CURRENT_RESULT / "january/joint_reserve", dest / "january/joint_reserve"
            )
            with patch(
                "microgrid.experiment.optimize",
                side_effect=AssertionError("should resume"),
            ):
                rows = run_variant(dest, "joint_reserve", "january", self.cfg)
            self.assertEqual(len(rows), 17)
            first = dest / "january/joint_reserve/2025-01-15.json"
            row = read_json(first)
            row["E0"] += 1
            write_json(first, row)
            with self.assertRaises(ValueError):
                run_variant(dest, "joint_reserve", "january", self.cfg)

    def test_resume_computes_missing_last_day(self):
        import shutil

        with tempfile.TemporaryDirectory(dir=FIXTURES) as temp:
            dest = Path(temp)
            for name in (
                "evaluation_bank.npz",
                "january_bank.npz",
                "initial_states.json",
                "selection_frozen.json",
            ):
                shutil.copy(CURRENT_RESULT / name, dest)
            shutil.copytree(
                CURRENT_RESULT / "evaluation/joint_reserve",
                dest / "evaluation/joint_reserve",
            )
            last = dest / "evaluation/joint_reserve/2025-12-31"
            last.with_suffix(".json").unlink()
            last.with_suffix(".npz").unlink()
            rows = run_variant(dest, "joint_reserve", "evaluation", self.cfg)
            self.assertEqual(len(rows), 334)
            old = read_json(CURRENT_RESULT / "evaluation/joint_reserve/2025-12-31.json")
            self.assertAlmostEqual(rows[-1]["cash_cost"], old["cash_cost"], places=5)
            self.assertAlmostEqual(rows[-1]["Eend"], old["Eend"], places=5)

    def test_plots_are_three_separate_figures(self):
        import matplotlib.pyplot as plt

        day = read_arrays(CURRENT_RESULT / "evaluation/joint_reserve/2025-03-20.npz")
        figures = make_figures(day, "2025-03-20", self.cfg)
        self.assertEqual(len(figures), 3)
        self.assertEqual(len(set(id(f) for f in figures.values())), 3)
        for fig in figures.values():
            self.assertEqual(len(fig.axes), 1)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
