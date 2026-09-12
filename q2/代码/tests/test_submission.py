"""交表内容、时间范围与固定对照标识的回归测试。"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from microgrid.config import CURRENT_RESULT, RESULT_DIR
from microgrid.storage import write_json, evaluation_choice
from microgrid.submission import make_payload, intervals


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def test_contiguous_emergency_periods(self):
        h = np.zeros(144)
        h[0:2] = [1, 2]
        h[3] = 4
        h[143] = 5
        self.assertEqual(
            intervals(h), [["0:00-0:20", 3.0], ["0:30-0:40", 4.0], ["23:50-24:00", 5.0]]
        )

    def test_complete_official_and_paper_tables(self):
        if not (CURRENT_RESULT / "evaluation/joint_reserve/2025-12-31.npz").is_file():
            self.skipTest("缺少此项固定金额断言对应的历史 B 全年结果")
        with tempfile.TemporaryDirectory(dir=RESULT_DIR) as output:
            data = make_payload(CURRENT_RESULT, output)
            self.assertEqual(data["model"], "B")
            self.assertEqual(
                (len(data["purchase"]), len(data["purchase"][0])), (335, 147)
            )
            self.assertEqual(data["purchase"][0][1], "0:00-0:10")
            self.assertEqual(data["purchase"][0][144], "23:50-24:00")
            self.assertEqual(len(data["storage"]), 2005)
            self.assertEqual(len(data["paper"]), 4)
            self.assertAlmostEqual(
                sum(r[-1] for r in data["purchase"][1:]), 14003414.740197701, places=5
            )
            self.assertAlmostEqual(
                sum(r[2] for r in data["emergency"][1:]), 110179.63528014044, places=6
            )
            self.assertEqual(
                [r[0] for r in data["paper"]["表1_购电结果"][::6]],
                ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"],
            )

    def test_incomplete_run_is_not_exported(self):
        with tempfile.TemporaryDirectory(dir=RESULT_DIR) as folder:
            source = Path(folder)
            write_json(source / "evaluation_choice.json", {"selected": "search_only"})
            with self.assertRaises(ValueError):
                make_payload(source, source / "export")

    def test_fixed_comparison_does_not_change_selection(self):
        with tempfile.TemporaryDirectory(dir=RESULT_DIR) as folder:
            source = Path(folder)
            write_json(source / "selection_frozen.json", {"selected": "joint_reserve"})
            write_json(
                source / "evaluation_choice.json",
                {
                    "selected": "search_only",
                    "january_selection_winner": "joint_reserve",
                },
            )
            self.assertEqual(evaluation_choice(source)["selected"], "search_only")
            self.assertIn(
                "joint_reserve", (source / "selection_frozen.json").read_text()
            )


if __name__ == "__main__":
    unittest.main()
