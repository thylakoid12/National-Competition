import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from q2_m3_resume import sweep, atomic_json, read_json

class Fake:
    cfg = SimpleNamespace(steps=(1., .25, .05), B=1., objective_tolerance=1e-7)
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at
    def evaluate(self, G, model):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('simulated power loss')
        return (float(np.sum(G * G)),)

class ResumeTests(unittest.TestCase):
    def state(self):
        G = np.arange(144, dtype=float) / 30
        return dict(G=G.tolist(), objective=float(G @ G), scale=0, rounds=[])

    def test_interrupted_round_is_not_committed(self):
        initial = self.state()
        state = copy.deepcopy(initial)
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            sweep(Fake(173), state)
        self.assertEqual(state, initial)

    def test_resume_equals_uninterrupted_and_preserves_steps(self):
        full, partial = self.state(), self.state()
        for _ in range(20):
            sweep(Fake(), full)
        for _ in range(5):
            sweep(Fake(), partial)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.json'
            atomic_json(path, partial)
            resumed = read_json(path)
            for _ in range(15):
                sweep(Fake(), resumed)
        self.assertEqual(full, resumed)
        values = [r['objective'] for r in full['rounds']]
        self.assertTrue(all(b <= a for a, b in zip(values, values[1:])))

    def test_no_improvement_advances_scale(self):
        state = dict(G=[0.] * 144, objective=0., scale=0, rounds=[])
        for index, step in enumerate(Fake.cfg.steps):
            row = sweep(Fake(), state)
            self.assertFalse(row['accepted'])
            self.assertEqual(row['attempts'], 288)
            self.assertEqual(row['step_kwh'], step)
            self.assertEqual(state['scale'], index + 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)

