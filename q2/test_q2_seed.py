import unittest
from unittest.mock import patch
from q2_seed import deterministic_seed
from q2_config import Config

class SeedTests(unittest.TestCase):
    def test_failed_optional_seed_is_explicit(self):
        with patch('common.optimization.solve_model',side_effect=RuntimeError('forced validation failure')):
            result,log=deterministic_seed(None,Config(),6000)
        self.assertIsNone(result)
        self.assertEqual(log['status'],'optional_q1_seed_failed')
        self.assertEqual(log['attempts'],2)

if __name__=='__main__':
    unittest.main()
