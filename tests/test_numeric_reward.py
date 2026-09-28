import unittest

from scripts.dataset_pilot.common import answer_score


class NumericRewardTests(unittest.TestCase):
    def test_extreme_exponents_do_not_abort_partial_credit(self):
        self.assertEqual(answer_score("1e999999", ["2e999999"]), 0.5)
        self.assertEqual(answer_score("1e1000000", ["2e1000000"]), 0.0)

    def test_ordinary_close_number_keeps_partial_credit(self):
        self.assertAlmostEqual(answer_score("42130", ["42138"]), 42130 / 42138)


if __name__ == "__main__":
    unittest.main()
