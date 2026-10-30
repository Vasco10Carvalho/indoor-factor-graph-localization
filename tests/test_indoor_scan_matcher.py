from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "indoor_localization"))
from indoor_scan_matcher import motion_prior_costs


class IndoorScanMatcherTests(unittest.TestCase):
    def test_motion_prior_breaks_near_ties_towards_prediction(self):
        predicted = np.asarray([0.0, 0.0, 0.0])
        candidates = np.asarray([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
        map_costs = np.asarray([0.011, 0.0])

        costs = motion_prior_costs(candidates, map_costs, predicted, 0.05, 0.20)

        self.assertEqual(int(np.argmin(costs)), 0)

    def test_decisive_map_improvement_overrides_motion_prior(self):
        predicted = np.asarray([0.0, 0.0, 0.0])
        candidates = np.asarray([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
        map_costs = np.asarray([0.10, 0.0])

        costs = motion_prior_costs(candidates, map_costs, predicted, 0.05, 0.20)

        self.assertEqual(int(np.argmin(costs)), 1)

    def test_zero_weights_reproduce_map_costs(self):
        predicted = np.asarray([1.0, 2.0, 0.3])
        candidates = np.asarray([[0.0, 0.0, -1.0], [4.0, 5.0, 2.0]])
        map_costs = np.asarray([0.03, 0.02])

        costs = motion_prior_costs(candidates, map_costs, predicted, 0.0, 0.0)

        np.testing.assert_array_equal(costs, map_costs)

    def test_negative_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            motion_prior_costs(
                np.zeros((1, 3)), np.zeros(1), np.zeros(3), -0.1, 0.0
            )


if __name__ == "__main__":
    unittest.main()
