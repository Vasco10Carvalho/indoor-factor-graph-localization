#!/usr/bin/env python3

"""Focused synthetic tests for the independent indoor Pose3/6D pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import gtsam
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indoor_localization_pose3"))

from pose3_common import (  # noqa: E402
    OdomSample,
    deskew_points,
    keyframe_due,
    map_visualization_mask,
    pose3_from_components,
)
from evaluate_pose3 import longest_true_run  # noqa: E402


class Pose3PipelineTests(unittest.TestCase):
    def test_map_visualization_mask_does_not_modify_source(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, -0.2], [1.0, 0.0, 1.2], [2.0, 0.0, 2.4]]
        )
        original = points.copy()
        mask = map_visualization_mask(points, -0.1, 2.15)
        self.assertEqual(mask.tolist(), [False, True, False])
        self.assertTrue(np.array_equal(points, original))

    def test_longest_true_run(self) -> None:
        self.assertEqual(longest_true_run(np.asarray([], dtype=bool)), 0)
        self.assertEqual(
            longest_true_run(np.asarray([False, True, True, False, True])),
            2,
        )
        self.assertEqual(longest_true_run(np.asarray([True, True, True])), 3)

    def test_keyframe_due_uses_timestamp_period(self) -> None:
        self.assertTrue(keyframe_due(None, 1_000, 500))
        self.assertFalse(keyframe_due(1_000, 1_499, 500))
        self.assertTrue(keyframe_due(1_000, 1_500, 500))
        self.assertFalse(keyframe_due(1_000, 999, 500))

    def test_pose3_factor_order(self) -> None:
        isam = gtsam.ISAM2()
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        x0 = gtsam.symbol("x", 0)
        x1 = gtsam.symbol("x", 1)
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
        )
        graph.add(gtsam.PriorFactorPose3(x0, gtsam.Pose3(), noise))
        delta = pose3_from_components(1.0, 0.2, 0.1, 1.0, -2.0, 5.0)
        graph.add(gtsam.BetweenFactorPose3(x0, x1, delta, noise))
        values.insert(x0, gtsam.Pose3())
        values.insert(x1, delta)
        isam.update(graph, values)
        result = isam.calculateEstimate().atPose3(x1)
        self.assertTrue(
            np.allclose(result.matrix(), delta.matrix(), atol=1e-8)
        )

    def test_deskew_constant_translation(self) -> None:
        samples = (
            OdomSample(0, pose3_from_components(0, 0, 0, 0, 0, 0)),
            OdomSample(
                1_000_000_000,
                pose3_from_components(1, 0, 0, 0, 0, 0),
            ),
        )
        points = np.asarray([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        times = np.asarray([0.0, 1_000_000_000.0])
        corrected, reference, used, status = deskew_points(
            points,
            times,
            0,
            samples,
            gtsam.Pose3(),
            time_scale=1e-9,
            reference="start",
            bins=64,
            max_odom_gap_sec=1.0,
        )
        self.assertEqual(reference, 0)
        self.assertTrue(used)
        self.assertEqual(status, "applied")
        self.assertTrue(np.allclose(corrected[:, 0], 10.0, atol=0.02))


if __name__ == "__main__":
    unittest.main()
