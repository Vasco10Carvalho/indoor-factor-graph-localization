"""Recovery gating and RViz discontinuity checks for the indoor graph."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock, patch

import gtsam
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "indoor_localization"))
from indoor_scan_matcher import MatchResult
from online_indoor_factor_graph_pose2 import IndoorPose2FactorGraph, OdomSample


def match(x: float, y: float = 0.0, yaw: float = 0.5) -> MatchResult:
    return MatchResult(x, y, yaw, 0.02, 0.8, 0.2, 1.0)


def make_node() -> IndoorPose2FactorGraph:
    node = IndoorPose2FactorGraph.__new__(IndoorPose2FactorGraph)
    node.args = SimpleNamespace(
        enable_scan_recovery=True,
        recovery_after_rejections=5,
        recovery_check_interval=1,
        recovery_particles=10,
        global_seed=7,
        truncation_distance=1.5,
        recovery_minimum_inlier_ratio=0.65,
        recovery_maximum_mean_distance=0.30,
        minimum_valid_ratio=0.75,
        recovery_maximum_xy_distance=8.0,
        recovery_maximum_yaw_deg=100.0,
        recovery_confirmation_xy_distance=1.0,
        recovery_confirmation_yaw_deg=15.0,
        recovery_confirmations=3,
        map_frame="map",
    )
    node.map = object()
    node.consecutive_scan_rejections = 5
    node.recovery_hypothesis = None
    node.recovery_attempts = 0
    return node


class IndoorScanRecoveryTests(unittest.TestCase):
    def test_requires_three_consistent_matches(self):
        node = make_node()
        points = np.zeros((30, 2))
        candidates = [match(1.0), match(1.1), match(1.2)]
        with patch(
            "online_indoor_factor_graph_pose2.global_scan_match",
            side_effect=candidates,
        ), patch.object(
            IndoorPose2FactorGraph,
            "get_logger",
            return_value=SimpleNamespace(warning=lambda _: None),
        ):
            results = [
                node.try_scan_recovery(
                    points,
                    gtsam.Pose2(0.1 * index, 0.0, 0.0),
                    OdomSample(index, gtsam.Pose2(0.1 * index, 0.0, 0.0)),
                )
                for index in range(3)
            ]
        self.assertEqual(results, [None, None, candidates[2]])
        self.assertEqual(node.recovery_attempts, 3)

    def test_rejects_distant_match(self):
        node = make_node()
        with patch("online_indoor_factor_graph_pose2.global_scan_match", return_value=match(20.0)):
            result = node.try_scan_recovery(
                np.zeros((30, 2)), gtsam.Pose2(), OdomSample(0, gtsam.Pose2())
            )
        self.assertIsNone(result)
        self.assertIsNone(node.recovery_hypothesis)

    def test_marker_does_not_draw_relocalization_jump(self):
        node = make_node()
        node.segments_publisher = MagicMock()
        node.recovery_markers_publisher = MagicMock()
        poses = [
            (index, gtsam.Pose2(x, 0.0, 0.0))
            for index, x in enumerate((0.0, 1.0, 10.0, 11.0))
        ]
        node.publish_path_markers(poses, {2})
        segments = node.segments_publisher.publish.call_args.args[0]
        recoveries = node.recovery_markers_publisher.publish.call_args.args[0]
        self.assertEqual(len(segments.points), 4)
        self.assertEqual([round(p.x) for p in segments.points], [0, 1, 10, 11])
        self.assertEqual(len(recoveries.points), 1)
        self.assertAlmostEqual(recoveries.points[0].x, 10.0)


if __name__ == "__main__":
    unittest.main()
