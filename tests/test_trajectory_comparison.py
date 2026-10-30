#!/usr/bin/env python3
"""Small dependency-free checks for the trajectory comparison geometry."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trajectory_comparison"))

from comparison_common import (  # noqa: E402
    Trajectory,
    align_initial_pose,
    load_trajectory,
    path_length,
    wrap_angle,
)


class TrajectoryComparisonTests(unittest.TestCase):
    def test_initial_se2_alignment(self) -> None:
        stamps = np.asarray([1_000_000_000, 2_000_000_000, 3_000_000_000])
        reference = Trajectory(
            "reference",
            stamps,
            np.asarray([10.0, 10.0, 9.0]),
            np.asarray([20.0, 21.0, 21.0]),
            np.asarray([math.pi / 2.0, math.pi / 2.0, math.pi]),
        )
        method = Trajectory(
            "method",
            stamps,
            np.asarray([3.0, 4.0, 4.0]),
            np.asarray([-2.0, -2.0, -1.0]),
            np.asarray([0.0, 0.0, math.pi / 2.0]),
        )
        aligned = align_initial_pose(method, reference)
        np.testing.assert_allclose(aligned.x, reference.x, atol=1e-9)
        np.testing.assert_allclose(aligned.y, reference.y, atol=1e-9)
        np.testing.assert_allclose(
            wrap_angle(aligned.yaw_rad - reference.yaw_rad), 0.0
        )
        self.assertTrue(math.isclose(path_length(aligned), 2.0))

    def test_factor_graph_csv_loader(self) -> None:
        contents = (
            "stamp_ns,map_x,map_y,map_z,roll_deg,pitch_deg,yaw_deg\n"
            "1790155290377113742,2.0,4.0,0,0,0,90\n"
            "1790155290377113642,1.0,3.0,0,0,0,0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "trajectory.csv"
            csv_path.write_text(contents, encoding="utf-8")
            trajectory = load_trajectory(csv_path, "fg")
        self.assertEqual(
            trajectory.stamp_ns.tolist(),
            [1790155290377113642, 1790155290377113742],
        )
        np.testing.assert_allclose(trajectory.x, [1.0, 2.0])
        np.testing.assert_allclose(trajectory.yaw_rad, [0.0, math.pi / 2.0])

    def test_rviz_trajectories_are_independent_and_colored(self) -> None:
        config_path = ROOT / "trajectory_comparison" / "trajectory_comparison.rviz"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        displays = config["Visualization Manager"]["Displays"]
        paths = {
            display["Topic"]["Value"]: display
            for display in displays
            if display.get("Class") == "rviz_default_plugins/Path"
        }
        expected = {
            "/trajectory_comparison/factor_graph_2d/path": "0; 72; 150",
            "/trajectory_comparison/slam_toolbox_localization/path": "0; 172; 193",
            "/trajectory_comparison/amcl/path": "156; 39; 176",
            "/trajectory_comparison/kiss_icp_odometry/path": "255; 152; 0",
            "/trajectory_comparison/cartographer_pure_localization/path": "67; 160; 71",
            "/trajectory_comparison/kiss_icp_cartographer_pure_localization/path": (
                "229; 57; 53"
            ),
        }
        self.assertEqual(set(paths), set(expected))
        for topic, color in expected.items():
            self.assertEqual(paths[topic]["Color"], color)
            self.assertTrue(paths[topic]["Enabled"])


if __name__ == "__main__":
    unittest.main()
