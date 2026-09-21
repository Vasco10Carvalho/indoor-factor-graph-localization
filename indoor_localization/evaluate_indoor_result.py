#!/usr/bin/env python3
"""Summarize and plot one indoor localization CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def percentile(values: np.ndarray, value: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--odom-csv", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--plot-out", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--map", dest="map_path", default="")
    args = parser.parse_args()

    with args.csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No poses found in {args.csv}")

    xyz = np.asarray(
        [[float(row["map_x"]), float(row["map_y"]), float(row["map_z"])] for row in rows]
    )
    rpy = np.asarray(
        [
            [float(row["roll_deg"]), float(row["pitch_deg"]), float(row["yaw_deg"])]
            for row in rows
        ]
    )
    inliers = np.asarray([float(row["scan_inlier_ratio"]) for row in rows])
    distances = np.asarray([float(row["scan_mean_distance"]) for row in rows])
    stamps = np.asarray([int(row["stamp_ns"]) for row in rows], dtype=np.int64)
    statuses = [row["scan_status"] for row in rows]
    rejected = np.asarray([status == "rejected_scan_match" for status in statuses])
    recovered = sum(status == "recovered_amcl" for status in statuses)
    loop_rows = [index for index, row in enumerate(rows) if row["loop_candidate_index"]]
    steps = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)

    runtime = None
    if args.status_json and args.status_json.exists():
        runtime = json.loads(args.status_json.read_text(encoding="utf-8"))

    odom_xy = None
    odom_stamps = None
    odom_comparison = None
    if args.odom_csv and args.odom_csv.exists():
        with args.odom_csv.open("r", encoding="utf-8", newline="") as stream:
            odom_rows = list(csv.DictReader(stream))
        if odom_rows:
            odom_xy = np.asarray(
                [[float(row["map_x"]), float(row["map_y"])] for row in odom_rows]
            )
            odom_stamps = np.asarray(
                [int(row["stamp_ns"]) for row in odom_rows], dtype=np.int64
            )
            overlap = (odom_stamps >= stamps[0]) & (odom_stamps <= stamps[-1])
            overlap_stamps = odom_stamps[overlap]
            if overlap_stamps.size:
                time_origin = min(int(stamps[0]), int(overlap_stamps[0]))
                fused_seconds = (stamps - time_origin) / 1e9
                odom_seconds = (overlap_stamps - time_origin) / 1e9
                fused_at_odom = np.column_stack(
                    (
                        np.interp(odom_seconds, fused_seconds, xyz[:, 0]),
                        np.interp(odom_seconds, fused_seconds, xyz[:, 1]),
                    )
                )
                separation = np.linalg.norm(odom_xy[overlap] - fused_at_odom, axis=1)
                odom_steps = np.linalg.norm(np.diff(odom_xy, axis=0), axis=1)
                odom_comparison = {
                    "poses": len(odom_rows),
                    "path_length_xy_m": float(np.sum(odom_steps)),
                    "end_xy_m": odom_xy[-1].tolist(),
                    "fused_vs_odom_xy_m": {
                        "median": percentile(separation, 50),
                        "p95": percentile(separation, 95),
                        "end": float(separation[-1]),
                    },
                }

    summary = {
        "dataset": args.dataset,
        "map": args.map_path,
        "duration_sec": float((stamps[-1] - stamps[0]) / 1e9),
        "keyframes": len(rows),
        "accepted": int(len(rows) - np.count_nonzero(rejected)),
        "rejected": int(np.count_nonzero(rejected)),
        "acceptance_percent": float(100.0 * np.count_nonzero(~rejected) / len(rows)),
        "amcl_recoveries": int(recovered),
        "loop_closures": len(loop_rows),
        "path_length_xy_m": float(np.sum(steps)),
        "net_displacement_xy_m": float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])),
        "median_step_xy_m": percentile(steps, 50),
        "p95_step_xy_m": percentile(steps, 95),
        "start_xyz_m": xyz[0].tolist(),
        "end_xyz_m": xyz[-1].tolist(),
        "xy_bounds_m": {
            "x": [float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0]))],
            "y": [float(np.min(xyz[:, 1])), float(np.max(xyz[:, 1]))],
        },
        "inlier_ratio": {"median": percentile(inliers, 50), "p05": percentile(inliers, 5)},
        "mean_wall_distance_m": {
            "median": percentile(distances, 50),
            "p95": percentile(distances, 95),
        },
        "attitude_deg": {
            "roll_abs_p95": percentile(np.abs(rpy[:, 0]), 95),
            "pitch_abs_p95": percentile(np.abs(rpy[:, 1]), 95),
        },
        "runtime_monitor": runtime,
        "odometry_only_comparison": odom_comparison,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    if odom_xy is not None:
        axis.plot(
            odom_xy[:, 0],
            odom_xy[:, 1],
            color="#f28e2b",
            linewidth=1.4,
            alpha=0.9,
            label="Odometry only",
        )
    axis.plot(xyz[:, 0], xyz[:, 1], color="#087e8b", linewidth=2.0, label="Factor graph")
    if np.any(rejected):
        axis.scatter(
            xyz[rejected, 0],
            xyz[rejected, 1],
            color="#ef476f",
            s=16,
            label="Rejected scan",
            zorder=3,
        )
    for position, row_index in enumerate(loop_rows):
        candidate = int(rows[row_index]["loop_candidate_index"])
        axis.plot(
            [xyz[candidate, 0], xyz[row_index, 0]],
            [xyz[candidate, 1], xyz[row_index, 1]],
            color="#ffb703",
            linewidth=0.8,
            alpha=0.55,
            label="Loop closure" if position == 0 else None,
        )
    axis.scatter(*xyz[0, :2], color="#2a9d8f", s=70, label="Start", zorder=4)
    axis.scatter(*xyz[-1, :2], color="#264653", s=70, label="End", zorder=4)
    model = runtime.get("model", "Indoor factor-graph localization") if runtime else "Indoor factor-graph localization"
    axis.set_title(model)
    axis.set_xlabel("map X [m]")
    axis.set_ylabel("map Y [m]")
    axis.axis("equal")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot_out, dpi=180)
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
