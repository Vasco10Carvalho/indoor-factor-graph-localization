#!/usr/bin/env python3

"""Evaluate and plot a recorded indoor Pose3 run without ground truth."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def load_rows(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV contains no keyframes: {path}")
    numeric = {}
    for name in rows[0]:
        try:
            numeric[name] = np.asarray(
                [float(row[name]) if row[name] != "" else np.nan for row in rows],
                dtype=float,
            )
        except ValueError:
            continue
    numeric["status_text"] = np.asarray([row["status"] for row in rows], dtype=object)
    return numeric


def aligned_odometry(data: dict[str, np.ndarray]) -> np.ndarray:
    odom_xyz = np.column_stack((data["odom_x"], data["odom_y"], data["odom_z"]))
    odom_rpy = np.column_stack(
        (data["odom_roll_deg"], data["odom_pitch_deg"], data["odom_yaw_deg"])
    )
    map_xyz0 = np.asarray([data["map_x"][0], data["map_y"][0], data["map_z"][0]])
    map_rotation0 = Rotation.from_euler(
        "xyz",
        [data["roll_deg"][0], data["pitch_deg"][0], data["yaw_deg"][0]],
        degrees=True,
    ).as_matrix()
    odom_rotation0 = Rotation.from_euler("xyz", odom_rpy[0], degrees=True).as_matrix()
    relative_translation = (odom_xyz - odom_xyz[0]) @ odom_rotation0
    return relative_translation @ map_rotation0.T + map_xyz0


def finite_percentile(values: np.ndarray, value: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, value)) if len(finite) else None


def longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def evaluate(data: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    xyz = np.column_stack((data["map_x"], data["map_y"], data["map_z"]))
    raw = aligned_odometry(data)
    jumps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    raw_jumps = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    elapsed = (data["stamp_ns"] - data["stamp_ns"][0]) / 1e9
    accepted = data["icp_accepted"] > 0.5
    statuses = data["status_text"]
    icp_attempted = accepted | np.asarray(
        [status.startswith("rejected_icp_") for status in statuses]
    )
    target_unavailable = statuses == "rejected_too_few_target_points"
    status_counts = Counter(str(status) for status in statuses)
    accepted_indices = np.flatnonzero(accepted)
    last_accepted_index = int(accepted_indices[-1]) if len(accepted_indices) else None
    map_constrained = np.zeros(len(xyz), dtype=bool)
    if last_accepted_index is not None:
        map_constrained[: last_accepted_index + 1] = True
    map_constrained_count = int(np.sum(map_constrained))
    late = elapsed >= min(60.0, float(elapsed[-1]) * 0.25)
    return_distance = np.linalg.norm(xyz - xyz[0], axis=1)
    raw_return_distance = np.linalg.norm(raw - raw[0], axis=1)
    summary = {
        "keyframes": int(len(xyz)),
        "duration_sec": float(elapsed[-1]),
        "icp": {
            "accepted": int(np.sum(accepted)),
            "rejected": int(np.sum(~accepted)),
            "accepted_percentage": float(100.0 * np.mean(accepted)),
            "fitness_p05": finite_percentile(data["fitness"][accepted], 5),
            "fitness_p50": finite_percentile(data["fitness"][accepted], 50),
            "rmse_p50_m": finite_percentile(data["rmse"][accepted], 50),
            "rmse_p95_m": finite_percentile(data["rmse"][accepted], 95),
            "correspondences_p05": finite_percentile(data["correspondences"][accepted], 5),
            "maximum_accepted_correction_xyz_m": finite_percentile(
                data["correction_xyz_m"][accepted], 100
            ),
            "maximum_accepted_correction_rotation_deg": finite_percentile(
                data["correction_rotation_deg"][accepted], 100
            ),
            "attempted": int(np.sum(icp_attempted)),
            "accepted_percentage_when_attempted": (
                float(100.0 * np.sum(accepted) / np.sum(icp_attempted))
                if np.any(icp_attempted)
                else None
            ),
        },
        "map_constraint": {
            "target_unavailable_keyframes": int(np.sum(target_unavailable)),
            "target_unavailable_percentage": float(
                100.0 * np.mean(target_unavailable)
            ),
            "first_target_unavailable_sec": (
                float(elapsed[np.flatnonzero(target_unavailable)[0]])
                if np.any(target_unavailable)
                else None
            ),
            "longest_target_unavailable_run": longest_true_run(target_unavailable),
            "last_accepted_sec": (
                float(elapsed[last_accepted_index])
                if last_accepted_index is not None
                else None
            ),
            "keyframes_through_last_accept": map_constrained_count,
            "accepted_percentage_through_last_accept": (
                float(100.0 * np.sum(accepted[map_constrained]) / map_constrained_count)
                if map_constrained_count
                else None
            ),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "runtime_ms": {
            "processing_p50": finite_percentile(data["processing_ms"], 50),
            "processing_p95": finite_percentile(data["processing_ms"], 95),
            "icp_p50": finite_percentile(data["icp_ms"], 50),
            "icp_p95": finite_percentile(data["icp_ms"], 95),
            "isam_p50": finite_percentile(data["isam_ms"], 50),
            "isam_p95": finite_percentile(data["isam_ms"], 95),
        },
        "trajectory": {
            "length_3d_m": float(np.sum(jumps)),
            "raw_odometry_length_3d_m": float(np.sum(raw_jumps)),
            "maximum_keyframe_jump_m": float(np.max(jumps)) if len(jumps) else 0.0,
            "endpoint_distance_to_start_m": float(return_distance[-1]),
            "raw_endpoint_distance_to_start_m": float(raw_return_distance[-1]),
            "closest_return_after_initial_segment_m": (
                float(np.min(return_distance[late])) if np.any(late) else None
            ),
            "raw_closest_return_after_initial_segment_m": (
                float(np.min(raw_return_distance[late])) if np.any(late) else None
            ),
            "z_min_m": float(np.min(xyz[:, 2])),
            "z_max_m": float(np.max(xyz[:, 2])),
            "roll_abs_max_deg": float(np.max(np.abs(data["roll_deg"]))),
            "pitch_abs_max_deg": float(np.max(np.abs(data["pitch_deg"]))),
            "map_constrained_z_min_m": (
                float(np.min(xyz[map_constrained, 2]))
                if np.any(map_constrained)
                else None
            ),
            "map_constrained_z_max_m": (
                float(np.max(xyz[map_constrained, 2]))
                if np.any(map_constrained)
                else None
            ),
            "map_constrained_roll_abs_max_deg": (
                float(np.max(np.abs(data["roll_deg"][map_constrained])))
                if np.any(map_constrained)
                else None
            ),
            "map_constrained_pitch_abs_max_deg": (
                float(np.max(np.abs(data["pitch_deg"][map_constrained])))
                if np.any(map_constrained)
                else None
            ),
        },
        "loop_closures_accepted": int(np.nansum(data["loop_accepted"])),
        "deskew_applied_percentage": float(100.0 * np.nanmean(data["deskew_applied"])),
        "note": "No ground truth was recorded; return and drift metrics are consistency indicators, not absolute accuracy.",
    }
    return summary, raw


def plot(data: dict[str, np.ndarray], raw: np.ndarray, path: Path) -> None:
    xyz = np.column_stack((data["map_x"], data["map_y"], data["map_z"]))
    predicted = np.column_stack((data["predicted_x"], data["predicted_y"], data["predicted_z"]))
    scan = np.column_stack((data["scan_x"], data["scan_y"], data["scan_z"]))
    elapsed = (data["stamp_ns"] - data["stamp_ns"][0]) / 1e9
    accepted = data["icp_accepted"] > 0.5
    accepted_scan = scan.copy()
    accepted_scan[~accepted] = np.nan
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(raw[:, 0], raw[:, 1], color="darkorange", label="odometria alinhada", linewidth=1.2)
    axes[0, 0].plot(predicted[:, 0], predicted[:, 1], color="gray", label="previsão do grafo", linewidth=1.0)
    axes[0, 0].plot(
        accepted_scan[:, 0],
        accepted_scan[:, 1],
        color="royalblue",
        label="scan-to-map 3D aceite",
        linewidth=1.0,
    )
    axes[0, 0].plot(xyz[:, 0], xyz[:, 1], color="crimson", label="factor graph Pose3", linewidth=1.6)
    axes[0, 0].scatter(xyz[0, 0], xyz[0, 1], color="blue", marker="o", label="início")
    axes[0, 0].axis("equal")
    axes[0, 0].set_title("Trajetória XY")
    axes[0, 0].set_xlabel("X [m]")
    axes[0, 0].set_ylabel("Y [m]")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(elapsed, raw[:, 2], color="darkorange", label="odom Z")
    axes[0, 1].plot(elapsed, xyz[:, 2], color="crimson", label="Pose3 Z")
    axes[0, 1].set_title("Altura")
    axes[0, 1].set_xlabel("Tempo [s]")
    axes[0, 1].set_ylabel("Z [m]")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(elapsed, data["fitness"], color="royalblue", label="fitness")
    rmse_axis = axes[1, 0].twinx()
    rmse_axis.plot(elapsed, data["rmse"], color="seagreen", alpha=0.8, label="RMSE")
    axes[1, 0].scatter(elapsed[~accepted], data["fitness"][~accepted], color="red", s=8, label="rejeitado")
    axes[1, 0].set_title("Qualidade ICP")
    axes[1, 0].set_xlabel("Tempo [s]")
    axes[1, 0].set_ylabel("Fitness")
    rmse_axis.set_ylabel("RMSE [m]")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(elapsed, data["processing_ms"], color="black", linewidth=1.0, label="total")
    axes[1, 1].plot(elapsed, data["icp_ms"], color="purple", linewidth=1.0, label="ICP")
    axes[1, 1].plot(elapsed, data["isam_ms"], color="teal", linewidth=1.0, label="iSAM2")
    axes[1, 1].set_title("Tempo de processamento")
    axes[1, 1].set_xlabel("Tempo [s]")
    axes[1, 1].set_ylabel("ms")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    data = load_rows(args.csv)
    summary, raw = evaluate(data)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot(data, raw, args.output_plot)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
