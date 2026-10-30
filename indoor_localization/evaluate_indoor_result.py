#!/usr/bin/env python3
"""Summarize and plot one indoor localization CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


def percentile(values: np.ndarray, value: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, value))


def path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def separation_summary(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    distances = np.linalg.norm(first - second, axis=1)
    return {
        "median": percentile(distances, 50),
        "p95": percentile(distances, 95),
        "maximum": float(np.max(distances)) if distances.size else None,
    }


def load_map_image(map_yaml: Path | None):
    if map_yaml is None:
        return None
    metadata = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    image_path = Path(str(metadata["image"]))
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read occupancy image: {image_path}")
    resolution = float(metadata["resolution"])
    origin = metadata.get("origin", [0.0, 0.0, 0.0])
    if abs(float(origin[2])) > 1e-9:
        raise ValueError("Rotated occupancy-grid origins are not supported yet")
    height, width = image.shape
    extent = (
        float(origin[0]),
        float(origin[0]) + width * resolution,
        float(origin[1]),
        float(origin[1]) + height * resolution,
    )
    return image, extent


def draw_map(axis, map_image) -> None:
    if map_image is None:
        return
    image, extent = map_image
    axis.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        extent=extent,
        alpha=0.72,
        zorder=0,
    )


def set_trajectory_limits(axis, trajectories: list[np.ndarray]) -> None:
    valid = []
    for trajectory in trajectories:
        if trajectory is None or trajectory.size == 0:
            continue
        xy = np.asarray(trajectory)[:, :2]
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if xy.size:
            valid.append(xy)
    if not valid:
        return
    combined = np.vstack(valid)
    minimum = np.min(combined, axis=0)
    maximum = np.max(combined, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    padding = np.maximum(1.0, 0.08 * span)
    axis.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    axis.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--odom-csv", type=Path)
    parser.add_argument("--components-csv", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--map-yaml", type=Path)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--plot-out", type=Path, required=True)
    parser.add_argument("--comparison-plot-out", type=Path)
    parser.add_argument("--timeseries-plot-out", type=Path)
    parser.add_argument("--quality-plot-out", type=Path)
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
    time_origin = int(stamps[0])
    fused_seconds = (stamps - time_origin) / 1e9
    map_image = load_map_image(args.map_yaml)

    component_rows = []
    components = None
    if args.components_csv and args.components_csv.exists():
        with args.components_csv.open("r", encoding="utf-8", newline="") as stream:
            component_rows = list(csv.DictReader(stream))
        if component_rows:
            def values(prefix: str) -> np.ndarray:
                return np.asarray(
                    [
                        [
                            float(row[f"{prefix}_x"]),
                            float(row[f"{prefix}_y"]),
                            math.radians(float(row[f"{prefix}_yaw_deg"])),
                        ]
                        for row in component_rows
                    ],
                    dtype=float,
                )

            aligned_values = []
            for row in component_rows:
                if row["aligned_odom_x"]:
                    aligned_values.append(
                        [
                            float(row["aligned_odom_x"]),
                            float(row["aligned_odom_y"]),
                            math.radians(float(row["aligned_odom_yaw_deg"])),
                        ]
                    )
                else:
                    aligned_values.append([math.nan, math.nan, math.nan])
            components = {
                "stamps": np.asarray(
                    [int(row["stamp_ns"]) for row in component_rows], dtype=np.int64
                ),
                "accepted": np.asarray(
                    [bool(int(row["scan_accepted"])) for row in component_rows]
                ),
                "raw_odom": values("raw_odom"),
                "aligned_odom": np.asarray(aligned_values, dtype=float),
                "predicted": values("predicted"),
                "scan": values("scan"),
                "optimized": values("optimized"),
                "inliers": np.asarray(
                    [float(row["scan_inlier_ratio"]) for row in component_rows]
                ),
                "distance": np.asarray(
                    [float(row["scan_mean_distance"]) for row in component_rows]
                ),
                "correction_xy": np.asarray(
                    [float(row["scan_correction_xy"]) for row in component_rows]
                ),
                "correction_yaw": np.asarray(
                    [float(row["scan_correction_yaw_deg"]) for row in component_rows]
                ),
            }

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

    component_comparison = None
    if components is not None:
        accepted_components = components["accepted"]
        finite_aligned = np.all(
            np.isfinite(components["aligned_odom"][:, :2]), axis=1
        )
        component_comparison = {
            "poses": len(component_rows),
            "raw_odometry_path_length_xy_m": path_length(
                components["raw_odom"][:, :2]
            ),
            "aligned_odometry_path_length_xy_m": path_length(
                components["aligned_odom"][finite_aligned, :2]
            ),
            "predicted_path_length_xy_m": path_length(
                components["predicted"][:, :2]
            ),
            "scan_to_map_accepted_path_length_xy_m": path_length(
                components["scan"][accepted_components, :2]
            ),
            "fused_path_length_xy_m": path_length(
                components["optimized"][:, :2]
            ),
            "fused_vs_predicted_xy_m": separation_summary(
                components["optimized"][:, :2], components["predicted"][:, :2]
            ),
            "fused_vs_scan_to_map_xy_m": separation_summary(
                components["optimized"][accepted_components, :2],
                components["scan"][accepted_components, :2],
            ),
            "fused_vs_aligned_odometry_xy_m": separation_summary(
                components["optimized"][finite_aligned, :2],
                components["aligned_odom"][finite_aligned, :2],
            ),
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
        "trajectory_components_comparison": component_comparison,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    draw_map(axis, map_image)
    if odom_xy is not None:
        axis.plot(
            odom_xy[:, 0],
            odom_xy[:, 1],
            color="#f28e2b",
            linewidth=1.4,
            alpha=0.9,
            label="Odometry only",
        )
    axis.plot(
        xyz[:, 0],
        xyz[:, 1],
        color="#00a676",
        linewidth=2.2,
        label="Factor graph fusion",
    )
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
    set_trajectory_limits(axis, [odom_xy, xyz])
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot_out, dpi=180)
    plt.close(figure)

    if components is not None and args.comparison_plot_out is not None:
        figure, (map_axis, raw_axis) = plt.subplots(
            1, 2, figsize=(18, 8), constrained_layout=True
        )
        draw_map(map_axis, map_image)
        if odom_xy is not None:
            map_axis.plot(
                odom_xy[:, 0],
                odom_xy[:, 1],
                color="#f28e2b",
                linewidth=1.2,
                alpha=0.85,
                label="Aligned odometry",
            )
        predicted = components["predicted"]
        scan = components["scan"]
        optimized = components["optimized"]
        accepted_components = components["accepted"]
        rejected_components = ~accepted_components
        map_axis.plot(
            predicted[:, 0],
            predicted[:, 1],
            color="#277da1",
            linewidth=1.1,
            alpha=0.9,
            label="Prediction before scan",
        )
        map_axis.plot(
            scan[accepted_components, 0],
            scan[accepted_components, 1],
            color="#c77dff",
            linewidth=0.9,
            marker=".",
            markersize=2.5,
            alpha=0.75,
            label="Accepted scan-to-map",
        )
        if np.any(rejected_components):
            map_axis.scatter(
                scan[rejected_components, 0],
                scan[rejected_components, 1],
                color="#ef233c",
                marker="x",
                s=22,
                label="Rejected scan-to-map",
                zorder=5,
            )
        map_axis.plot(
            optimized[:, 0],
            optimized[:, 1],
            color="#00a676",
            linewidth=2.3,
            label="Factor graph fusion",
        )
        for position, row_index in enumerate(loop_rows):
            candidate = int(rows[row_index]["loop_candidate_index"])
            if candidate >= len(optimized) or row_index >= len(optimized):
                continue
            map_axis.plot(
                [optimized[candidate, 0], optimized[row_index, 0]],
                [optimized[candidate, 1], optimized[row_index, 1]],
                color="#ffb703",
                linewidth=1.0,
                alpha=0.7,
                label="Loop closure" if position == 0 else None,
            )
        map_axis.scatter(
            optimized[0, 0], optimized[0, 1], color="#2a9d8f", s=70, zorder=6
        )
        map_axis.scatter(
            optimized[-1, 0], optimized[-1, 1], color="#264653", s=70, zorder=6
        )
        map_axis.set_title("Map-frame trajectory comparison")
        map_axis.set_xlabel("map X [m]")
        map_axis.set_ylabel("map Y [m]")
        map_axis.axis("equal")
        set_trajectory_limits(
            map_axis,
            [odom_xy, predicted[:, :2], scan[:, :2], optimized[:, :2]],
        )
        map_axis.grid(alpha=0.2)
        map_axis.legend(loc="best")

        raw = components["raw_odom"]
        raw_axis.plot(
            raw[:, 0],
            raw[:, 1],
            color="#8c564b",
            linewidth=1.5,
            label="Raw /odometry/filtered",
        )
        raw_axis.scatter(raw[0, 0], raw[0, 1], color="#2a9d8f", s=60, label="Start")
        raw_axis.scatter(raw[-1, 0], raw[-1, 1], color="#264653", s=60, label="End")
        raw_axis.set_title("Raw odometry frame (not map-aligned)")
        raw_axis.set_xlabel("odom X [m]")
        raw_axis.set_ylabel("odom Y [m]")
        raw_axis.axis("equal")
        raw_axis.grid(alpha=0.2)
        raw_axis.legend(loc="best")
        args.comparison_plot_out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.comparison_plot_out, dpi=180)
        plt.close(figure)

    if components is not None and args.timeseries_plot_out is not None:
        seconds = (components["stamps"] - time_origin) / 1e9
        figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
        series = (
            ("aligned_odom", "Aligned odometry", "#f28e2b"),
            ("predicted", "Prediction", "#277da1"),
            ("scan", "Scan-to-map", "#c77dff"),
            ("optimized", "Factor graph fusion", "#00a676"),
        )
        for name, label, color in series:
            values = components[name]
            mask = np.all(np.isfinite(values), axis=1)
            if name == "scan":
                mask &= components["accepted"]
            axes[0].plot(seconds[mask], values[mask, 0], color=color, label=label)
            axes[1].plot(seconds[mask], values[mask, 1], color=color, label=label)
            yaw = np.degrees(np.unwrap(values[mask, 2]))
            axes[2].plot(seconds[mask], yaw, color=color, label=label)
        axes[0].set_ylabel("X [m]")
        axes[1].set_ylabel("Y [m]")
        axes[2].set_ylabel("Unwrapped yaw [deg]")
        axes[2].set_xlabel("Time [s]")
        axes[0].set_title("Trajectory components over time")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(loc="best", ncol=2)
        args.timeseries_plot_out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.timeseries_plot_out, dpi=180)
        plt.close(figure)

    if components is not None and args.quality_plot_out is not None:
        seconds = (components["stamps"] - time_origin) / 1e9
        accepted_components = components["accepted"]
        figure, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, constrained_layout=True)
        axes[0, 0].plot(seconds, components["inliers"], color="#577590")
        axes[0, 0].set_ylabel("Inlier ratio")
        axes[0, 1].plot(seconds, components["distance"], color="#43aa8b")
        axes[0, 1].set_ylabel("Mean wall distance [m]")
        axes[1, 0].plot(seconds, components["correction_xy"], color="#f8961e")
        axes[1, 0].set_ylabel("Scan correction XY [m]")
        axes[1, 1].plot(seconds, components["correction_yaw"], color="#9d4edd")
        axes[1, 1].set_ylabel("Scan correction yaw [deg]")
        for axis in axes.flat:
            if np.any(~accepted_components):
                axis.scatter(
                    seconds[~accepted_components],
                    np.interp(
                        seconds[~accepted_components],
                        seconds,
                        axis.lines[0].get_ydata(),
                    ),
                    color="#ef233c",
                    marker="x",
                    s=20,
                    label="Rejected",
                    zorder=4,
                )
                axis.legend(loc="best")
            axis.grid(alpha=0.2)
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 1].set_xlabel("Time [s]")
        axes[0, 0].set_title("Scan matching quality and corrections")
        args.quality_plot_out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.quality_plot_out, dpi=180)
        plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
