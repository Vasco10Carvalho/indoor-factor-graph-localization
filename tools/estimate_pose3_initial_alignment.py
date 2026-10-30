#!/usr/bin/env python3

"""Estimate an initial map_T_base pose from a local LiDAR scan and a PLY map."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402


def inclusive_values(lower: float, upper: float, step: float) -> np.ndarray:
    if step <= 0.0 or upper < lower:
        raise ValueError("Search bounds require upper >= lower and step > 0")
    count = int(math.floor((upper - lower) / step + 1.0e-9)) + 1
    return lower + np.arange(count, dtype=float) * step


def pose_matrix(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = (x, y, z)
    return transform


def matrix_pose(transform: np.ndarray) -> dict[str, list[float]]:
    rotation = transform[:3, :3]
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return {
        "xyz": transform[:3, 3].astype(float).tolist(),
        "rpy_deg": np.degrees([roll, pitch, yaw]).astype(float).tolist(),
    }


def load_structure_cloud(
    path: Path,
    voxel: float,
    z_min: float,
    z_max: float,
) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if not len(cloud.points):
        raise RuntimeError(f"Empty or unreadable point cloud: {path}")
    cloud = cloud.voxel_down_sample(voxel)
    points = np.asarray(cloud.points)
    keep = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    cloud = cloud.select_by_index(np.flatnonzero(keep).tolist())
    if not len(cloud.points):
        raise RuntimeError(f"No structural points remain in {path}")
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(voxel * 4.0, 0.50),
            max_nn=50,
        )
    )
    return cloud


def coarse_score(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    max_correspondence: float,
) -> tuple[float, float, float]:
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        max_correspondence,
        transform,
    )
    score = float(evaluation.fitness) - 0.25 * float(evaluation.inlier_rmse)
    return score, float(evaluation.fitness), float(evaluation.inlier_rmse)


def refine_candidate(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial: np.ndarray,
    coarse_distance: float,
    fine_distance: float,
    planar: bool,
) -> dict[str, object]:
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    current = initial.copy()
    if planar:
        fixed_z = float(initial[2, 3])
        # Coordinate descent stays in SE(2), so repeated horizontal surfaces
        # cannot pull a ground robot vertically during initialization.
        stages = (
            (coarse_distance, 0.40, 1.00),
            (max(fine_distance * 2.0, 0.40), 0.12, 0.30),
            (fine_distance, 0.03, 0.08),
        )
        for distance, translation_step, yaw_step in stages:
            for _ in range(30):
                yaw = math.degrees(math.atan2(current[1, 0], current[0, 0]))
                x, y = float(current[0, 3]), float(current[1, 3])
                choices = [
                    current,
                    pose_matrix(x + translation_step, y, fixed_z, yaw),
                    pose_matrix(x - translation_step, y, fixed_z, yaw),
                    pose_matrix(x, y + translation_step, fixed_z, yaw),
                    pose_matrix(x, y - translation_step, fixed_z, yaw),
                    pose_matrix(x, y, fixed_z, yaw + yaw_step),
                    pose_matrix(x, y, fixed_z, yaw - yaw_step),
                ]
                scored = [
                    (coarse_score(source, target, choice, distance)[0], choice)
                    for choice in choices
                ]
                best_score, best_choice = max(scored, key=lambda item: item[0])
                current_score = scored[0][0]
                if best_score <= current_score + 1.0e-9:
                    break
                current = best_choice
        fine = o3d.pipelines.registration.evaluate_registration(
            source,
            target,
            fine_distance,
            current,
        )
        final_transform = current
    else:
        coarse = o3d.pipelines.registration.registration_icp(
            source,
            target,
            coarse_distance,
            current,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=45),
        )
        fine = o3d.pipelines.registration.registration_icp(
            source,
            target,
            fine_distance,
            coarse.transformation,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        final_transform = np.asarray(fine.transformation)
    return {
        "fitness": float(fine.fitness),
        "rmse": float(fine.inlier_rmse),
        "pose": matrix_pose(final_transform),
        "transform": final_transform.astype(float).tolist(),
    }


def read_odometry(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                (
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    float(row["roll_deg"]),
                    float(row["pitch_deg"]),
                    float(row["yaw_deg"]),
                )
            )
    if not rows:
        raise RuntimeError(f"No odometry rows found in {path}")
    return np.asarray(rows, dtype=float)


def rpy_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def odom_matrix(row: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(*row[3:])
    transform[:3, 3] = row[:3]
    return transform


def aligned_odometry(odometry: np.ndarray, map_to_base: np.ndarray) -> np.ndarray:
    first = odom_matrix(odometry[0])
    map_to_odom = map_to_base @ np.linalg.inv(first)
    aligned = np.empty((len(odometry), 3), dtype=float)
    for index, row in enumerate(odometry):
        aligned[index] = (map_to_odom @ odom_matrix(row))[:3, 3]
    return aligned


def write_preview(
    output: Path,
    target: o3d.geometry.PointCloud,
    source: o3d.geometry.PointCloud,
    transform: np.ndarray,
    odometry: np.ndarray | None,
) -> None:
    target_points = np.asarray(target.points)
    if len(target_points) > 250_000:
        selected = np.random.default_rng(20261022).choice(
            len(target_points), 250_000, replace=False
        )
        target_points = target_points[selected]
    source_points = np.asarray(source.points)
    transformed_source = (
        source_points @ transform[:3, :3].T + transform[:3, 3]
    )

    figure, axis = plt.subplots(figsize=(10, 13), dpi=160)
    axis.scatter(
        target_points[:, 0], target_points[:, 1], s=0.20, c="#555f66", alpha=0.40
    )
    axis.scatter(
        transformed_source[:, 0],
        transformed_source[:, 1],
        s=1.0,
        c="#00a6d6",
        alpha=0.75,
        label="initial LiDAR scans",
    )
    if odometry is not None:
        route = aligned_odometry(odometry, transform)
        axis.plot(
            route[:, 0],
            route[:, 1],
            color="#e33d32",
            linewidth=1.6,
            label="raw odometry aligned at start",
        )
        axis.scatter(route[0, 0], route[0, 1], s=55, c="#00a650", label="start")
        axis.scatter(route[-1, 0], route[-1, 1], s=55, c="#ff8c00", label="end")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Map X [m]")
    axis.set_ylabel("Map Y [m]")
    axis.set_title("Pose3 initial alignment validation")
    axis.grid(color="#dddddd", linewidth=0.35)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--odom-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x-range", nargs=3, type=float, required=True)
    parser.add_argument("--y-range", nargs=3, type=float, required=True)
    parser.add_argument("--yaw-range", nargs=3, type=float, required=True)
    parser.add_argument("--initial-z", type=float, default=0.0)
    parser.add_argument("--voxel", type=float, default=0.20)
    parser.add_argument("--source-z-min", type=float, default=0.20)
    parser.add_argument("--source-z-max", type=float, default=3.15)
    parser.add_argument("--target-z-min", type=float, default=0.20)
    parser.add_argument("--target-z-max", type=float, default=3.15)
    parser.add_argument("--coarse-distance", type=float, default=0.80)
    parser.add_argument("--fine-distance", type=float, default=0.25)
    parser.add_argument("--refine-count", type=int, default=12)
    parser.add_argument(
        "--allow-6d-refinement",
        action="store_true",
        help="Allow ICP to change Z, roll and pitch during initial alignment",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    map_path = args.map.expanduser().resolve()
    scan_path = args.scan.expanduser().resolve()
    source = load_structure_cloud(
        scan_path, args.voxel, args.source_z_min, args.source_z_max
    )
    target = load_structure_cloud(
        map_path, args.voxel, args.target_z_min, args.target_z_max
    )
    print(
        f"[INFO] Structural points: source={len(source.points)}, "
        f"target={len(target.points)}",
        flush=True,
    )

    coarse_candidates = []
    x_values = inclusive_values(*args.x_range)
    y_values = inclusive_values(*args.y_range)
    yaw_values = inclusive_values(*args.yaw_range)
    total = len(x_values) * len(y_values) * len(yaw_values)
    tested = 0
    for yaw in yaw_values:
        for x in x_values:
            for y in y_values:
                transform = pose_matrix(x, y, args.initial_z, yaw)
                score, fitness, rmse = coarse_score(
                    source, target, transform, args.coarse_distance
                )
                coarse_candidates.append(
                    {
                        "score": score,
                        "fitness": fitness,
                        "rmse": rmse,
                        "seed": [float(x), float(y), args.initial_z, float(yaw)],
                        "transform": transform,
                    }
                )
                tested += 1
        print(f"[INFO] Coarse search: {tested}/{total}", flush=True)
    coarse_candidates.sort(key=lambda item: item["score"], reverse=True)

    refined = []
    for rank, candidate in enumerate(
        coarse_candidates[: max(1, args.refine_count)], start=1
    ):
        result = refine_candidate(
            source,
            target,
            candidate["transform"],
            args.coarse_distance,
            args.fine_distance,
            planar=not args.allow_6d_refinement,
        )
        result["seed"] = candidate["seed"]
        result["coarse_rank"] = rank
        refined.append(result)
        print(
            f"[INFO] Refined {rank}: fitness={result['fitness']:.3f}, "
            f"rmse={result['rmse']:.3f}, pose={result['pose']}",
            flush=True,
        )
    refined.sort(key=lambda item: (-item["fitness"], item["rmse"]))
    best = refined[0]
    best_transform = np.asarray(best["transform"], dtype=float)

    odometry = None
    if args.odom_csv is not None:
        odometry = read_odometry(args.odom_csv.expanduser().resolve())
    preview_path = output / "initial_alignment_preview.png"
    write_preview(preview_path, target, source, best_transform, odometry)
    payload = {
        "map": str(map_path),
        "scan": str(scan_path),
        "odometry_csv": str(args.odom_csv) if args.odom_csv else None,
        "search": {
            "x_range": args.x_range,
            "y_range": args.y_range,
            "yaw_range_deg": args.yaw_range,
            "initial_z": args.initial_z,
            "tested_candidates": total,
            "voxel": args.voxel,
            "coarse_distance": args.coarse_distance,
            "fine_distance": args.fine_distance,
            "planar_refinement": not args.allow_6d_refinement,
        },
        "best": best,
        "refined_candidates": refined,
        "preview": str(preview_path),
    }
    payload_path = output / "initial_alignment.json"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2), flush=True)
    print(f"[INFO] Preview: {preview_path}", flush=True)
    print(f"[INFO] Result: {payload_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
