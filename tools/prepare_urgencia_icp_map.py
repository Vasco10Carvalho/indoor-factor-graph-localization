#!/usr/bin/env python3
"""Prepare the Urgencia LiScan PLY as a fixed local-frame Pose3 ICP map."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402


def bounds(points: np.ndarray) -> dict[str, list[float]]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return {
        "min": minimum.astype(float).tolist(),
        "max": maximum.astype(float).tolist(),
        "dimensions": (maximum - minimum).astype(float).tolist(),
    }


def clean_in_local_frame(
    source: Path,
    clean_voxel: float,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, dict[str, list[float]], int]:
    print(f"[INFO] Loading source without modifying it: {source}", flush=True)
    cloud = o3d.io.read_point_cloud(str(source), print_progress=True)
    points = np.asarray(cloud.points)
    if not len(points):
        raise RuntimeError(f"Empty or unreadable point cloud: {source}")

    finite = np.isfinite(points).all(axis=1)
    source_bounds = bounds(points[finite])
    origin = np.asarray(
        [
            (source_bounds["min"][0] + source_bounds["max"][0]) * 0.5,
            (source_bounds["min"][1] + source_bounds["max"][1]) * 0.5,
            source_bounds["min"][2],
        ],
        dtype=float,
    )
    invalid_points = int(np.sum(~finite))
    if invalid_points:
        cloud.remove_non_finite_points()
    cloud.translate(-origin)

    print(f"[INFO] Clean voxel downsample: {clean_voxel:.3f} m", flush=True)
    clean_before = cloud.voxel_down_sample(clean_voxel)
    del cloud, points
    gc.collect()
    before_filter = len(clean_before.points)
    clean, _ = clean_before.remove_statistical_outlier(
        nb_neighbors=16,
        std_ratio=4.0,
    )
    del clean_before
    gc.collect()

    normal_radius = clean_voxel * 3.0
    clean.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=60)
    )
    clean_points = np.asarray(clean.points)
    clean.orient_normals_towards_camera_location(
        np.asarray([0.0, 0.0, float(clean_points[:, 2].max() + 30.0)])
    )
    print(
        f"[INFO] Clean points: {before_filter} -> {len(clean.points)}",
        flush=True,
    )
    return clean, origin, source_bounds, invalid_points


def crop_interior(
    cloud: o3d.geometry.PointCloud,
    crop: dict[str, float],
    floor_z: float,
) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    keep = (
        (points[:, 0] >= crop["x_min"])
        & (points[:, 0] <= crop["x_max"])
        & (points[:, 1] >= crop["y_min"])
        & (points[:, 1] <= crop["y_max"])
        & (points[:, 2] >= crop["z_min"])
        & (points[:, 2] <= crop["z_max"])
    )
    cropped = cloud.select_by_index(np.flatnonzero(keep).tolist())
    if not len(cropped.points):
        raise RuntimeError("The configured interior crop contains no points")
    cropped.translate((0.0, 0.0, -floor_z))
    print(f"[INFO] Interior points after crop: {len(cropped.points)}", flush=True)
    return cropped


def write_preview(cloud: o3d.geometry.PointCloud, output: Path) -> None:
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    if len(points) > 400_000:
        selected = np.random.default_rng(20261020).choice(
            len(points), 400_000, replace=False
        )
        points = points[selected]
        if colors is not None:
            colors = colors[selected]

    figure, axis = plt.subplots(figsize=(14, 8), dpi=160)
    if colors is None:
        axis.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.12)
    else:
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=np.clip(colors, 0.0, 1.0),
            s=0.12,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_title("Urgencia - fixed 3D map for Pose3 ICP")
    axis.grid(color="#dddddd", linewidth=0.35)
    figure.tight_layout()
    figure.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def write_product(
    cloud: o3d.geometry.PointCloud,
    output_dir: Path,
    name: str,
    voxel_size: float,
    common_metadata: dict,
) -> dict[str, str]:
    result = cloud.voxel_down_sample(voxel_size)
    result.remove_non_finite_points()
    result_points = np.asarray(result.points)
    if not len(result_points):
        raise RuntimeError(f"No points remain at voxel size {voxel_size}")

    suffix = str(voxel_size).replace(".", "p")
    ply_path = output_dir / f"{name}_{suffix}.ply"
    json_path = ply_path.with_suffix(".json")
    if not o3d.io.write_point_cloud(
        str(ply_path), result, write_ascii=False, compressed=False
    ):
        raise RuntimeError(f"Could not write {ply_path}")

    metadata = {
        **common_metadata,
        "voxel_size": voxel_size,
        "point_count": int(len(result_points)),
        "colors_present": result.has_colors(),
        "normals_present": result.has_normals(),
        "bbox_map": bounds(result_points),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"[INFO] {voxel_size:.2f} m: {len(result_points)} points -> {ply_path}",
        flush=True,
    )
    return {"ply": str(ply_path), "metadata": str(json_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original LiScan PLY")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clean-voxel", type=float, default=0.03)
    parser.add_argument("--x-min", type=float, default=-55.0)
    parser.add_argument("--x-max", type=float, default=55.0)
    parser.add_argument("--y-min", type=float, default=12.0)
    parser.add_argument("--y-max", type=float, default=70.0)
    parser.add_argument("--z-min", type=float, default=2.15)
    parser.add_argument("--z-max", type=float, default=4.95)
    parser.add_argument("--floor-z", type=float, default=2.3175)
    parser.add_argument("--name", default="urgencia_3d_icp_map")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean, origin, source_bounds, invalid_points = clean_in_local_frame(
        source,
        args.clean_voxel,
    )
    crop = {
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
    }
    interior = crop_interior(clean, crop, args.floor_z)
    del clean
    gc.collect()

    applied_translation = -origin
    applied_translation[2] -= args.floor_z
    common_metadata = {
        "purpose": "Fixed 3D map for Pose3 scan-to-map ICP",
        "source": str(source),
        "source_preserved": True,
        "coordinate_frame": "map",
        "units": "metres",
        "up_axis": "Z",
        "source_bounds": source_bounds,
        "source_invalid_xyz": invalid_points,
        "source_local_origin": origin.tolist(),
        "clean_voxel": args.clean_voxel,
        "clean_statistical_filter": {"neighbors": 16, "std_ratio": 4.0},
        "floor_z_before_shift": args.floor_z,
        "applied_translation": applied_translation.tolist(),
        "crop_bounds_before_floor_shift": crop,
        "geometry_policy": {
            "exterior_removed": True,
            "floor_preserved": True,
            "walls_preserved": True,
            "ceiling_preserved": True,
            "reason_for_ceiling": "Constrains Z, roll and pitch during 6D ICP",
        },
    }
    products = [
        write_product(interior, output_dir, args.name, voxel, common_metadata)
        for voxel in (0.05, 0.10)
    ]
    preview_path = output_dir / f"{args.name}_preview.png"
    write_preview(interior.voxel_down_sample(0.10), preview_path)
    manifest = {
        "recommended_map": products[0]["ply"],
        "realtime_light_map": products[1]["ply"],
        "preview": str(preview_path),
        "products": products,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
