#!/usr/bin/env python3

"""Prepare the pharmacy LiScan PLY as a fixed local-frame Pose3 ICP map."""

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


def crop_cloud(
    cloud: o3d.geometry.PointCloud,
    crop: dict[str, float],
) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    finite = np.isfinite(points).all(axis=1)
    keep = (
        finite
        & (points[:, 0] >= crop["x_min"])
        & (points[:, 0] <= crop["x_max"])
        & (points[:, 1] >= crop["y_min"])
        & (points[:, 1] <= crop["y_max"])
        & (points[:, 2] >= crop["z_min"])
        & (points[:, 2] <= crop["z_max"])
    )
    cropped = cloud.select_by_index(np.flatnonzero(keep).tolist())
    if not len(cropped.points):
        raise RuntimeError("The configured pharmacy crop contains no points")
    return cropped


def clean_cloud(
    cloud: o3d.geometry.PointCloud,
    clean_voxel: float,
) -> tuple[o3d.geometry.PointCloud, int]:
    downsampled = cloud.voxel_down_sample(clean_voxel)
    del cloud
    gc.collect()
    before_filter = len(downsampled.points)
    clean, _ = downsampled.remove_statistical_outlier(
        nb_neighbors=16,
        std_ratio=4.0,
    )
    del downsampled
    gc.collect()
    clean.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=clean_voxel * 3.0,
            max_nn=60,
        )
    )
    points = np.asarray(clean.points)
    clean.orient_normals_towards_camera_location(
        np.asarray([0.0, 0.0, float(points[:, 2].max() + 30.0)])
    )
    return clean, before_filter


def write_preview(cloud: o3d.geometry.PointCloud, output: Path) -> None:
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    if len(points) > 400_000:
        selected = np.random.default_rng(20261022).choice(
            len(points), 400_000, replace=False
        )
        points = points[selected]
        if colors is not None:
            colors = colors[selected]

    figure, axis = plt.subplots(figsize=(10, 14), dpi=160)
    if colors is None:
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=points[:, 2],
            cmap="turbo",
            s=0.12,
        )
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
    axis.set_title("Pharmacy - fixed 3D map for Pose3 ICP")
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
) -> dict[str, object]:
    result = cloud.voxel_down_sample(voxel_size)
    result.remove_non_finite_points()
    points = np.asarray(result.points)
    if not len(points):
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
        "point_count": int(len(points)),
        "colors_present": result.has_colors(),
        "normals_present": result.has_normals(),
        "bbox_map": bounds(points),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"[INFO] {voxel_size:.2f} m: {len(points)} points -> {ply_path}",
        flush=True,
    )
    return {
        "ply": str(ply_path),
        "metadata": str(json_path),
        "point_count": int(len(points)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original pharmacy LiScan PLY")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clean-voxel", type=float, default=0.03)
    parser.add_argument("--x-min", type=float, default=-55.0)
    parser.add_argument("--x-max", type=float, default=60.0)
    parser.add_argument("--y-min", type=float, default=-105.0)
    parser.add_argument("--y-max", type=float, default=22.0)
    parser.add_argument("--z-min", type=float, default=-0.30)
    parser.add_argument("--z-max", type=float, default=4.20)
    parser.add_argument("--name", default="pharmacy_3d_icp_map")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading source without modifying it: {source}", flush=True)
    cloud = o3d.io.read_point_cloud(str(source), print_progress=True)
    source_points = np.asarray(cloud.points)
    if not len(source_points):
        raise RuntimeError(f"Empty or unreadable point cloud: {source}")
    finite = np.isfinite(source_points).all(axis=1)
    source_bounds = bounds(source_points[finite])
    invalid_points = int(np.sum(~finite))

    crop = {
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
    }
    cropped = crop_cloud(cloud, crop)
    del cloud, source_points
    gc.collect()
    cropped_count = len(cropped.points)
    print(f"[INFO] Interior crop: {cropped_count} points", flush=True)

    clean, before_filter = clean_cloud(cropped, args.clean_voxel)
    print(
        f"[INFO] Clean points: {before_filter} -> {len(clean.points)}",
        flush=True,
    )
    common_metadata = {
        "purpose": "Fixed pharmacy 3D map for Pose3 scan-to-map ICP",
        "source": str(source),
        "source_preserved": True,
        "coordinate_frame": "LiScan local map (preserved)",
        "units": "metres",
        "up_axis": "Z",
        "source_bounds": source_bounds,
        "source_invalid_xyz": invalid_points,
        "crop_bounds": crop,
        "cropped_point_count": cropped_count,
        "clean_voxel": args.clean_voxel,
        "clean_statistical_filter": {"neighbors": 16, "std_ratio": 4.0},
        "applied_translation": [0.0, 0.0, 0.0],
        "geometry_policy": {
            "exterior_and_high_facades_removed": True,
            "floor_preserved": True,
            "walls_preserved": True,
            "ceiling_preserved": True,
            "reason_for_ceiling": "Constrains Z, roll and pitch during 6D ICP",
        },
    }
    products = [
        write_product(clean, output_dir, args.name, voxel, common_metadata)
        for voxel in (0.05, 0.10)
    ]
    preview_path = output_dir / f"{args.name}_preview.png"
    write_preview(clean.voxel_down_sample(0.10), preview_path)
    manifest = {
        "recommended_map": products[0]["ply"],
        "products": products,
        "preview": str(preview_path),
        **common_metadata,
    }
    manifest_path = output_dir / f"{args.name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[INFO] Preview: {preview_path}", flush=True)
    print(f"[INFO] Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
