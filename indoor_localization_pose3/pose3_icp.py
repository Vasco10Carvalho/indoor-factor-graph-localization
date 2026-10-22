#!/usr/bin/env python3

"""Fixed-map loading, local cropping and two-stage Point-to-Plane ICP."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_map(
    path: Path,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, cKDTree]:
    started = time.monotonic()
    print(f"[INFO] Loading fixed PLY: {path}", flush=True)
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise ValueError(f"PLY contains no points: {path}")
    original_count = len(cloud.points)
    if args.map_voxel > 0.0:
        cloud = cloud.voxel_down_sample(args.map_voxel)
    points = np.asarray(cloud.points, dtype=np.float64)
    mask = np.ones(len(points), dtype=bool)
    if args.map_z_min is not None:
        mask &= points[:, 2] >= args.map_z_min
    if args.map_z_max is not None:
        mask &= points[:, 2] <= args.map_z_max
    if not np.all(mask):
        cloud = cloud.select_by_index(np.flatnonzero(mask))
    if not cloud.has_normals():
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.map_normal_radius,
                max_nn=args.map_normal_max_nn,
            )
        )
    else:
        normals = np.asarray(cloud.normals)
        invalid = ~np.isfinite(normals).all(axis=1)
        invalid |= np.linalg.norm(normals, axis=1) < 1e-6
        if np.any(invalid):
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=args.map_normal_radius,
                    max_nn=args.map_normal_max_nn,
                )
            )
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    colors = (
        np.asarray(cloud.colors, dtype=np.float64)
        if cloud.has_colors()
        else None
    )
    if len(points) == 0:
        raise ValueError("Map filters removed every PLY point")
    print(
        f"[INFO] Fixed map ready: {original_count} -> {len(points)} points "
        f"in {time.monotonic() - started:.2f}s",
        flush=True,
    )
    return points, normals, colors, cKDTree(points[:, :2])


def crop_target_map(
    map_points: np.ndarray,
    map_normals: np.ndarray,
    tree_xy: cKDTree,
    center_xyz: np.ndarray,
    args,
) -> o3d.geometry.PointCloud:
    indices = tree_xy.query_ball_point(center_xyz[:2], r=args.crop_radius)
    if not indices:
        return o3d.geometry.PointCloud()
    selected_indices = np.asarray(indices, dtype=np.int64)
    selected = map_points[selected_indices]
    z_mask = (
        (selected[:, 2] >= center_xyz[2] - args.crop_z_margin_below)
        & (selected[:, 2] <= center_xyz[2] + args.crop_z_margin_above)
    )
    selected_indices = selected_indices[z_mask]
    if len(selected_indices) == 0:
        return o3d.geometry.PointCloud()
    if (
        args.max_target_points > 0
        and len(selected_indices) > args.max_target_points
    ):
        rng = np.random.default_rng(args.random_seed)
        selected_indices = rng.choice(
            selected_indices,
            size=args.max_target_points,
            replace=False,
        )
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(map_points[selected_indices])
    )
    cloud.normals = o3d.utility.Vector3dVector(map_normals[selected_indices])
    return cloud


def run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    args,
):
    estimator = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=args.icp_iterations
    )
    coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        args.coarse_max_correspondence,
        initial_transform,
        estimator,
        criteria,
    )
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        args.fine_max_correspondence,
        coarse.transformation,
        estimator,
        criteria,
    )
