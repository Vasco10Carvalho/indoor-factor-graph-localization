#!/usr/bin/env python3
"""Occupancy-grid loading and robust correlative 2D scan matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class MatchResult:
    x: float
    y: float
    yaw: float
    cost: float
    inlier_ratio: float
    mean_distance: float
    valid_ratio: float


class OccupancyMap:
    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path.resolve()
        metadata = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8"))
        image_path = Path(str(metadata["image"]))
        if not image_path.is_absolute():
            image_path = self.yaml_path.parent / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read occupancy image: {image_path}")

        self.image_path = image_path.resolve()
        self.image = image
        self.height, self.width = image.shape
        self.resolution = float(metadata["resolution"])
        origin = metadata.get("origin", [0.0, 0.0, 0.0])
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.origin_yaw = float(origin[2])
        if abs(self.origin_yaw) > 1e-9:
            raise ValueError("Rotated occupancy-grid origins are not supported yet")

        negate = int(metadata.get("negate", 0))
        occupied_threshold = float(metadata.get("occupied_thresh", 0.65))
        free_threshold = float(metadata.get("free_thresh", 0.196))
        probability = image.astype(np.float32) / 255.0
        if negate == 0:
            probability = 1.0 - probability
        occupied = probability >= occupied_threshold
        free = probability <= free_threshold
        self.occupied = occupied
        self.free = free
        self.unknown = ~(occupied | free)

        distance_input = (~occupied).astype(np.uint8)
        distance_pixels = cv2.distanceTransform(
            distance_input,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        self.distance_m = distance_pixels.astype(np.float32) * self.resolution
        rows, columns = np.nonzero(free)
        self.free_x = self.origin_x + (columns.astype(np.float64) + 0.5) * self.resolution
        self.free_y = (
            self.origin_y
            + (self.height - rows.astype(np.float64) - 0.5) * self.resolution
        )

    def distance_at(
        self,
        x: np.ndarray,
        y: np.ndarray,
        outside_distance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        columns = np.floor((x - self.origin_x) / self.resolution).astype(np.int64)
        rows = self.height - 1 - np.floor(
            (y - self.origin_y) / self.resolution
        ).astype(np.int64)
        valid = (
            (rows >= 0)
            & (rows < self.height)
            & (columns >= 0)
            & (columns < self.width)
        )
        result = np.full(x.shape, outside_distance, dtype=np.float32)
        result[valid] = self.distance_m[rows[valid], columns[valid]]
        return result, valid

    def occupancy_data(self) -> np.ndarray:
        data = np.full(self.image.shape, -1, dtype=np.int8)
        data[self.free] = 0
        data[self.occupied] = 100
        return np.flipud(data).reshape(-1)


def laser_scan_points(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    maximum_range: float,
    maximum_points: int,
) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float64)
    indices = np.arange(len(values), dtype=np.float64)
    angles = angle_min + indices * angle_increment
    valid = (
        np.isfinite(values)
        & (values >= range_min)
        & (values <= min(range_max, maximum_range))
    )
    values = values[valid]
    angles = angles[valid]
    points = np.column_stack((values * np.cos(angles), values * np.sin(angles)))
    if maximum_points > 0 and len(points) > maximum_points:
        selected = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
        points = points[selected]
    return points


def evaluate_candidates(
    occupancy_map: OccupancyMap,
    points: np.ndarray,
    candidates: np.ndarray,
    truncation_distance: float,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    costs = np.empty(len(candidates), dtype=np.float64)
    inlier_ratios = np.empty(len(candidates), dtype=np.float64)
    mean_distances = np.empty(len(candidates), dtype=np.float64)
    valid_ratios = np.empty(len(candidates), dtype=np.float64)
    px = points[:, 0][None, :]
    py = points[:, 1][None, :]
    for begin in range(0, len(candidates), chunk_size):
        end = min(begin + chunk_size, len(candidates))
        batch = candidates[begin:end]
        cosine = np.cos(batch[:, 2])[:, None]
        sine = np.sin(batch[:, 2])[:, None]
        world_x = batch[:, 0, None] + cosine * px - sine * py
        world_y = batch[:, 1, None] + sine * px + cosine * py
        distances, valid = occupancy_map.distance_at(
            world_x,
            world_y,
            truncation_distance,
        )
        clipped = np.minimum(distances, truncation_distance)
        costs[begin:end] = np.mean(clipped * clipped, axis=1)
        inlier_ratios[begin:end] = np.mean(distances <= 0.25, axis=1)
        mean_distances[begin:end] = np.mean(clipped, axis=1)
        valid_ratios[begin:end] = np.mean(valid, axis=1)
        costs[begin:end] += (1.0 - valid_ratios[begin:end]) * truncation_distance**2
    return costs, inlier_ratios, mean_distances, valid_ratios


def candidate_grid(
    center: np.ndarray,
    xy_window: float,
    xy_step: float,
    yaw_window: float,
    yaw_step: float,
) -> np.ndarray:
    dx = np.arange(-xy_window, xy_window + 0.5 * xy_step, xy_step)
    dy = np.arange(-xy_window, xy_window + 0.5 * xy_step, xy_step)
    da = np.arange(-yaw_window, yaw_window + 0.5 * yaw_step, yaw_step)
    gx, gy, ga = np.meshgrid(dx, dy, da, indexing="ij")
    return np.column_stack(
        (
            center[0] + gx.ravel(),
            center[1] + gy.ravel(),
            wrap_angle(center[2] + ga.ravel()),
        )
    )


def motion_prior_costs(
    candidates: np.ndarray,
    map_costs: np.ndarray,
    predicted_pose: np.ndarray,
    translation_weight: float,
    rotation_weight: float,
) -> np.ndarray:
    """Regularize ambiguous map matches towards the odometry prediction.

    The occupancy-map score is a mean squared distance in m².  The additional
    terms only break near ties when a corridor is weakly constrained along its
    longitudinal axis; a decisive map improvement can still override the
    motion prior.  Setting both weights to zero reproduces pure map scoring.
    """
    if translation_weight < 0.0 or rotation_weight < 0.0:
        raise ValueError("Motion-prior weights must be non-negative")
    predicted_pose = np.asarray(predicted_pose, dtype=float)
    delta_xy = candidates[:, :2] - predicted_pose[None, :2]
    delta_yaw = wrap_angle(candidates[:, 2] - predicted_pose[2])
    return (
        map_costs
        + translation_weight * np.sum(delta_xy * delta_xy, axis=1)
        + rotation_weight * np.square(delta_yaw)
    )


def local_scan_match(
    occupancy_map: OccupancyMap,
    points: np.ndarray,
    predicted_pose: np.ndarray,
    coarse_xy_window: float = 1.0,
    coarse_xy_step: float = 0.20,
    coarse_yaw_window_deg: float = 8.0,
    coarse_yaw_step_deg: float = 1.0,
    fine_xy_window: float = 0.20,
    fine_xy_step: float = 0.05,
    fine_yaw_window_deg: float = 1.0,
    fine_yaw_step_deg: float = 0.25,
    truncation_distance: float = 1.5,
    translation_prior_weight: float = 0.05,
    rotation_prior_weight: float = 0.20,
) -> MatchResult:
    if len(points) < 20:
        raise ValueError("Not enough valid laser points")
    predicted_pose = np.asarray(predicted_pose, dtype=float)
    coarse = candidate_grid(
        predicted_pose,
        coarse_xy_window,
        coarse_xy_step,
        math.radians(coarse_yaw_window_deg),
        math.radians(coarse_yaw_step_deg),
    )
    coarse_metrics = evaluate_candidates(
        occupancy_map,
        points,
        coarse,
        truncation_distance,
    )
    coarse_costs = motion_prior_costs(
        coarse,
        coarse_metrics[0],
        predicted_pose,
        translation_prior_weight,
        rotation_prior_weight,
    )
    best_coarse = coarse[int(np.argmin(coarse_costs))]
    fine = candidate_grid(
        best_coarse,
        fine_xy_window,
        fine_xy_step,
        math.radians(fine_yaw_window_deg),
        math.radians(fine_yaw_step_deg),
    )
    fine_metrics = evaluate_candidates(
        occupancy_map,
        points,
        fine,
        truncation_distance,
    )
    fine_costs = motion_prior_costs(
        fine,
        fine_metrics[0],
        predicted_pose,
        translation_prior_weight,
        rotation_prior_weight,
    )
    index = int(np.argmin(fine_costs))
    pose = fine[index]
    return MatchResult(
        x=float(pose[0]),
        y=float(pose[1]),
        yaw=float(pose[2]),
        cost=float(fine_costs[index]),
        inlier_ratio=float(fine_metrics[1][index]),
        mean_distance=float(fine_metrics[2][index]),
        valid_ratio=float(fine_metrics[3][index]),
    )


def global_scan_match(
    occupancy_map: OccupancyMap,
    points: np.ndarray,
    particle_count: int = 30000,
    seed: int = 7,
    truncation_distance: float = 1.5,
) -> MatchResult:
    if len(points) < 20:
        raise ValueError("Not enough valid laser points")
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(occupancy_map.free_x), size=particle_count)
    candidates = np.column_stack(
        (
            occupancy_map.free_x[choices],
            occupancy_map.free_y[choices],
            rng.uniform(-math.pi, math.pi, particle_count),
        )
    )
    metrics = evaluate_candidates(
        occupancy_map,
        points,
        candidates,
        truncation_distance,
        chunk_size=256,
    )
    elite_count = min(80, particle_count)
    elite_indices = np.argpartition(metrics[0], elite_count - 1)[:elite_count]
    elite_indices = elite_indices[np.argsort(metrics[0][elite_indices])]
    elite = candidates[elite_indices]
    refined = []
    for center in elite[:20]:
        refined.append(
            candidate_grid(
                center,
                xy_window=0.6,
                xy_step=0.15,
                yaw_window=math.radians(6.0),
                yaw_step=math.radians(1.5),
            )
        )
    candidates = np.vstack(refined)
    metrics = evaluate_candidates(
        occupancy_map,
        points,
        candidates,
        truncation_distance,
    )
    index = int(np.argmin(metrics[0]))
    coarse_pose = candidates[index]
    result = local_scan_match(
        occupancy_map,
        points,
        coarse_pose,
        coarse_xy_window=0.3,
        coarse_xy_step=0.1,
        coarse_yaw_window_deg=3.0,
        coarse_yaw_step_deg=0.5,
    )
    return result
