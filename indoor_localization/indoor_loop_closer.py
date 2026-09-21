#!/usr/bin/env python3
"""Conservative 2D LiDAR loop-closure detection and ICP verification."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from indoor_scan_matcher import wrap_angle


@dataclass(frozen=True)
class LoopMatch:
    candidate_index: int
    relative_x: float
    relative_y: float
    relative_yaw: float
    rmse: float
    inlier_ratio: float
    correspondences: int
    correction_translation: float
    correction_yaw_deg: float


@dataclass(frozen=True)
class LoopKeyframe:
    index: int
    stamp_ns: int
    points: np.ndarray


def compose_pose2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine = math.cos(float(first[2]))
    sine = math.sin(float(first[2]))
    return np.asarray(
        [
            first[0] + cosine * second[0] - sine * second[1],
            first[1] + sine * second[0] + cosine * second[1],
            wrap_angle(first[2] + second[2]),
        ],
        dtype=float,
    )


def between_pose2(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dx = float(second[0] - first[0])
    dy = float(second[1] - first[1])
    cosine = math.cos(float(first[2]))
    sine = math.sin(float(first[2]))
    return np.asarray(
        [
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            wrap_angle(second[2] - first[2]),
        ],
        dtype=float,
    )


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    return np.asarray(points, dtype=float) @ rotation.T + pose[:2]


def voxel_downsample_2d(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0 or len(points) == 0:
        return np.asarray(points, dtype=float)
    cells = np.floor(np.asarray(points, dtype=float) / voxel_size).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return np.asarray(points, dtype=float)[np.sort(indices)]


def rigid_alignment_2d(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return np.asarray(
        [translation[0], translation[1], math.atan2(rotation[1, 0], rotation[0, 0])],
        dtype=float,
    )


def icp_2d(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    maximum_correspondence_distance: float,
    maximum_iterations: int = 20,
    trim_fraction: float = 0.85,
) -> LoopMatch | None:
    if len(source) < 20 or len(target) < 20:
        return None
    tree = cKDTree(target)
    estimate = np.asarray(initial, dtype=float).copy()
    for _ in range(maximum_iterations):
        transformed = transform_points(source, estimate)
        distances, indices = tree.query(transformed, k=1, workers=1)
        valid_indices = np.flatnonzero(distances <= maximum_correspondence_distance)
        if len(valid_indices) < 20:
            return None
        keep = max(20, int(len(valid_indices) * trim_fraction))
        order = np.argsort(distances[valid_indices])[:keep]
        selected = valid_indices[order]
        delta = rigid_alignment_2d(transformed[selected], target[indices[selected]])
        estimate = compose_pose2(delta, estimate)
        if np.linalg.norm(delta[:2]) < 1e-4 and abs(delta[2]) < math.radians(0.01):
            break

    transformed = transform_points(source, estimate)
    distances, _ = tree.query(transformed, k=1, workers=1)
    inliers = distances <= maximum_correspondence_distance
    count = int(np.count_nonzero(inliers))
    if count < 20:
        return None
    correction = between_pose2(np.asarray(initial, dtype=float), estimate)
    return LoopMatch(
        candidate_index=-1,
        relative_x=float(estimate[0]),
        relative_y=float(estimate[1]),
        relative_yaw=float(estimate[2]),
        rmse=float(np.sqrt(np.mean(np.square(distances[inliers])))),
        inlier_ratio=float(np.mean(inliers)),
        correspondences=count,
        correction_translation=float(np.linalg.norm(correction[:2])),
        correction_yaw_deg=abs(math.degrees(float(correction[2]))),
    )


class LoopCloser:
    def __init__(
        self,
        *,
        enabled: bool,
        check_interval: int,
        minimum_separation: int,
        candidate_radius: float,
        maximum_candidates: int,
        submap_half_width: int,
        voxel_size: float,
        maximum_correspondence_distance: float,
        minimum_inlier_ratio: float,
        maximum_rmse: float,
        minimum_correspondences: int,
        maximum_correction: float,
        maximum_correction_yaw_deg: float,
        confirmation_count: int,
        confirmation_index_tolerance: int,
    ) -> None:
        self.enabled = enabled
        self.check_interval = max(1, check_interval)
        self.minimum_separation = max(1, minimum_separation)
        self.candidate_radius = candidate_radius
        self.maximum_candidates = max(1, maximum_candidates)
        self.submap_half_width = max(0, submap_half_width)
        self.voxel_size = voxel_size
        self.maximum_correspondence_distance = maximum_correspondence_distance
        self.minimum_inlier_ratio = minimum_inlier_ratio
        self.maximum_rmse = maximum_rmse
        self.minimum_correspondences = minimum_correspondences
        self.maximum_correction = maximum_correction
        self.maximum_correction_yaw_deg = maximum_correction_yaw_deg
        self.confirmation_count = max(1, confirmation_count)
        self.confirmation_index_tolerance = max(0, confirmation_index_tolerance)
        self.keyframes: dict[int, LoopKeyframe] = {}
        self.closed_pairs: set[tuple[int, int]] = set()
        self.pending_candidate_index: int | None = None
        self.pending_confirmations = 0

    def add_keyframe(self, index: int, stamp_ns: int, points: np.ndarray) -> None:
        self.keyframes[index] = LoopKeyframe(
            index=index,
            stamp_ns=stamp_ns,
            points=voxel_downsample_2d(points, self.voxel_size),
        )

    def find_loop(
        self,
        current_index: int,
        current_points: np.ndarray,
        current_pose: np.ndarray,
        optimized_poses: dict[int, np.ndarray],
    ) -> LoopMatch | None:
        if (
            not self.enabled
            or current_index < self.minimum_separation
            or current_index % self.check_interval != 0
        ):
            return None

        candidates: list[tuple[float, int]] = []
        for index, pose in optimized_poses.items():
            if current_index - index < self.minimum_separation:
                continue
            if (index, current_index) in self.closed_pairs:
                continue
            distance = float(np.linalg.norm(current_pose[:2] - pose[:2]))
            if distance <= self.candidate_radius:
                candidates.append((distance, index))
        candidates.sort()

        source = voxel_downsample_2d(current_points, self.voxel_size)
        accepted: list[LoopMatch] = []
        for _, candidate_index in candidates[: self.maximum_candidates]:
            candidate_pose = optimized_poses[candidate_index]
            target_parts = []
            for index in range(
                candidate_index - self.submap_half_width,
                candidate_index + self.submap_half_width + 1,
            ):
                record = self.keyframes.get(index)
                pose = optimized_poses.get(index)
                if record is None or pose is None:
                    continue
                target_parts.append(
                    transform_points(record.points, between_pose2(candidate_pose, pose))
                )
            if not target_parts:
                continue
            target = voxel_downsample_2d(np.vstack(target_parts), self.voxel_size)
            initial = between_pose2(candidate_pose, current_pose)
            result = icp_2d(
                source,
                target,
                initial,
                self.maximum_correspondence_distance,
            )
            if result is None:
                continue
            result = LoopMatch(candidate_index=candidate_index, **{
                field: getattr(result, field)
                for field in (
                    "relative_x",
                    "relative_y",
                    "relative_yaw",
                    "rmse",
                    "inlier_ratio",
                    "correspondences",
                    "correction_translation",
                    "correction_yaw_deg",
                )
            })
            if (
                result.inlier_ratio >= self.minimum_inlier_ratio
                and result.rmse <= self.maximum_rmse
                and result.correspondences >= self.minimum_correspondences
                and result.correction_translation <= self.maximum_correction
                and result.correction_yaw_deg <= self.maximum_correction_yaw_deg
            ):
                accepted.append(result)

        if not accepted:
            self.pending_candidate_index = None
            self.pending_confirmations = 0
            return None
        best = min(accepted, key=lambda match: match.rmse / max(match.inlier_ratio, 0.05))
        if self.confirmation_count > 1:
            same_region = (
                self.pending_candidate_index is not None
                and abs(best.candidate_index - self.pending_candidate_index)
                <= self.confirmation_index_tolerance
            )
            if same_region:
                self.pending_confirmations += 1
            else:
                self.pending_candidate_index = best.candidate_index
                self.pending_confirmations = 1
            if self.pending_confirmations < self.confirmation_count:
                return None
            self.pending_candidate_index = None
            self.pending_confirmations = 0
        self.closed_pairs.add((best.candidate_index, current_index))
        return best
