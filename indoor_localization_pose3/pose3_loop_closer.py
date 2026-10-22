#!/usr/bin/env python3

"""Conservative Point-to-Plane ICP loop closures for Pose3 keyframes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gtsam
import numpy as np
import open3d as o3d

from pose3_common import rotation_angle_deg, transform_points


@dataclass(frozen=True)
class LoopKeyframe:
    index: int
    stamp_ns: int
    lidar_points: np.ndarray


@dataclass(frozen=True)
class LoopMatch:
    candidate_index: int
    relative_base_pose: gtsam.Pose3
    fitness: float
    rmse: float
    correspondences: int
    correction_translation: float
    correction_rotation_deg: float


class Pose3LoopCloser:
    def __init__(self, args, base_to_lidar: gtsam.Pose3) -> None:
        self.args = args
        self.base_to_lidar = base_to_lidar
        self.lidar_to_base = base_to_lidar.inverse()
        self.keyframes: dict[int, LoopKeyframe] = {}
        self.closed_pairs: set[tuple[int, int]] = set()
        self.pending_candidate: int | None = None
        self.pending_confirmations = 0

    def add_keyframe(
        self,
        index: int,
        stamp_ns: int,
        lidar_points: np.ndarray,
    ) -> None:
        cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(lidar_points, dtype=float))
        )
        if self.args.loop_voxel > 0.0:
            cloud = cloud.voxel_down_sample(self.args.loop_voxel)
        points = np.asarray(cloud.points, dtype=np.float32)
        maximum = self.args.loop_max_points_per_keyframe
        if maximum > 0 and len(points) > maximum:
            rng = np.random.default_rng(self.args.random_seed + index)
            points = points[
                rng.choice(len(points), size=maximum, replace=False)
            ]
        self.keyframes[index] = LoopKeyframe(index, stamp_ns, points)

    def find_loop(
        self,
        current_index: int,
        current_points: np.ndarray,
        current_pose: gtsam.Pose3,
        optimized_poses: dict[int, gtsam.Pose3],
    ) -> LoopMatch | None:
        args = self.args
        if (
            not args.enable_loop_closure
            or current_index < args.loop_minimum_separation
            or current_index % args.loop_check_interval != 0
        ):
            return None

        current_xyz = np.asarray(current_pose.translation(), dtype=float)
        candidates: list[tuple[float, int]] = []
        for index, pose in optimized_poses.items():
            if current_index - index < args.loop_minimum_separation:
                continue
            if (index, current_index) in self.closed_pairs:
                continue
            distance = float(
                np.linalg.norm(
                    current_xyz - np.asarray(pose.translation(), dtype=float)
                )
            )
            if distance <= args.loop_candidate_radius:
                candidates.append((distance, index))
        candidates.sort()

        accepted: list[LoopMatch] = []
        for _, candidate_index in candidates[: args.loop_maximum_candidates]:
            match = self._verify(
                candidate_index,
                current_points,
                current_pose,
                optimized_poses,
            )
            if match is not None:
                accepted.append(match)
        if not accepted:
            self.pending_candidate = None
            self.pending_confirmations = 0
            return None

        best = min(
            accepted,
            key=lambda item: item.rmse / max(item.fitness, 0.05),
        )
        if args.loop_confirmation_count > 1:
            same_region = (
                self.pending_candidate is not None
                and abs(best.candidate_index - self.pending_candidate)
                <= args.loop_confirmation_index_tolerance
            )
            if same_region:
                self.pending_confirmations += 1
            else:
                self.pending_candidate = best.candidate_index
                self.pending_confirmations = 1
            if self.pending_confirmations < args.loop_confirmation_count:
                return None
            self.pending_candidate = None
            self.pending_confirmations = 0
        self.closed_pairs.add((best.candidate_index, current_index))
        return best

    def _verify(
        self,
        candidate_index: int,
        current_points: np.ndarray,
        current_pose: gtsam.Pose3,
        optimized_poses: dict[int, gtsam.Pose3],
    ) -> LoopMatch | None:
        args = self.args
        candidate_pose = optimized_poses.get(candidate_index)
        if candidate_pose is None:
            return None
        candidate_lidar = candidate_pose.compose(self.base_to_lidar)
        target_parts = []
        for index in range(
            candidate_index - args.loop_submap_half_width,
            candidate_index + args.loop_submap_half_width + 1,
        ):
            record = self.keyframes.get(index)
            pose = optimized_poses.get(index)
            if record is None or pose is None:
                continue
            candidate_lidar_to_lidar = candidate_lidar.inverse().compose(
                pose.compose(self.base_to_lidar)
            )
            target_parts.append(
                transform_points(record.lidar_points, candidate_lidar_to_lidar)
            )
        if not target_parts:
            return None

        source = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(current_points, dtype=float))
        )
        target = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.vstack(target_parts))
        )
        if args.loop_voxel > 0.0:
            source = source.voxel_down_sample(args.loop_voxel)
            target = target.voxel_down_sample(args.loop_voxel)
        if (
            len(source.points) < args.loop_minimum_correspondences
            or len(target.points) < args.loop_minimum_correspondences
        ):
            return None
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.loop_normal_radius,
                max_nn=args.loop_normal_max_nn,
            )
        )

        initial_base = candidate_pose.between(current_pose)
        initial_lidar = self.lidar_to_base.compose(initial_base).compose(
            self.base_to_lidar
        )
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=args.loop_icp_iterations
        )
        estimator = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            args.loop_max_correspondence,
            np.asarray(initial_lidar.matrix(), dtype=float),
            estimator,
            criteria,
        )
        correspondences = len(result.correspondence_set)
        measured_lidar = gtsam.Pose3(
            np.asarray(result.transformation, dtype=float)
        )
        correction = initial_lidar.between(measured_lidar)
        correction_translation = float(
            np.linalg.norm(np.asarray(correction.translation(), dtype=float))
        )
        correction_rotation = rotation_angle_deg(correction.rotation())
        if (
            result.fitness < args.loop_minimum_fitness
            or result.inlier_rmse > args.loop_maximum_rmse
            or correspondences < args.loop_minimum_correspondences
            or correction_translation > args.loop_maximum_correction
            or correction_rotation > args.loop_maximum_rotation_deg
        ):
            return None
        relative_base = self.base_to_lidar.compose(measured_lidar).compose(
            self.lidar_to_base
        )
        return LoopMatch(
            candidate_index=candidate_index,
            relative_base_pose=relative_base,
            fitness=float(result.fitness),
            rmse=float(result.inlier_rmse),
            correspondences=correspondences,
            correction_translation=correction_translation,
            correction_rotation_deg=correction_rotation,
        )


def loop_sigmas(match: LoopMatch, args) -> np.ndarray:
    quality = max(match.fitness, 0.05)
    translation = float(
        np.clip(
            match.rmse / math.sqrt(quality),
            args.loop_min_sigma_translation,
            args.loop_max_sigma_translation,
        )
    )
    rotation_deg = float(
        np.clip(
            1.0 + 8.0 * (1.0 - min(match.fitness, 1.0)) + 4.0 * match.rmse,
            args.loop_min_sigma_rotation_deg,
            args.loop_max_sigma_rotation_deg,
        )
    )
    return np.asarray(
        [
            math.radians(rotation_deg),
            math.radians(rotation_deg),
            math.radians(rotation_deg),
            translation,
            translation,
            translation,
        ],
        dtype=float,
    )
