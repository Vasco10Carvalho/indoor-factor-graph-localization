#!/usr/bin/env python3

"""Shared SE(3), PointCloud2 and ROS helpers for indoor Pose3 localization."""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gtsam
import numpy as np
from sensor_msgs_py import point_cloud2


@dataclass(frozen=True)
class OdomSample:
    stamp_ns: int
    pose: gtsam.Pose3


@dataclass(frozen=True)
class Pose2AidSample:
    stamp_ns: int
    x: float
    y: float
    yaw_deg: float


def load_pose2_aid_csv(path: Path) -> tuple[Pose2AidSample, ...]:
    """Load a timestamped Pose2 trajectory produced by the indoor pipeline."""

    samples: list[Pose2AidSample] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"stamp_ns", "map_x", "map_y", "yaw_deg"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Pose2 aid CSV is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            samples.append(
                Pose2AidSample(
                    stamp_ns=int(row["stamp_ns"]),
                    x=float(row["map_x"]),
                    y=float(row["map_y"]),
                    yaw_deg=float(row["yaw_deg"]),
                )
            )
    if not samples:
        raise ValueError(f"Pose2 aid CSV contains no poses: {path}")
    samples.sort(key=lambda sample: sample.stamp_ns)
    return tuple(samples)


def nearest_pose2_aid(
    samples: Sequence[Pose2AidSample],
    stamps: Sequence[int],
    target_ns: int,
    max_difference_sec: float,
) -> Pose2AidSample | None:
    """Return the closest Pose2 aid pose within the synchronization window."""

    if not samples:
        return None
    index = bisect.bisect_left(stamps, target_ns)
    candidates = []
    if index < len(samples):
        candidates.append(samples[index])
    if index > 0:
        candidates.append(samples[index - 1])
    best = min(candidates, key=lambda sample: abs(sample.stamp_ns - target_ns))
    if abs(best.stamp_ns - target_ns) / 1e9 > max_difference_sec:
        return None
    return best


def stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def set_stamp(message_stamp, value_ns: int) -> None:
    message_stamp.sec = int(value_ns // 1_000_000_000)
    message_stamp.nanosec = int(value_ns % 1_000_000_000)


def keyframe_due(
    last_keyframe_ns: int | None,
    candidate_ns: int,
    period_ns: int,
) -> bool:
    """Return whether a scan timestamp is due for keyframe processing."""

    return (
        last_keyframe_ns is None
        or candidate_ns - last_keyframe_ns >= period_ns
    )


def map_visualization_mask(
    points: np.ndarray,
    z_min: float | None,
    z_max: float | None,
) -> np.ndarray:
    """Select the map slice shown in RViz without changing the ICP map."""

    mask = np.ones(len(points), dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    return mask


def pose3_from_components(
    x: float,
    y: float,
    z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> gtsam.Pose3:
    rotation = gtsam.Rot3.Ypr(
        math.radians(yaw_deg),
        math.radians(pitch_deg),
        math.radians(roll_deg),
    )
    return gtsam.Pose3(rotation, gtsam.Point3(x, y, z))


def pose3_from_odometry(message) -> gtsam.Pose3:
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    norm = math.sqrt(
        orientation.x**2
        + orientation.y**2
        + orientation.z**2
        + orientation.w**2
    )
    if norm <= 1e-12:
        rotation = gtsam.Rot3()
    else:
        rotation = gtsam.Rot3.Quaternion(
            orientation.w / norm,
            orientation.x / norm,
            orientation.y / norm,
            orientation.z / norm,
        )
    return gtsam.Pose3(
        rotation,
        gtsam.Point3(position.x, position.y, position.z),
    )


def pose_rpy_deg(pose: gtsam.Pose3) -> np.ndarray:
    return np.degrees(np.asarray(pose.rotation().rpy(), dtype=float))


def rotation_angle_deg(rotation: gtsam.Rot3) -> float:
    tangent = np.asarray(gtsam.Rot3.Logmap(rotation), dtype=float)
    return math.degrees(float(np.linalg.norm(tangent)))


def odometry_interval_is_continuous(
    samples: Sequence[OdomSample],
    start: OdomSample,
    end: OdomSample,
    maximum_step: float,
    maximum_rotation_deg: float,
) -> bool:
    """Validate a large keyframe delta using the intervening odometry samples."""

    if end.stamp_ns <= start.stamp_ns:
        return False
    interval = [start]
    interval.extend(
        sample
        for sample in samples
        if start.stamp_ns < sample.stamp_ns < end.stamp_ns
    )
    interval.append(end)
    if len(interval) <= 2:
        return False

    for previous, current in zip(interval, interval[1:]):
        delta = previous.pose.between(current.pose)
        translation = float(
            np.linalg.norm(np.asarray(delta.translation(), dtype=float))
        )
        rotation = rotation_angle_deg(delta.rotation())
        if (
            translation > maximum_step
            or rotation > maximum_rotation_deg
        ):
            return False
    return True


def robust_noise(sigmas: np.ndarray, loss: str, parameter: float):
    diagonal = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(sigmas, dtype=float)
    )
    if parameter <= 0.0 or loss == "none":
        return diagonal
    if loss == "huber":
        estimator = gtsam.noiseModel.mEstimator.Huber.Create(float(parameter))
    elif loss == "dcs":
        estimator = gtsam.noiseModel.mEstimator.DCS.Create(float(parameter))
    else:
        raise ValueError(f"Unknown robust loss: {loss}")
    return gtsam.noiseModel.Robust.Create(estimator, diagonal)


def interpolate_pose(
    samples: Sequence[OdomSample],
    target_ns: int,
    max_gap_sec: float,
) -> gtsam.Pose3 | None:
    """Interpolate an odometry pose on SE(3) at a requested timestamp."""

    if not samples:
        return None
    stamps = [sample.stamp_ns for sample in samples]
    index = bisect.bisect_left(stamps, target_ns)
    if index < len(samples) and samples[index].stamp_ns == target_ns:
        return samples[index].pose
    if index == 0 or index == len(samples):
        nearest = samples[0] if index == 0 else samples[-1]
        if abs(nearest.stamp_ns - target_ns) / 1e9 <= max_gap_sec:
            return nearest.pose
        return None
    before = samples[index - 1]
    after = samples[index]
    if max(
        target_ns - before.stamp_ns,
        after.stamp_ns - target_ns,
    ) / 1e9 > max_gap_sec:
        return None
    interval = after.stamp_ns - before.stamp_ns
    if interval <= 0:
        return before.pose
    alpha = float(target_ns - before.stamp_ns) / float(interval)
    relative = before.pose.between(after.pose)
    tangent = np.asarray(gtsam.Pose3.Logmap(relative), dtype=float)
    return before.pose.compose(gtsam.Pose3.Expmap(alpha * tangent))


def nearest_odom(
    samples: Sequence[OdomSample],
    target_ns: int,
    max_difference_sec: float,
) -> OdomSample | None:
    if not samples:
        return None
    stamps = [sample.stamp_ns for sample in samples]
    index = bisect.bisect_left(stamps, target_ns)
    candidates = []
    if index < len(samples):
        candidates.append(samples[index])
    if index > 0:
        candidates.append(samples[index - 1])
    best = min(candidates, key=lambda sample: abs(sample.stamp_ns - target_ns))
    if abs(best.stamp_ns - target_ns) / 1e9 > max_difference_sec:
        return None
    return best


def pointcloud2_xyz_time(message, time_field: str = "t"):
    """Return XYZ and optional per-point time values without assuming a layout."""

    available = {field.name for field in message.fields}
    names = ("x", "y", "z", time_field) if time_field in available else (
        "x",
        "y",
        "z",
    )
    records = point_cloud2.read_points(
        message,
        field_names=names,
        skip_nans=True,
    )
    records = np.asarray(records)
    if records.dtype.names:
        points = np.column_stack(
            (records["x"], records["y"], records["z"])
        ).astype(np.float64, copy=False)
        times = (
            np.asarray(records[time_field], dtype=np.float64)
            if time_field in records.dtype.names
            else None
        )
    else:
        values = np.asarray(records)
        if values.ndim != 2:
            values = values.reshape((-1, len(names)))
        points = np.asarray(values[:, :3], dtype=np.float64)
        times = (
            np.asarray(values[:, 3], dtype=np.float64)
            if values.shape[1] > 3
            else None
        )
    finite = np.isfinite(points).all(axis=1)
    if times is not None:
        finite &= np.isfinite(times)
    return points[finite], times[finite] if times is not None else None


def deskew_points(
    points: np.ndarray,
    relative_times: np.ndarray | None,
    scan_start_ns: int,
    odom_samples: Sequence[OdomSample],
    base_to_lidar: gtsam.Pose3,
    *,
    time_scale: float,
    reference: str,
    bins: int,
    max_odom_gap_sec: float,
) -> tuple[np.ndarray, int, bool, str]:
    """Deskew a scan into its start or end LiDAR frame using odometry."""

    if relative_times is None or len(points) == 0:
        return points, scan_start_ns, False, "no_point_time_field"
    relative_seconds = relative_times * time_scale
    if not np.isfinite(relative_seconds).all():
        return points, scan_start_ns, False, "invalid_point_times"
    minimum = float(np.min(relative_seconds))
    maximum = float(np.max(relative_seconds))
    duration = maximum - minimum
    if duration <= 1e-5 or duration > 1.0:
        return points, scan_start_ns, False, "invalid_scan_duration"

    start_ns = scan_start_ns + int(round(minimum * 1e9))
    end_ns = scan_start_ns + int(round(maximum * 1e9))
    reference_ns = start_ns if reference == "start" else end_ns
    reference_base = interpolate_pose(
        odom_samples,
        reference_ns,
        max_odom_gap_sec,
    )
    if reference_base is None:
        return points, reference_ns, False, "missing_reference_odometry"
    reference_lidar = reference_base.compose(base_to_lidar)

    bin_count = max(2, int(bins))
    normalized = np.clip((relative_seconds - minimum) / duration, 0.0, 1.0)
    bin_indices = np.minimum(
        (normalized * bin_count).astype(np.int32),
        bin_count - 1,
    )
    output = points.copy()
    for bin_index in np.unique(bin_indices):
        alpha = (float(bin_index) + 0.5) / float(bin_count)
        point_stamp_ns = start_ns + int(round(alpha * (end_ns - start_ns)))
        point_base = interpolate_pose(
            odom_samples,
            point_stamp_ns,
            max_odom_gap_sec,
        )
        if point_base is None:
            return points, reference_ns, False, "missing_deskew_odometry"
        reference_lidar_to_point_lidar = reference_lidar.inverse().compose(
            point_base.compose(base_to_lidar)
        )
        indices = np.flatnonzero(bin_indices == bin_index)
        rotation = np.asarray(
            reference_lidar_to_point_lidar.rotation().matrix(),
            dtype=float,
        )
        translation = np.asarray(
            reference_lidar_to_point_lidar.translation(),
            dtype=float,
        )
        output[indices] = points[indices] @ rotation.T + translation
    return output, reference_ns, True, "applied"


def transform_points(points: np.ndarray, pose: gtsam.Pose3) -> np.ndarray:
    rotation = np.asarray(pose.rotation().matrix(), dtype=float)
    translation = np.asarray(pose.translation(), dtype=float)
    return points @ rotation.T + translation
