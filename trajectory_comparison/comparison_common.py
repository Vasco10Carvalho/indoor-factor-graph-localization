#!/usr/bin/env python3
"""Data loading, SE(2) alignment, and map helpers for trajectory comparison."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class Trajectory:
    label: str
    stamp_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw_rad: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(self.stamp_ns), len(self.x), len(self.y), len(self.yaw_rad)}
        if len(lengths) != 1:
            raise ValueError(f"Trajectory {self.label!r} contains unequal column lengths")
        if not len(self.x):
            raise ValueError(f"Trajectory {self.label!r} contains no poses")


@dataclass(frozen=True)
class MapMetadata:
    yaml_path: Path
    image_path: Path
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    negate: bool
    occupied_thresh: float
    free_thresh: float


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_to_euler(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _first_present(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    raise KeyError(f"None of these columns is populated: {', '.join(names)}")


def load_trajectory(path: Path, label: str | None = None) -> Trajectory:
    """Load either a Factor Graph CSV or this tool's canonical CSV."""
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No poses found in {path}")

    stamps: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    yaws: list[float] = []
    for index, row in enumerate(rows):
        try:
            stamp_text = _first_present(row, ("stamp_ns", "timestamp_ns", "time_ns"))
            x_text = _first_present(row, ("map_x", "x", "aligned_x"))
            y_text = _first_present(row, ("map_y", "y", "aligned_y"))
            if row.get("yaw_rad") not in (None, ""):
                yaw = float(row["yaw_rad"])
            elif row.get("aligned_yaw_rad") not in (None, ""):
                yaw = float(row["aligned_yaw_rad"])
            else:
                yaw = math.radians(float(_first_present(row, ("yaw_deg",))))
            values = (float(x_text), float(y_text), yaw)
            if not all(math.isfinite(value) for value in values):
                continue
            try:
                stamp_ns = int(stamp_text)
            except ValueError:
                stamp_ns = int(float(stamp_text))
            stamps.append(stamp_ns)
            xs.append(values[0])
            ys.append(values[1])
            yaws.append(values[2])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid row {index + 2} in {path}: {error}") from error

    if not stamps:
        raise ValueError(f"No finite poses found in {path}")

    order = np.argsort(np.asarray(stamps, dtype=np.int64), kind="stable")
    stamp_array = np.asarray(stamps, dtype=np.int64)[order]
    x_array = np.asarray(xs, dtype=float)[order]
    y_array = np.asarray(ys, dtype=float)[order]
    yaw_array = np.asarray(yaws, dtype=float)[order]
    unique = np.concatenate(([True], stamp_array[1:] != stamp_array[:-1]))
    inferred_label = rows[0].get("source") or path.stem
    return Trajectory(
        label=label or inferred_label,
        stamp_ns=stamp_array[unique],
        x=x_array[unique],
        y=y_array[unique],
        yaw_rad=yaw_array[unique],
    )


def interpolate_pose(reference: Trajectory, stamp_ns: int) -> tuple[float, float, float]:
    if len(reference.stamp_ns) == 1:
        return float(reference.x[0]), float(reference.y[0]), float(reference.yaw_rad[0])
    clipped = int(np.clip(stamp_ns, reference.stamp_ns[0], reference.stamp_ns[-1]))
    yaw_unwrapped = np.unwrap(reference.yaw_rad)
    return (
        float(np.interp(clipped, reference.stamp_ns, reference.x)),
        float(np.interp(clipped, reference.stamp_ns, reference.y)),
        float(np.interp(clipped, reference.stamp_ns, yaw_unwrapped)),
    )


def align_initial_pose(trajectory: Trajectory, reference: Trajectory) -> Trajectory:
    """Rigidly align only the first method pose to the reference at that time."""
    ref_x, ref_y, ref_yaw = interpolate_pose(reference, int(trajectory.stamp_ns[0]))
    delta_yaw = float(wrap_angle(ref_yaw - trajectory.yaw_rad[0]))
    cosine = math.cos(delta_yaw)
    sine = math.sin(delta_yaw)
    local_x = trajectory.x - trajectory.x[0]
    local_y = trajectory.y - trajectory.y[0]
    aligned_x = ref_x + cosine * local_x - sine * local_y
    aligned_y = ref_y + sine * local_x + cosine * local_y
    aligned_yaw = wrap_angle(trajectory.yaw_rad + delta_yaw)
    return Trajectory(
        label=trajectory.label,
        stamp_ns=trajectory.stamp_ns.copy(),
        x=np.asarray(aligned_x),
        y=np.asarray(aligned_y),
        yaw_rad=np.asarray(aligned_yaw),
    )


def path_length(trajectory: Trajectory) -> float:
    if len(trajectory.x) < 2:
        return 0.0
    return float(np.hypot(np.diff(trajectory.x), np.diff(trajectory.y)).sum())


def trajectory_metrics(trajectory: Trajectory, reference: Trajectory) -> dict[str, object]:
    start_ns = max(int(trajectory.stamp_ns[0]), int(reference.stamp_ns[0]))
    end_ns = min(int(trajectory.stamp_ns[-1]), int(reference.stamp_ns[-1]))
    reference_mask = (reference.stamp_ns >= start_ns) & (reference.stamp_ns <= end_ns)
    errors = np.asarray([], dtype=float)
    yaw_errors = np.asarray([], dtype=float)
    sampled_path_length = 0.0
    comparison_samples = 0
    if end_ns >= start_ns and np.any(reference_mask):
        stamps = reference.stamp_ns[reference_mask]
        sampled_x = np.interp(stamps, trajectory.stamp_ns, trajectory.x)
        sampled_y = np.interp(stamps, trajectory.stamp_ns, trajectory.y)
        sampled_yaw = np.interp(stamps, trajectory.stamp_ns, np.unwrap(trajectory.yaw_rad))
        ref_x = reference.x[reference_mask]
        ref_y = reference.y[reference_mask]
        ref_yaw = np.unwrap(reference.yaw_rad)[reference_mask]
        errors = np.hypot(sampled_x - ref_x, sampled_y - ref_y)
        yaw_errors = np.abs(wrap_angle(sampled_yaw - ref_yaw))
        sampled_path_length = float(np.hypot(np.diff(sampled_x), np.diff(sampled_y)).sum())
        comparison_samples = int(len(stamps))

    def percentile(values: np.ndarray, q: float) -> float | None:
        return float(np.percentile(values, q)) if len(values) else None

    duration = max(0.0, (int(trajectory.stamp_ns[-1]) - int(trajectory.stamp_ns[0])) / 1e9)
    return {
        "poses": int(len(trajectory.x)),
        "start_stamp_ns": int(trajectory.stamp_ns[0]),
        "end_stamp_ns": int(trajectory.stamp_ns[-1]),
        "duration_s": duration,
        "path_length_m": sampled_path_length,
        "raw_path_length_m": path_length(trajectory),
        "comparison_samples": comparison_samples,
        "net_displacement_m": float(
            math.hypot(trajectory.x[-1] - trajectory.x[0], trajectory.y[-1] - trajectory.y[0])
        ),
        "return_distance_m": float(
            math.hypot(trajectory.x[-1] - reference.x[0], trajectory.y[-1] - reference.y[0])
        ),
        "common_duration_s": max(0.0, (end_ns - start_ns) / 1e9),
        "reference_position_error_m": {
            "median": percentile(errors, 50.0),
            "p95": percentile(errors, 95.0),
            "max": float(errors.max()) if len(errors) else None,
            "end": float(errors[-1]) if len(errors) else None,
        },
        "reference_yaw_error_deg": {
            "median": math.degrees(percentile(yaw_errors, 50.0)) if len(yaw_errors) else None,
            "p95": math.degrees(percentile(yaw_errors, 95.0)) if len(yaw_errors) else None,
            "max": math.degrees(float(yaw_errors.max())) if len(yaw_errors) else None,
            "end": math.degrees(float(yaw_errors[-1])) if len(yaw_errors) else None,
        },
    }


def load_map_metadata(yaml_path: Path) -> MapMetadata:
    yaml_path = yaml_path.expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    image_path = Path(data["image"]).expanduser()
    if not image_path.is_absolute():
        image_path = (yaml_path.parent / image_path).resolve()
    origin = data.get("origin", [0.0, 0.0, 0.0])
    return MapMetadata(
        yaml_path=yaml_path,
        image_path=image_path,
        resolution=float(data["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        negate=bool(int(data.get("negate", 0))),
        occupied_thresh=float(data.get("occupied_thresh", 0.65)),
        free_thresh=float(data.get("free_thresh", 0.196)),
    )
