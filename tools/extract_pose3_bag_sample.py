#!/usr/bin/env python3

"""Audit a ROS 2 bag and export initial LiDAR scans in the first base frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import gtsam
import numpy as np
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indoor_localization_pose3"))

from pose3_common import (  # noqa: E402
    OdomSample,
    deskew_points,
    interpolate_pose,
    pointcloud2_xyz_time,
    pose3_from_components,
    pose3_from_odometry,
    pose_rpy_deg,
    stamp_ns,
    transform_points,
)


def pose_record(pose: gtsam.Pose3) -> dict[str, list[float]]:
    return {
        "xyz": np.asarray(pose.translation(), dtype=float).tolist(),
        "rpy_deg": pose_rpy_deg(pose).tolist(),
    }


def transform_record(transform) -> dict[str, object]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    norm = math.sqrt(
        rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2
    )
    if norm <= 1e-12:
        pose = gtsam.Pose3()
    else:
        pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(
                rotation.w / norm,
                rotation.x / norm,
                rotation.y / norm,
                rotation.z / norm,
            ),
            gtsam.Point3(translation.x, translation.y, translation.z),
        )
    return {
        "parent": transform.header.frame_id,
        "child": transform.child_frame_id,
        **pose_record(pose),
    }


def write_cloud(path: Path, points: np.ndarray, voxel_size: float) -> int:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(voxel_size)
    cloud.remove_non_finite_points()
    if not len(cloud.points):
        raise RuntimeError("No points remain after filtering")
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Could not write {path}")
    return len(cloud.points)


def bag_storage_identifier(bag: Path) -> str:
    metadata_path = bag / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    try:
        identifier = metadata["rosbag2_bagfile_information"]["storage_identifier"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Could not determine the storage plugin from {metadata_path}"
        ) from error
    if not isinstance(identifier, str) or not identifier:
        raise RuntimeError(f"Invalid storage identifier in {metadata_path}")
    return identifier


def extract_full_odometry(
    bag: Path,
    storage_id: str,
    topic: str,
    message_type,
    output: Path,
) -> dict[str, object]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    count = 0
    first_pose: gtsam.Pose3 | None = None
    previous_pose: gtsam.Pose3 | None = None
    last_pose: gtsam.Pose3 | None = None
    first_stamp: int | None = None
    last_stamp: int | None = None
    path_length = 0.0
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    csv_path = output / "odometry_full.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("stamp_ns", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg")
        )
        while reader.has_next():
            _, serialized, _ = reader.read_next()
            message = deserialize_message(serialized, message_type)
            timestamp = stamp_ns(message)
            pose = pose3_from_odometry(message)
            xyz = np.asarray(pose.translation(), dtype=float)
            rpy = pose_rpy_deg(pose)
            writer.writerow((timestamp, *xyz.tolist(), *rpy.tolist()))
            if first_pose is None:
                first_pose = pose
                first_stamp = timestamp
            if previous_pose is not None:
                path_length += float(
                    np.linalg.norm(
                        np.asarray(
                            previous_pose.between(pose).translation(),
                            dtype=float,
                        )
                    )
                )
            previous_pose = pose
            last_pose = pose
            last_stamp = timestamp
            minimum = np.minimum(minimum, xyz)
            maximum = np.maximum(maximum, xyz)
            count += 1
    if first_pose is None or last_pose is None:
        raise RuntimeError(f"No messages found on {topic}")
    return {
        "samples": count,
        "first_stamp_ns": first_stamp,
        "last_stamp_ns": last_stamp,
        "duration_sec": (last_stamp - first_stamp) / 1e9,
        "first_pose": pose_record(first_pose),
        "last_pose": pose_record(last_pose),
        "xyz_min": minimum.tolist(),
        "xyz_max": maximum.tolist(),
        "path_length_3d_m": path_length,
        "csv": str(csv_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--odom-topic", default="/odometry/filtered")
    parser.add_argument("--lidar-topic", default="/ouster/points")
    parser.add_argument("--scan-count", type=int, default=5)
    parser.add_argument("--settle-sec", type=float, default=0.25)
    parser.add_argument("--voxel", type=float, default=0.08)
    parser.add_argument("--min-range", type=float, default=0.8)
    parser.add_argument("--max-range", type=float, default=30.0)
    parser.add_argument("--source-z-min", type=float, default=-1.2)
    parser.add_argument("--source-z-max", type=float, default=2.5)
    parser.add_argument("--point-time-field", default="t")
    parser.add_argument("--point-time-scale", type=float, default=1.0e-9)
    parser.add_argument("--deskew-bins", type=int, default=64)
    parser.add_argument("--odom-max-gap-sec", type=float, default=0.05)
    parser.add_argument(
        "--base-to-lidar",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=(0.0, 0.0, 0.60118, 0.0, 0.0, 180.0),
    )
    args = parser.parse_args()

    bag = args.bag.expanduser().resolve()
    if not (bag / "metadata.yaml").is_file():
        raise FileNotFoundError(f"metadata.yaml not found in {bag}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    storage_id = bag_storage_identifier(bag)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    required = {args.odom_topic, args.lidar_topic}
    missing = required.difference(topic_types)
    if missing:
        raise RuntimeError(f"Bag is missing topics: {', '.join(sorted(missing))}")
    selected_topics = [args.odom_topic, args.lidar_topic]
    if "/tf_static" in topic_types:
        selected_topics.append("/tf_static")
    reader.set_filter(rosbag2_py.StorageFilter(topics=selected_topics))
    message_types = {
        topic: get_message(topic_types[topic]) for topic in selected_topics
    }

    odometry: list[OdomSample] = []
    scans = []
    static_transforms: list[dict[str, object]] = []
    stop_after_ns: int | None = None
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        message = deserialize_message(serialized, message_types[topic])
        if topic == args.odom_topic:
            odometry.append(
                OdomSample(stamp_ns(message), pose3_from_odometry(message))
            )
        elif topic == "/tf_static":
            static_transforms.extend(
                transform_record(transform) for transform in message.transforms
            )
        elif topic == args.lidar_topic and len(scans) < args.scan_count:
            scans.append(message)
            if len(scans) == args.scan_count:
                stop_after_ns = stamp_ns(message) + int(args.settle_sec * 1e9)
        if (
            stop_after_ns is not None
            and odometry
            and odometry[-1].stamp_ns >= stop_after_ns
        ):
            break

    if not scans:
        raise RuntimeError("No PointCloud2 messages were read")
    if not odometry:
        raise RuntimeError("No odometry messages were read")
    odometry.sort(key=lambda sample: sample.stamp_ns)
    base_to_lidar = pose3_from_components(*args.base_to_lidar)

    first_valid_index = next(
        (
            index
            for index, scan in enumerate(scans)
            if interpolate_pose(
                odometry, stamp_ns(scan), args.odom_max_gap_sec
            )
            is not None
        ),
        None,
    )
    if first_valid_index is None:
        scan_range = (stamp_ns(scans[0]), stamp_ns(scans[-1]))
        odom_range = (odometry[0].stamp_ns, odometry[-1].stamp_ns)
        raise RuntimeError(
            "No synchronized odometry for the extracted scans; "
            f"scan stamps={scan_range}, odom stamps={odom_range}"
        )
    skipped_scans = first_valid_index
    scans = scans[first_valid_index:]
    first_scan_ns = stamp_ns(scans[0])
    first_base = interpolate_pose(
        odometry, first_scan_ns, args.odom_max_gap_sec
    )
    assert first_base is not None

    accumulated: list[np.ndarray] = []
    scan_records = []
    fields = [field.name for field in scans[0].fields]
    for index, scan in enumerate(scans):
        points, relative_times = pointcloud2_xyz_time(
            scan, args.point_time_field
        )
        scan_start_ns = stamp_ns(scan)
        corrected, reference_ns, deskewed, deskew_status = deskew_points(
            points,
            relative_times,
            scan_start_ns,
            odometry,
            base_to_lidar,
            time_scale=args.point_time_scale,
            reference="start",
            bins=args.deskew_bins,
            max_odom_gap_sec=args.odom_max_gap_sec,
        )
        reference_base = interpolate_pose(
            odometry, reference_ns, args.odom_max_gap_sec
        )
        if reference_base is None:
            reference_base = interpolate_pose(
                odometry, scan_start_ns, args.odom_max_gap_sec
            )
        if reference_base is None:
            raise RuntimeError(f"No synchronized odometry for scan {index}")

        ranges = np.linalg.norm(corrected, axis=1)
        keep = (
            (ranges >= args.min_range)
            & (ranges <= args.max_range)
            & (corrected[:, 2] >= args.source_z_min)
            & (corrected[:, 2] <= args.source_z_max)
        )
        filtered = corrected[keep]
        first_base_to_lidar = first_base.inverse().compose(
            reference_base.compose(base_to_lidar)
        )
        accumulated.append(transform_points(filtered, first_base_to_lidar))
        scan_records.append(
            {
                "index": index,
                "stamp_ns": scan_start_ns,
                "frame_id": scan.header.frame_id,
                "width": int(scan.width),
                "height": int(scan.height),
                "raw_points": int(len(points)),
                "filtered_points": int(len(filtered)),
                "deskewed": deskewed,
                "deskew_status": deskew_status,
                "odom_pose": pose_record(reference_base),
            }
        )

    cloud_points = np.concatenate(accumulated, axis=0)
    cloud_path = output / "initial_scans_first_base.ply"
    written_points = write_cloud(cloud_path, cloud_points, args.voxel)
    full_odometry = extract_full_odometry(
        bag,
        storage_id,
        args.odom_topic,
        message_types[args.odom_topic],
        output,
    )
    metadata = {
        "bag": str(bag),
        "bag_preserved": True,
        "storage_identifier": storage_id,
        "topics": {topic: topic_types[topic] for topic in selected_topics},
        "point_fields": fields,
        "base_to_lidar": pose_record(base_to_lidar),
        "first_base_odom_pose": pose_record(first_base),
        "odometry_samples_read": len(odometry),
        "leading_scans_without_odometry": skipped_scans,
        "static_transforms": static_transforms,
        "scans": scan_records,
        "output_frame": "first base_link pose",
        "output_voxel": args.voxel,
        "output_points": written_points,
        "output_ply": str(cloud_path),
        "full_odometry": full_odometry,
    }
    metadata_path = output / "bag_sample.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
