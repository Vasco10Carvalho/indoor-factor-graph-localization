#!/usr/bin/env python3

"""Republish every saved Pose3 trajectory and an RViz-only map slice."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gtsam
import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from pose3_common import map_visualization_mask, pose3_from_components


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "stamp_ns",
            "map_x",
            "map_y",
            "map_z",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "odom_x",
            "odom_y",
            "odom_z",
            "odom_roll_deg",
            "odom_pitch_deg",
            "odom_yaw_deg",
            "predicted_x",
            "predicted_y",
            "predicted_z",
            "predicted_roll_deg",
            "predicted_pitch_deg",
            "predicted_yaw_deg",
            "scan_x",
            "scan_y",
            "scan_z",
            "scan_roll_deg",
            "scan_pitch_deg",
            "scan_yaw_deg",
            "pose2_aid_used",
            "pose2_aid_x",
            "pose2_aid_y",
            "pose2_aid_yaw_deg",
            "icp_accepted",
            "loop_accepted",
            "loop_candidate",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Pose3 CSV is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Pose3 CSV contains no poses: {path}")
    return rows


def pose_from_row(
    row: dict[str, str],
    fields: tuple[str, str, str, str, str, str],
) -> gtsam.Pose3 | None:
    values = [row.get(field, "") for field in fields]
    if any(value == "" for value in values):
        return None
    return pose3_from_components(*(float(value) for value in values))


OPTIMIZED_FIELDS = (
    "map_x",
    "map_y",
    "map_z",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
)
ODOM_FIELDS = (
    "odom_x",
    "odom_y",
    "odom_z",
    "odom_roll_deg",
    "odom_pitch_deg",
    "odom_yaw_deg",
)
PREDICTED_FIELDS = (
    "predicted_x",
    "predicted_y",
    "predicted_z",
    "predicted_roll_deg",
    "predicted_pitch_deg",
    "predicted_yaw_deg",
)
SCAN_FIELDS = (
    "scan_x",
    "scan_y",
    "scan_z",
    "scan_roll_deg",
    "scan_pitch_deg",
    "scan_yaw_deg",
)


def fill_pose(message_pose, pose: gtsam.Pose3) -> None:
    translation = pose.translation()
    quaternion = pose.rotation().toQuaternion()
    message_pose.position.x = float(translation[0])
    message_pose.position.y = float(translation[1])
    message_pose.position.z = float(translation[2])
    message_pose.orientation.x = float(quaternion.x())
    message_pose.orientation.y = float(quaternion.y())
    message_pose.orientation.z = float(quaternion.z())
    message_pose.orientation.w = float(quaternion.w())


class SavedPose3ResultPublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("indoor_saved_pose3_result_publisher")
        self.args = args
        self.rows = read_rows(args.csv)
        self.map_message = self.make_map_message()
        self.paths = self.make_paths()
        self.rejected_marker = self.make_rejected_marker()
        self.loop_markers = self.make_loop_markers()

        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_publisher = self.create_publisher(
            PointCloud2, args.map_topic, latched_qos
        )
        self.path_publishers = {
            name: self.create_publisher(PathMsg, topic, latched_qos)
            for name, topic in (
                ("optimized", args.optimized_topic),
                ("odometry", args.odometry_topic),
                ("predicted", args.predicted_topic),
                ("scan", args.scan_topic),
                ("pose2_aid", args.pose2_aid_topic),
            )
        }
        self.rejected_publisher = self.create_publisher(
            Marker, args.rejected_topic, latched_qos
        )
        self.loops_publisher = self.create_publisher(
            MarkerArray, args.loops_topic, latched_qos
        )

        self.publish_all()
        self.create_timer(2.0, self.publish_all)
        self.get_logger().info(
            "Publishing saved Pose3 result: "
            f"keyframes={len(self.rows)}, "
            f"optimized={len(self.paths['optimized'].poses)}, "
            f"odometry={len(self.paths['odometry'].poses)}, "
            f"predicted={len(self.paths['predicted'].poses)}, "
            f"scan_accepted={len(self.paths['scan'].poses)}, "
            f"pose2_aid={len(self.paths['pose2_aid'].poses)}, "
            f"rejected={len(self.rejected_marker.points)}"
        )

    def header(self) -> Header:
        header = Header()
        header.frame_id = self.args.map_frame
        header.stamp = self.get_clock().now().to_msg()
        return header

    def make_path(self, poses: list[gtsam.Pose3]) -> PathMsg:
        message = PathMsg()
        message.header = self.header()
        for pose in poses:
            stamped = PoseStamped()
            stamped.header = message.header
            fill_pose(stamped.pose, pose)
            message.poses.append(stamped)
        return message

    def make_paths(self) -> dict[str, PathMsg]:
        optimized = [
            pose
            for row in self.rows
            if (pose := pose_from_row(row, OPTIMIZED_FIELDS)) is not None
        ]
        predicted = [
            pose
            for row in self.rows
            if (pose := pose_from_row(row, PREDICTED_FIELDS)) is not None
        ]
        scans = [
            pose
            for row in self.rows
            if int(row["icp_accepted"])
            and (pose := pose_from_row(row, SCAN_FIELDS)) is not None
        ]

        odometry = [
            pose
            for row in self.rows
            if (pose := pose_from_row(row, ODOM_FIELDS)) is not None
        ]
        if odometry:
            initial = pose_from_row(self.rows[0], PREDICTED_FIELDS)
            if initial is None:
                initial = optimized[0]
            odom_anchor = odometry[0]
            odometry = [
                initial.compose(odom_anchor.between(pose)) for pose in odometry
            ]

        pose2_aid = []
        for row in self.rows:
            if not int(row["pose2_aid_used"]):
                continue
            predicted_pose = pose_from_row(row, PREDICTED_FIELDS)
            if predicted_pose is None:
                continue
            predicted_rpy = np.degrees(
                np.asarray(predicted_pose.rotation().rpy(), dtype=float)
            )
            pose2_aid.append(
                pose3_from_components(
                    float(row["pose2_aid_x"]),
                    float(row["pose2_aid_y"]),
                    float(predicted_pose.translation()[2]),
                    float(predicted_rpy[0]),
                    float(predicted_rpy[1]),
                    float(row["pose2_aid_yaw_deg"]),
                )
            )

        return {
            "optimized": self.make_path(optimized),
            "odometry": self.make_path(odometry),
            "predicted": self.make_path(predicted),
            "scan": self.make_path(scans),
            "pose2_aid": self.make_path(pose2_aid),
        }

    def make_map_message(self) -> PointCloud2:
        cloud = o3d.io.read_point_cloud(str(self.args.map_ply))
        if cloud.is_empty():
            raise ValueError(f"PLY contains no points: {self.args.map_ply}")
        points = np.asarray(cloud.points, dtype=np.float64)
        colors = (
            np.asarray(cloud.colors, dtype=np.float64)
            if cloud.has_colors()
            else None
        )
        visible = map_visualization_mask(
            points,
            self.args.map_publish_z_min,
            self.args.map_publish_z_max,
        )
        points = points[visible]
        if colors is not None:
            colors = colors[visible]
        if not len(points):
            raise ValueError("RViz map height filter removed every map point")

        maximum = self.args.map_publish_max_points
        if maximum > 0 and len(points) > maximum:
            rng = np.random.default_rng(self.args.random_seed)
            indices = rng.choice(len(points), size=maximum, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]

        header = self.header()
        if colors is None:
            message = point_cloud2.create_cloud_xyz32(
                header, np.asarray(points, dtype=np.float32)
            )
        else:
            fields = [
                PointField(
                    name="x", offset=0, datatype=PointField.FLOAT32, count=1
                ),
                PointField(
                    name="y", offset=4, datatype=PointField.FLOAT32, count=1
                ),
                PointField(
                    name="z", offset=8, datatype=PointField.FLOAT32, count=1
                ),
                PointField(
                    name="rgb", offset=12, datatype=PointField.UINT32, count=1
                ),
            ]
            packed = np.empty(
                len(points), dtype=point_cloud2.dtype_from_fields(fields)
            )
            packed["x"] = points[:, 0].astype(np.float32)
            packed["y"] = points[:, 1].astype(np.float32)
            packed["z"] = points[:, 2].astype(np.float32)
            rgb = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint32)
            packed["rgb"] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
            message = point_cloud2.create_cloud(header, fields, packed)
        self.get_logger().info(
            f"Prepared {len(points)} RViz map points at "
            f"Z=[{self.args.map_publish_z_min}, {self.args.map_publish_z_max}]; "
            "the source PLY was read only"
        )
        return message

    def make_rejected_marker(self) -> Marker:
        marker = Marker()
        marker.header = self.header()
        marker.ns = "indoor_pose3_rejected_scan_matches"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color = ColorRGBA(r=0.95, g=0.12, b=0.25, a=0.95)
        for row in self.rows:
            if int(row["icp_accepted"]):
                continue
            pose = pose_from_row(row, SCAN_FIELDS)
            if pose is None:
                continue
            xyz = pose.translation()
            marker.points.append(
                Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
            )
        return marker

    def make_loop_markers(self) -> MarkerArray:
        marker = Marker()
        marker.header = self.header()
        marker.ns = "indoor_pose3_6d_loops"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.color = ColorRGBA(r=1.0, g=0.1, b=0.9, a=1.0)
        optimized = [pose_from_row(row, OPTIMIZED_FIELDS) for row in self.rows]
        for index, row in enumerate(self.rows):
            if not int(row["loop_accepted"]) or row["loop_candidate"] == "":
                continue
            candidate = int(row["loop_candidate"])
            if candidate < 0 or candidate >= len(optimized):
                continue
            for pose in (optimized[candidate], optimized[index]):
                if pose is None:
                    continue
                xyz = pose.translation()
                marker.points.append(
                    Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
                )
        result = MarkerArray()
        result.markers.append(marker)
        return result

    def publish_all(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.map_message.header.stamp = stamp
        self.map_publisher.publish(self.map_message)
        for name, message in self.paths.items():
            message.header.stamp = stamp
            for pose in message.poses:
                pose.header.stamp = stamp
            self.path_publishers[name].publish(message)
        self.rejected_marker.header.stamp = stamp
        self.rejected_publisher.publish(self.rejected_marker)
        for marker in self.loop_markers.markers:
            marker.header.stamp = stamp
        self.loops_publisher.publish(self.loop_markers)


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--map-ply", type=Path, required=True)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--map-topic", default="/indoor_pose3/map_points")
    parser.add_argument("--optimized-topic", default="/indoor_pose3/path")
    parser.add_argument("--odometry-topic", default="/indoor_pose3/odometry_path")
    parser.add_argument("--predicted-topic", default="/indoor_pose3/predicted_path")
    parser.add_argument("--scan-topic", default="/indoor_pose3/scan_to_map_path")
    parser.add_argument("--pose2-aid-topic", default="/indoor_pose3/pose2_aid_path")
    parser.add_argument(
        "--rejected-topic", default="/indoor_pose3/rejected_scan_markers"
    )
    parser.add_argument("--loops-topic", default="/indoor_pose3/loops")
    parser.add_argument("--map-publish-z-min", type=float, default=-0.5)
    parser.add_argument("--map-publish-z-max", type=float, default=2.15)
    parser.add_argument("--map-publish-max-points", type=int, default=300000)
    parser.add_argument("--random-seed", type=int, default=7)
    args, ros_args = parser.parse_known_args()
    if not args.csv.is_file():
        parser.error(f"Pose3 CSV does not exist: {args.csv}")
    if not args.map_ply.is_file():
        parser.error(f"map PLY does not exist: {args.map_ply}")
    if args.map_publish_z_min > args.map_publish_z_max:
        parser.error("map-publish-z-min cannot exceed map-publish-z-max")
    return args, ros_args


def main() -> None:
    args, ros_args = parse_arguments()
    rclpy.init(args=ros_args)
    node = SavedPose3ResultPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
