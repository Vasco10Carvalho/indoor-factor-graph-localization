#!/usr/bin/env python3
"""Republish saved indoor-localization trajectories for RViz comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMsg
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from indoor_scan_matcher import OccupancyMap


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def yaw_quaternion(message, yaw_deg: float) -> None:
    yaw = math.radians(yaw_deg)
    message.orientation.z = math.sin(yaw / 2.0)
    message.orientation.w = math.cos(yaw / 2.0)


class SavedResultPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("indoor_saved_result_publisher")
        self.args = args
        self.map = OccupancyMap(args.map_yaml)
        self.trajectory_rows = read_rows(args.trajectory_csv)
        self.odom_rows = read_rows(args.odom_csv)
        self.component_rows = read_rows(args.components_csv)
        if not self.trajectory_rows or not self.component_rows:
            raise RuntimeError("Saved trajectory files contain no poses")

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.output_publishers = {
            "map": self.create_publisher(OccupancyGrid, args.map_topic, latched_qos),
            "fused": self.create_publisher(PathMsg, args.fused_topic, latched_qos),
            "odom": self.create_publisher(PathMsg, args.odom_topic, latched_qos),
            "predicted": self.create_publisher(
                PathMsg, args.predicted_topic, latched_qos
            ),
            "scan": self.create_publisher(PathMsg, args.scan_topic, latched_qos),
            "rejected": self.create_publisher(
                Marker, args.rejected_topic, latched_qos
            ),
            "recoveries": self.create_publisher(
                Marker, args.recoveries_topic, latched_qos
            ),
            "loops": self.create_publisher(Marker, args.loops_topic, latched_qos),
            "segments": self.create_publisher(
                Marker, args.segments_topic, latched_qos
            ),
        }
        self.publish_all()
        self.create_timer(2.0, self.publish_all)
        accepted = sum(int(row["scan_accepted"]) for row in self.component_rows)
        self.get_logger().info(
            f"Publishing saved result: poses={len(self.trajectory_rows)}, "
            f"scan_accepted={accepted}, "
            f"scan_rejected={len(self.component_rows) - accepted}"
        )

    def stamp(self):
        return self.get_clock().now().to_msg()

    def make_path(
        self,
        rows: list[dict[str, str]],
        x_field: str,
        y_field: str,
        yaw_field: str,
    ) -> PathMsg:
        message = PathMsg()
        message.header.frame_id = self.args.map_frame
        message.header.stamp = self.stamp()
        for row in rows:
            if not row.get(x_field) or not row.get(y_field) or not row.get(yaw_field):
                continue
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(row[x_field])
            pose.pose.position.y = float(row[y_field])
            yaw_quaternion(pose.pose, float(row[yaw_field]))
            message.poses.append(pose)
        return message

    def make_map(self) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.frame_id = self.args.map_frame
        message.header.stamp = self.stamp()
        message.info.resolution = self.map.resolution
        message.info.width = self.map.width
        message.info.height = self.map.height
        message.info.origin.position.x = self.map.origin_x
        message.info.origin.position.y = self.map.origin_y
        message.info.origin.orientation.w = 1.0
        message.data = self.map.occupancy_data().tolist()
        return message

    def make_points_marker(
        self,
        rows: list[dict[str, str]],
        namespace: str,
        color: ColorRGBA,
        scale: float,
        use_scan_pose: bool,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.args.map_frame
        marker.header.stamp = self.stamp()
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = color
        prefix = "scan" if use_scan_pose else "optimized"
        marker.points = [
            Point(
                x=float(row[f"{prefix}_x"]),
                y=float(row[f"{prefix}_y"]),
                z=0.18,
            )
            for row in rows
        ]
        return marker

    def make_loops_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.args.map_frame
        marker.header.stamp = self.stamp()
        marker.ns = "indoor_dcs_loop_closures"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.10
        marker.color = ColorRGBA(r=1.0, g=0.55, b=0.05, a=0.9)
        for row_index, row in enumerate(self.component_rows):
            candidate_text = row.get("loop_candidate_index", "")
            if not candidate_text:
                continue
            candidate = int(candidate_text)
            if candidate >= len(self.component_rows):
                continue
            first = self.component_rows[candidate]
            marker.points.extend(
                (
                    Point(
                        x=float(first["optimized_x"]),
                        y=float(first["optimized_y"]),
                        z=0.15,
                    ),
                    Point(
                        x=float(row["optimized_x"]),
                        y=float(row["optimized_y"]),
                        z=0.15,
                    ),
                )
            )
        return marker

    def make_segments_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.args.map_frame
        marker.header.stamp = self.stamp()
        marker.ns = "indoor_factor_graph_segments"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.12
        marker.color = ColorRGBA(r=0.0, g=0.82, b=0.57, a=1.0)
        recovery_statuses = {"accepted_recovery", "recovered_amcl"}
        for index in range(1, len(self.component_rows)):
            if self.component_rows[index]["scan_status"] in recovery_statuses:
                continue
            previous = self.component_rows[index - 1]
            current = self.component_rows[index]
            marker.points.extend(
                (
                    Point(
                        x=float(previous["optimized_x"]),
                        y=float(previous["optimized_y"]),
                        z=0.08,
                    ),
                    Point(
                        x=float(current["optimized_x"]),
                        y=float(current["optimized_y"]),
                        z=0.08,
                    ),
                )
            )
        return marker

    def publish_all(self) -> None:
        accepted_rows = [
            row for row in self.component_rows if int(row["scan_accepted"])
        ]
        rejected_rows = [
            row for row in self.component_rows if not int(row["scan_accepted"])
        ]
        recovery_rows = [
            row
            for row in self.component_rows
            if row["scan_status"] in {"accepted_recovery", "recovered_amcl"}
        ]
        self.output_publishers["map"].publish(self.make_map())
        self.output_publishers["fused"].publish(
            self.make_path(self.trajectory_rows, "map_x", "map_y", "yaw_deg")
        )
        self.output_publishers["odom"].publish(
            self.make_path(self.odom_rows, "map_x", "map_y", "yaw_deg")
        )
        self.output_publishers["predicted"].publish(
            self.make_path(
                self.component_rows,
                "predicted_x",
                "predicted_y",
                "predicted_yaw_deg",
            )
        )
        self.output_publishers["scan"].publish(
            self.make_path(accepted_rows, "scan_x", "scan_y", "scan_yaw_deg")
        )
        self.output_publishers["rejected"].publish(
            self.make_points_marker(
                rejected_rows,
                "indoor_rejected_scan_matches",
                ColorRGBA(r=0.95, g=0.12, b=0.25, a=0.95),
                0.30,
                True,
            )
        )
        self.output_publishers["recoveries"].publish(
            self.make_points_marker(
                recovery_rows,
                "indoor_scan_recoveries",
                ColorRGBA(r=1.0, g=0.45, b=0.05, a=1.0),
                0.55,
                False,
            )
        )
        self.output_publishers["loops"].publish(self.make_loops_marker())
        self.output_publishers["segments"].publish(self.make_segments_marker())


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--odom-csv", type=Path, required=True)
    parser.add_argument("--components-csv", type=Path, required=True)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--map-topic", default="/indoor/map")
    parser.add_argument("--fused-topic", default="/indoor/factor_graph/path")
    parser.add_argument("--odom-topic", default="/indoor/odometry_only/path")
    parser.add_argument("--predicted-topic", default="/indoor/predicted/path")
    parser.add_argument("--scan-topic", default="/indoor/scan_to_map/path")
    parser.add_argument("--rejected-topic", default="/indoor/scan_to_map/rejected")
    parser.add_argument(
        "--recoveries-topic", default="/indoor/factor_graph/recovery_markers"
    )
    parser.add_argument("--loops-topic", default="/indoor/factor_graph/loops")
    parser.add_argument("--segments-topic", default="/indoor/factor_graph/segments")
    return parser.parse_known_args()


def main() -> None:
    args, ros_args = parse_arguments()
    rclpy.init(args=ros_args)
    node = SavedResultPublisher(args)
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
