#!/usr/bin/env python3
"""Record a ROS 2 odometry topic or TF transform to a portable trajectory CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re

from nav_msgs.msg import Odometry
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from comparison_common import quaternion_to_euler


FIELDS = (
    "stamp_ns",
    "source",
    "frame_id",
    "child_frame_id",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
)


class TrajectoryRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        source_slug = re.sub(r"[^a-z0-9_]", "_", args.source.lower().replace(" ", "_"))
        super().__init__(f"trajectory_recorder_{source_slug}")
        self.args = args
        args.output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = args.output.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.writer.writeheader()
        self.stream.flush()
        self.last_stamp_ns: int | None = None
        self.count = 0
        self.tf_buffer: Buffer | None = None
        self.tf_listener: TransformListener | None = None
        self.gate_open_at_ns: int | None = None

        if args.mode == "odometry":
            self.subscription = self.create_subscription(
                Odometry, args.topic, self.on_odometry, qos_profile_sensor_data
            )
            self.get_logger().info(f"Recording {args.topic} -> {args.output}")
        else:
            self.tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
            self.tf_listener = TransformListener(self.tf_buffer, self)
            if args.gate_topic:
                self.gate_subscription = self.create_subscription(
                    LaserScan, args.gate_topic, self.on_gate_scan, qos_profile_sensor_data
                )
            self.timer = self.create_timer(1.0 / args.rate, self.on_tf_timer)
            self.get_logger().info(
                f"Recording TF {args.world_frame} -> {args.child_frame} -> {args.output}"
            )

    def on_gate_scan(self, message: LaserScan) -> None:
        if self.gate_open_at_ns is not None:
            return
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        if stamp_ns > 0:
            self.gate_open_at_ns = stamp_ns + int(self.args.gate_delay * 1_000_000_000)

    def write_pose(
        self,
        stamp_ns: int,
        frame_id: str,
        child_frame_id: str,
        position,
        orientation,
    ) -> None:
        if stamp_ns <= 0 or stamp_ns == self.last_stamp_ns:
            return
        roll, pitch, yaw = quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        self.writer.writerow(
            {
                "stamp_ns": stamp_ns,
                "source": self.args.source,
                "frame_id": frame_id,
                "child_frame_id": child_frame_id,
                "x": f"{position.x:.9f}",
                "y": f"{position.y:.9f}",
                "z": f"{position.z:.9f}",
                "qx": f"{orientation.x:.12f}",
                "qy": f"{orientation.y:.12f}",
                "qz": f"{orientation.z:.12f}",
                "qw": f"{orientation.w:.12f}",
                "roll_deg": f"{math.degrees(roll):.9f}",
                "pitch_deg": f"{math.degrees(pitch):.9f}",
                "yaw_deg": f"{math.degrees(yaw):.9f}",
            }
        )
        self.stream.flush()
        self.last_stamp_ns = stamp_ns
        self.count += 1

    def on_odometry(self, message: Odometry) -> None:
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        self.write_pose(
            stamp_ns,
            message.header.frame_id,
            message.child_frame_id,
            message.pose.pose.position,
            message.pose.pose.orientation,
        )

    def on_tf_timer(self) -> None:
        assert self.tf_buffer is not None
        if self.args.gate_topic and self.gate_open_at_ns is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.world_frame, self.args.child_frame, Time()
            )
        except TransformException:
            return
        stamp_ns = int(transform.header.stamp.sec) * 1_000_000_000 + int(
            transform.header.stamp.nanosec
        )
        if self.gate_open_at_ns is not None and stamp_ns < self.gate_open_at_ns:
            return
        self.write_pose(
            stamp_ns,
            transform.header.frame_id,
            transform.child_frame_id,
            transform.transform.translation,
            transform.transform.rotation,
        )

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()
            self.get_logger().info(f"Saved {self.count} poses to {self.args.output}")


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("odometry", "tf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--topic", default="/odom")
    parser.add_argument("--world-frame", default="map")
    parser.add_argument("--child-frame", default="base_link")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument(
        "--gate-topic",
        default="",
        help="For TF mode, wait for the first LaserScan on this topic",
    )
    parser.add_argument(
        "--gate-delay",
        type=float,
        default=0.0,
        help="Additional simulated seconds after the first gate scan",
    )
    args, ros_args = parser.parse_known_args()
    if args.rate <= 0.0:
        parser.error("--rate must be positive")
    if args.gate_delay < 0.0:
        parser.error("--gate-delay must be non-negative")
    return args, ros_args


def main() -> None:
    args, ros_args = parse_arguments()
    rclpy.init(args=ros_args)
    node = TrajectoryRecorder(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
