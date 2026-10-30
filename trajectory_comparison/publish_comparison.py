#!/usr/bin/env python3
"""Publish a saved multi-method comparison map and trajectories for RViz."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re

import cv2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMessage
import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from comparison_common import load_map_metadata


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "trajectory"


class ComparisonPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("trajectory_comparison_publisher")
        self.args = args
        self.metadata = load_map_metadata(args.map_yaml)
        self.rows_by_source: dict[str, list[dict[str, str]]] = {}
        with args.trajectories.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                self.rows_by_source.setdefault(row["source"], []).append(row)
        if not self.rows_by_source:
            raise RuntimeError(f"No trajectories found in {args.trajectories}")

        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_publisher = self.create_publisher(
            OccupancyGrid, "/trajectory_comparison/map", self.qos
        )
        self.path_publishers = {
            source: self.create_publisher(
                PathMessage, f"/trajectory_comparison/{slug(source)}/path", self.qos
            )
            for source in self.rows_by_source
        }
        self.publish_all()
        self.timer = self.create_timer(2.0, self.publish_all)
        topics = ", ".join(
            f"/trajectory_comparison/{slug(source)}/path"
            for source in self.rows_by_source
        )
        self.get_logger().info(f"Comparison ready. Path topics: {topics}")

    def stamp(self):
        return self.get_clock().now().to_msg()

    def make_map(self) -> OccupancyGrid:
        image = cv2.imread(str(self.metadata.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read {self.metadata.image_path}")
        probability = image.astype(np.float32) / 255.0
        if not self.metadata.negate:
            probability = 1.0 - probability
        occupancy = np.full(image.shape, -1, dtype=np.int8)
        occupancy[probability > self.metadata.occupied_thresh] = 100
        occupancy[probability < self.metadata.free_thresh] = 0

        message = OccupancyGrid()
        message.header.frame_id = self.args.frame
        message.header.stamp = self.stamp()
        message.info.resolution = self.metadata.resolution
        message.info.width = image.shape[1]
        message.info.height = image.shape[0]
        message.info.origin.position.x = self.metadata.origin_x
        message.info.origin.position.y = self.metadata.origin_y
        message.info.origin.orientation.z = math.sin(self.metadata.origin_yaw / 2.0)
        message.info.origin.orientation.w = math.cos(self.metadata.origin_yaw / 2.0)
        message.data = np.flipud(occupancy).reshape(-1).astype(int).tolist()
        return message

    def make_path(self, rows: list[dict[str, str]]) -> PathMessage:
        message = PathMessage()
        message.header.frame_id = self.args.frame
        message.header.stamp = self.stamp()
        for row in rows:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(row["x"])
            pose.pose.position.y = float(row["y"])
            yaw = float(row["yaw_rad"])
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            message.poses.append(pose)
        return message

    def publish_all(self) -> None:
        self.map_publisher.publish(self.make_map())
        for source, rows in self.rows_by_source.items():
            self.path_publishers[source].publish(self.make_path(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--frame", default="map")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = ComparisonPublisher(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
