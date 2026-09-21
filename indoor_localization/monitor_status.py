#!/usr/bin/env python3
"""Print compact indoor-localization diagnostics without truncating JSON."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def formatted_rate(values: dict, key: str) -> str:
    return f"{float(values.get(key, 0.0)):5.1f}"


class StatusMonitor(Node):
    def __init__(self) -> None:
        super().__init__("indoor_factor_graph_monitor")
        self.create_subscription(
            String,
            "/indoor/factor_graph/status",
            self.status_callback,
            10,
        )
        self.get_logger().info("Waiting for /indoor/factor_graph/status")

    @staticmethod
    def status_callback(message: String) -> None:
        status = json.loads(message.data)
        source = status.get("source_hz", {})
        arrival = status.get("arrival_hz", {})
        match = status.get("last_match") or {}
        loops = status.get("loop_closure") or {}
        reasons = status.get("rejection_reasons") or {}
        reason_text = ",".join(
            f"{name}:{count}" for name, count in sorted(reasons.items())
        ) or "none"
        print(
            f"state={status.get('state')} k={status.get('keyframes', 0)} "
            f"source[odom/robot_imu/ouster/scan]="
            f"{formatted_rate(source, 'local_odometry')}/"
            f"{formatted_rate(source, 'robot_imu')}/"
            f"{formatted_rate(source, 'lidar_imu')}/"
            f"{formatted_rate(source, 'scan')}Hz "
            f"arrival={formatted_rate(arrival, 'local_odometry')}/"
            f"{formatted_rate(arrival, 'robot_imu')}/"
            f"{formatted_rate(arrival, 'lidar_imu')}/"
            f"{formatted_rate(arrival, 'scan')}Hz "
            f"processed={float(status.get('processed_hz', 0.0)):4.1f}Hz "
            f"p95={float(status.get('processing_ms_p95') or 0.0):5.1f}ms "
            f"ok/rej/drop={status.get('accepted', 0)}/"
            f"{status.get('rejected', 0)}/{status.get('dropped_busy', 0)} "
            f"loops[check/ok]={loops.get('checks', 0)}/"
            f"{loops.get('accepted', 0)} "
            f"match[inliers/dist]={float(match.get('inlier_ratio', 0.0)):.2f}/"
            f"{float(match.get('mean_distance_m', 0.0)):.2f}m "
            f"reject_reasons={reason_text}",
            flush=True,
        )


def main() -> None:
    rclpy.init()
    node = StatusMonitor()
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
