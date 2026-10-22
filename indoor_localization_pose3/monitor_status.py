#!/usr/bin/env python3

"""Print live indoor Pose3 diagnostics in a compact human-readable form."""

from __future__ import annotations

import argparse
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class StatusMonitor(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("indoor_pose3_status_monitor")
        self.create_subscription(String, topic, self.callback, 10)
        self.get_logger().info(f"Monitoring {topic}")

    def callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            print(message.data, flush=True)
            return
        scan = status.get("scan_to_map", {})
        runtime = status.get("runtime", {})
        loops = status.get("loop_closure", {})
        pose = status.get("pose", {})
        deskew = status.get("deskew", {})
        print(
            "k={keyframes:4d} {state:>24s} "
            "xyz=({x:7.2f},{y:7.2f},{z:5.2f}) yaw={yaw:7.2f} "
            "ICP={accepted:4d}/{processed:4d} ({percentage:5.1f}%) "
            "fit={fitness:.3f} rmse={rmse} corr={corr:5d} "
            "deskew={deskew_status} loops={loop_accepted}/{loop_checks} "
            "p95={p95}".format(
                keyframes=int(status.get("keyframes", 0)),
                state=str(scan.get("status", status.get("state", "unknown"))),
                x=float(pose.get("x", 0.0)),
                y=float(pose.get("y", 0.0)),
                z=float(pose.get("z", 0.0)),
                yaw=float(pose.get("yaw_deg", 0.0)),
                accepted=int(scan.get("accepted", 0)),
                processed=int(scan.get("processed", 0)),
                percentage=float(scan.get("accepted_percentage", 0.0)),
                fitness=float(scan.get("fitness", 0.0)),
                rmse=(
                    "  n/a"
                    if scan.get("rmse") is None
                    else f"{float(scan['rmse']):.3f}"
                ),
                corr=int(scan.get("correspondences", 0)),
                deskew_status=str(deskew.get("last_status", "n/a")),
                loop_accepted=int(loops.get("accepted", 0)),
                loop_checks=int(loops.get("checks", 0)),
                p95=(
                    "n/a"
                    if runtime.get("processing_ms_p95") is None
                    else f"{float(runtime['processing_ms_p95']):.1f}ms"
                ),
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/indoor_pose3/status")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = StatusMonitor(args.topic)
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
